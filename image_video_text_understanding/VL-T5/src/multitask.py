# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import logging
import os
import pickle
from copy import deepcopy
from pathlib import Path
from pprint import pformat
from typing import Optional

import multitask_data
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import wandb
from adapters import AdapterController, MetaLayersAdapterController
from ddp_fix import ddp_forward
from dist_utils import reduce_dict
from packaging import version
from param import parse_args
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from trainer_base import TrainerBase
from utils import LossMeter, set_global_logging_level, path_print
from multitask_model import VLBartMultiTask, VLT5MultiTask
from vis_encoder import get_vis_encoder

proj_dir = Path(__file__).resolve().parent.parent

class Trainer(TrainerBase):
    def __init__(self, args, train_loader=None, val_loader=None, test_loader=None, train=True):
        super().__init__(
            args,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            train=train)

        if not self.verbose:
            set_global_logging_level(logging.ERROR, ["transformers"])

        model_kwargs = {}
        if 't5' in args.backbone:
            model_class = VLT5MultiTask
        elif 'bart' in args.backbone:
            model_class = VLBartMultiTask

        config = self.create_config()
        self.tokenizer = self.create_tokenizer()

        if 'bart' in self.args.tokenizer:
            num_added_toks = 0
            if config.use_vis_order_embedding:
                additional_special_tokens = [f'<extra_id_{i}>' for i in range(100-1, -1, -1)] + \
                        [f'<vis_extra_id_{i}>' for i in range(100-1, -1, -1)]
                special_tokens_dict = {'additional_special_tokens': additional_special_tokens}
                num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)

                config.default_obj_order_ids = self.tokenizer.convert_tokens_to_ids([f'<vis_extra_id_{i}>' for i in range(100)])

        self.model = self.create_model(model_class, config, **model_kwargs)

        if 't5' in self.args.tokenizer:
            self.model.resize_token_embeddings(self.tokenizer.vocab_size)
        elif 'bart' in self.args.tokenizer:
            self.model.resize_token_embeddings(self.model.model.shared.num_embeddings + num_added_toks)

        self.model.tokenizer = self.tokenizer
        if 't5' in self.args.tokenizer or 'bart' in self.args.tokenizer:
            self.model.true_id = self.tokenizer('true', add_special_tokens=False).input_ids[0]
            self.model.false_id = self.tokenizer('false', add_special_tokens=False).input_ids[0]

        if self.include_vis_encoder:
            # train vision encoder end-to-end
            vis_encoder_type = self.args.feature_type.split("_")[-1]

            if self.args.use_vis_adapter:
                self.vis_encoder = get_vis_encoder(
                    backbone=vis_encoder_type, 
                    image_size=eval(self.args.image_size)[0],
                    adapter_type=self.args.vis_adapter_type,
                    reduction_factor=self.args.vis_reduction_factor,
                    use_bn=not self.args.remove_bn_vis_adapter,
                )
            else:
                self.vis_encoder = get_vis_encoder(
                    backbone=vis_encoder_type, 
                    image_size=eval(self.args.image_size)[0],
                    adapter_type=None,
                )

            print("include vision encoder")
            self.model.vis_encoder = self.vis_encoder
        # Load Checkpoint
        self.start_epoch = None
        if args.load is not None:
            ckpt_path = args.load
            self.load_checkpoint(ckpt_path)
        if self.args.from_scratch:
            self.init_weights()

        # GPU Options
        print(f'Model Launching at GPU {self.args.gpu}')
        if self.verbose:
            from time import time
            start = time()
        self.model = self.model.to(args.gpu)

        self.freeze_whole_model() # freeze whole parameters first
        self.unfreeze_parameters() # unfreeze selected parameters
        if self.verbose:
            print(self.model)
        self.percent_updated_parameters = self.print_trainable_params_percentage(self.model)

        # Optimizer
        if train:
            self.optim, self.lr_scheduler = self.create_optimizer_and_scheduler()

            if self.args.fp16:
                self.scaler = torch.cuda.amp.GradScaler()

        if args.multiGPU:
            if args.distributed:
                self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
                self.model = DDP(self.model, device_ids=[args.gpu],
                                 find_unused_parameters=False
                                 )
        if self.verbose:
            print(f'It took {time() - start:.1f}s')

    def train(self):
        if self.verbose:
            vqa_loss_meter = LossMeter()
            refcoco_loss_meter = LossMeter()
            
            quesid2ans = {}
            best_vqa_valid = 0.
            best_vqa_epoch = 0

            # gqa
            best_gqa_valid = 0
            best_gqa_epoch = 0

            # nlvr
            best_nlvr_valid = 0
            best_nlvr_epoch = 0

            # vcr
            best_valid_Q_AR = 0
            best_vcr_epoch = 0

            # refcoco
            best_refcoco_valid = 0
            best_refcoco_epoch = 0

            # caption
            best_caption_valid = 0
            best_caption_epoch = 0

            # mmt
            best_mmt_valid = 0
            best_mmt_epoch = 0

            # classification
            best_cls_valid = 0
            best_cls_epoch = 0

            wandb.init(project=self.args.project_name)
            wandb.run.name = self.args.run_name
            wandb.config.update(self.args)
            wandb.watch(self.model)
            wandb.run.summary["percent of updated parameters (%)"] = self.percent_updated_parameters

            src_dir = Path(__file__).resolve().parent
            base_path = str(src_dir.parent)
            src_dir = str(src_dir)
            wandb.save(os.path.join(src_dir + "/*.py"), base_path=base_path)

        global_step = 0

        for epoch in range(self.args.epochs):
            if self.start_epoch is not None:
                epoch += self.start_epoch
            self.model.train()
            self.partial_eval()

            if self.args.distributed:
                self.train_loader.set_epoch(epoch)
            if self.verbose:
                pbar = tqdm(total=len(self.train_loader), ncols=250)

            epoch_results = {
                'loss': 0.,
            }

            task_counter = {
                'vqa': 0,
                'gqa': 0,
                'nlvr': 0,
                'refcoco': 0,
                'vcr': 0,
                'caption': 0,
                'mmt': 0,
                'cls': 0,
            }

            # vqa
            quesid2ans = {}
            train_acc = 0.

            # refcoco
            n_correct = 0
            n_total = 0


            for step_i, batch in enumerate(self.train_loader):
                task = batch['task']
                task_counter[task] += 1
                batch['log_train_accuracy'] = self.args.log_train_accuracy

                if self.args.fp16:
                    with autocast():
                        if self.args.distributed:
                            results = ddp_forward(self.model, batch)
                        else:
                            results = self.model.train_step(batch)
                else:
                    if self.args.distributed:
                        results = ddp_forward(self.model, batch)
                    else:
                        results = self.model.train_step(batch)

                loss = results['loss']

                if self.args.track_z:
                    reg_loss = 0
                    layer_num = 0
                    for name, sub_module in self.model.named_modules():
                        if isinstance(sub_module, (AdapterController)):
                            reg_loss += ((sub_module.adapters[task].z) ** 2).mean()
                            layer_num += 1

                        if isinstance(sub_module, (MetaLayersAdapterController)):
                            reg_loss += ((sub_module.z) ** 2).mean()
                            layer_num += 1

                    reg_loss = reg_loss / layer_num

                    loss = loss + self.args.lambda_z * reg_loss

                if self.args.fp16:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                loss = loss.detach()

                # Update Parameters
                if self.args.clip_grad_norm > 0:
                    if self.args.fp16:
                        self.scaler.unscale_(self.optim)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.args.clip_grad_norm)

                if self.args.fp16:
                    self.scaler.step(self.optim)
                    self.scaler.update()
                else:
                    self.optim.step()

                if self.lr_scheduler:
                    self.lr_scheduler.step()
                
                self.optim.zero_grad(set_to_none=True)

                global_step += 1

                for k, v in results.items():
                    if k in epoch_results:
                        epoch_results[k] += v.item()

                if self.lr_scheduler:
                    lr = self.lr_scheduler.get_last_lr()[0]
                else:
                    try:
                        lr = self.optim.get_lr()[0]
                    except AttributeError:
                        lr = self.args.lr

                if self.args.log_train_accuracy and task == 'refcoco':
                    correct = results['correct']
                    n_correct += sum(correct)
                    n_total += len(correct)

                if self.verbose:
                    if task == 'vqa':
                        vqa_loss_meter.update(loss.item())
                    elif task == 'refcoco':
                        refcoco_loss_meter.update(loss.item())

                    desc_str = f'Epoch {epoch} | LR {lr:.6f}'

                    desc_str += f" |"
                    if 'vqa' in self.args.tasks:
                        desc_str += f" VQA {task_counter['vqa']}"
                    if 'gqa' in self.args.tasks:
                        desc_str += f" GQA {task_counter['gqa']}"
                    if 'nlvr' in self.args.tasks:
                        desc_str += f" NLVR {task_counter['nlvr']}"
                    if 'vcr' in self.args.tasks:
                        desc_str += f" VCR {task_counter['vcr']}"
                    if 'refcoco' in self.args.tasks:
                        desc_str += f" RefCOCOg {task_counter['refcoco']}"
                    if 'caption' in self.args.tasks:
                        desc_str += f" COCO {task_counter['caption']}"
                    if 'mmt' in self.args.tasks:
                        desc_str += f" MMT {task_counter['mmt']}"
                    if 'cls' in self.args.tasks:
                        desc_str += f" CLS {task_counter['cls']}"

                    if len(vqa_loss_meter) > 0:
                        desc_str += f' | VQA Loss {vqa_loss_meter.val:4f}'
                    if len(refcoco_loss_meter) > 0:
                        desc_str += f' | RefCOCOg Loss {refcoco_loss_meter.val:.3f}'

                    if self.args.log_train_accuracy and n_total > 0:
                        desc_str += f' | RefCOCOg Acc'
                        desc_str += f' Correct {n_correct:.0f}'
                        desc_str += f' (Acc {n_correct/n_total*100:.1f}%)'

                    pbar.set_description(desc_str)
                    pbar.update(1)

            if self.verbose:
                pbar.close()

            if self.args.log_train_accuracy:
                train_score_dict = {
                    'n_correct': n_correct,
                    'n_total': n_total
                }
                train_score_dict = reduce_dict(train_score_dict, self.args.gpu)

            # Validation
            log_str = ''
            wandb_log_dict = {}

            if 'vqa' in self.args.tasks:
                # VQA
                vqa_val_loader = self.val_loader['vqa']
                score_dict = self.vqa_evaluate(vqa_val_loader)
                if self.args.gpu == 0:
                    valid_score = score_dict['topk_score'] * 100.
                    valid_score_raw = score_dict['overall']
                    if valid_score_raw > best_vqa_valid or epoch == 0:
                        best_vqa_valid = valid_score_raw
                        best_vqa_epoch = epoch
                        # self.save("VQA_BEST")
                    log_str += f"VQA"
                    log_str += "\nEpoch %d: Valid Raw %0.2f Topk %0.2f" % (epoch, valid_score_raw, valid_score)
                    log_str += "\nEpoch %d: Best Raw %0.2f\n" % (best_vqa_epoch, best_vqa_valid)
                    wandb_log_dict['VQA/Valid/score'] = valid_score
                    wandb_log_dict['VQA/Valid/raw_score'] = score_dict['overall']
            if 'gqa' in self.args.tasks:
                # GQA
                gqa_val_loader = self.val_loader['gqa']
                valid_score = self.gqa_evaluate(gqa_val_loader)
                if self.args.gpu == 0:
                    valid_score *= 100
                    if valid_score > best_gqa_valid or epoch == 0:
                        best_gqa_valid = valid_score
                        best_gqa_epoch = epoch
                    wandb_log_dict['GQA/Valid/Acc'] = valid_score
                    log_str += f"GQA"
                    log_str += "\nEpoch %d: Valid %0.2f" % (epoch, valid_score)
                    log_str += "\nEpoch %d: Best %0.2f\n" % (best_gqa_epoch, best_gqa_valid)
            if 'nlvr' in self.args.tasks:
                # NLVR
                nlvr_val_loader = self.val_loader['nlvr']
                valid_score_dict = self.nlvr_evaluate(nlvr_val_loader)
                if self.args.gpu == 0:
                    valid_acc = valid_score_dict['accuracy'] * 100.
                    if valid_acc > best_nlvr_valid or epoch == 0:
                        best_nlvr_valid = valid_acc
                        best_nlvr_epoch = epoch
                    wandb_log_dict['NLVR/Valid/Acc'] = valid_acc
                    log_str += f"NLVR"
                    log_str += "\nEpoch %d: Valid %0.2f" % (epoch, valid_acc)
                    log_str += "\nEpoch %d: Best %0.2f\n" % (best_nlvr_epoch, best_nlvr_valid)

            if 'vcr' in self.args.tasks:
                # VCR
                vcr_val_loader = self.val_loader['vcr']
                valid_score_dict = self.vcr_evaluate(vcr_val_loader)
                if self.args.gpu == 0:
                    valid_Q_A = valid_score_dict['Q_A']/valid_score_dict['n_total'] * 100
                    valid_QA_R = valid_score_dict['QA_R']/valid_score_dict['n_total'] * 100
                    valid_Q_AR = valid_score_dict['Q_AR']/valid_score_dict['n_total'] * 100
                    valid_n_total = int(valid_score_dict['n_total'])
                    if valid_Q_AR > best_valid_Q_AR or epoch == 0:
                        best_valid_Q_AR = valid_Q_AR
                        best_vcr_epoch = epoch
                    wandb_log_dict['VCR/Valid/Q_A'] = valid_Q_A
                    wandb_log_dict['VCR/Valid/QA_R'] = valid_QA_R
                    wandb_log_dict['VCR/Valid/Q_AR'] = valid_Q_AR
                    log_str += f"VCR"
                    log_str += "\nEpoch %d: Valid %0.2f" % (epoch, valid_Q_AR)
                    log_str += "\nEpoch %d: Best %0.2f\n" % (best_vcr_epoch, best_valid_Q_AR)

            if 'refcoco' in self.args.tasks:
                # RefCOCO
                refcoco_val_loader = self.val_loader['refcoco']
                valid_score_dict = self.refcoco_evaluate(refcoco_val_loader)
                if self.args.gpu == 0:
                    valid_acc = valid_score_dict['n_correct']/valid_score_dict['n_total'] * 100
                    valid_n_correct = int(valid_score_dict['n_correct'])
                    valid_n_total = int(valid_score_dict['n_total'])
                    if valid_acc > best_refcoco_valid or epoch == 0:
                        best_refcoco_valid = valid_acc
                        best_refcoco_epoch = epoch
                    if self.args.log_train_accuracy:
                        train_acc = train_score_dict['n_correct']/train_score_dict['n_total'] * 100
                        train_n_correct = int(train_score_dict['n_correct'])
                        train_n_total = int(train_score_dict['n_total'])
                        wandb_log_dict['RefCOCO/Train/Acc'] = train_acc
                    wandb_log_dict['RefCOCO/Valid/Acc'] = valid_acc
                    log_str += f"RefCOCOg"
                    if self.args.log_train_accuracy:
                        log_str += f"\nEpoch {epoch}: Train"
                        log_str += f" Acc {train_acc:.2f}% |"
                        log_str += f" # correct {train_n_correct} # total {train_n_total}"
                    log_str += f"\nEpoch {epoch}: Valid"
                    log_str += f" Acc {valid_acc:.2f}% |"
                    log_str += f" # correct {valid_n_correct} # total {valid_n_total}"
                    log_str += f"\nEpoch {best_refcoco_epoch}: Best Acc {best_refcoco_valid:.2f}%\n"

            if 'caption' in self.args.tasks:
                # COCO Caption
                caption_val_loader = self.val_loader['caption']
                valid_results = self.caption_evaluate(caption_val_loader)
                if self.args.gpu == 0:
                    valid_score = valid_results['CIDEr'] * 100
                    if valid_score > best_caption_valid or epoch == 0:
                        best_caption_valid = valid_score
                        best_caption_epoch = epoch
                    for score_name, score in valid_results.items():
                        wandb_log_dict[f'Caption/Valid/{score_name}'] = score * 100
                    log_str += f"COCO Caption"
                    log_str += "\nEpoch %d: Valid CIDEr %0.2f" % (epoch, valid_score)
                    log_str += "\nEpoch %d: Best %0.2f\n" % (best_caption_epoch, best_caption_valid)

            if 'mmt' in self.args.tasks:
                # MMT
                mmt_val_loader = self.val_loader['mmt']
                valid_results = self.mmt_evaluate(mmt_val_loader)
                if self.args.gpu == 0:
                    valid_score = valid_results['BLEU']
                    if valid_score > best_mmt_valid:
                        best_mmt_valid = valid_score
                        best_mmt_epoch = epoch
                    for score_name, score in valid_results.items():
                        wandb_log_dict[f'MMT/Valid/{score_name}'] = score
                    log_str += f"Multi30K En-De"
                    log_str += "\nEpoch %d: Valid BLEU %0.2f" % (epoch, valid_score)
                    log_str += "\nEpoch %d: Best %0.2f\n" % (best_mmt_epoch, best_mmt_valid)

            if 'cls' in self.args.tasks:
                cls_val_loader = self.val_loader['cls']
                valid_results = self.cls_evaluate(cls_val_loader)
                if self.args.gpu == 0:
                    valid_score = valid_results['overall']
                    if valid_score > best_cls_valid:
                        best_cls_valid = valid_score
                        best_cls_epoch = epoch
                    for score_name, score in valid_results.items():
                        wandb_log_dict[f'CLS/Valid/{score_name}'] = score
                    log_str += f"TinyImagenet"
                    log_str += "\nEpoch %d: Top1 %0.2f" % (epoch, valid_score)
                    log_str += "\nEpoch %d: Best %0.2f\n" % (best_cls_epoch, best_cls_valid)

            if self.verbose:
                wandb.log(wandb_log_dict, step=epoch)
                print(log_str)

        # Test Set
        if self.args.gpu == 0:
            self.save("LAST")
            log_str = ''
            wandb_log_dict = {}

        if 'vqa' in self.args.tasks:
            # VQA
            vqa_test_loader = self.test_loader['vqa']
            dump_path = os.path.join(self.args.output, 'karpathy_test_predict.json')
            quesid2ans = self.vqa_predict(vqa_test_loader, dump_path)
            if self.args.gpu == 0:
                wandb.save(dump_path, base_path=self.args.output)
                evaluator = vqa_test_loader.evaluator
                acc_dict_all = evaluator.evaluate_raw(quesid2ans)
                acc_dict_answerable = evaluator.evaluate_raw(quesid2ans, is_topk_optimal=True)
                acc_dict_unanswerable = evaluator.evaluate_raw(quesid2ans, is_topk_optimal=False)

                wandb_log_dict['VQA/Test/overall'] = acc_dict_all['overall']
                wandb_log_dict['VQA/Test/topk_optimal'] = acc_dict_answerable['overall']
                wandb_log_dict['VQA/Test/topk_not_optimal'] = acc_dict_unanswerable['overall']

            if self.test_loader.get("vqa_submit", None):
                vqa_submit_test_loader = self.test_loader['vqa_submit']
                dump_path = os.path.join(self.args.output, 'vqa_submit.json')
                self.vqa_predict(vqa_submit_test_loader, dump_path=dump_path)
                if self.args.gpu == 0:
                    wandb.save(dump_path, base_path=self.args.output)

        if 'nlvr' in self.args.tasks:
            # NLVR
            nlvr_test_loader = self.test_loader['nlvr']
            dump_path = os.path.join(self.args.output, 'nlvr_submit.csv')
            test_score_dict = self.nlvr_evaluate(nlvr_test_loader, dump_path=dump_path)
            if self.args.gpu == 0:
                wandb.save(dump_path, base_path=self.args.output)
                for score_name, score in test_score_dict.items():
                    wandb_log_dict[f'NLVR/Test/{score_name}'] = score * 100.
        if 'refcoco' in self.args.tasks:
            # RefCOCO
            refcoco_test_loader = self.test_loader['refcoco']
            test_score_dict = self.refcoco_evaluate(refcoco_test_loader)
            if self.args.gpu == 0:
                test_acc = test_score_dict['n_correct'] / test_score_dict['n_total'] * 100
                test_n_correct = int(test_score_dict['n_correct'])
                test_n_total = int(test_score_dict['n_total'])
                wandb_log_dict['RefCOCO/test/Acc'] = test_acc
                log_str = 'RefCOCOg'
                log_str += f"\nTest Acc {test_acc:.2f}%"
                log_str += f"\nTest # correct {test_n_correct} # total {test_n_total}"
        if 'caption' in self.args.tasks:
            # COCO Caption
            caption_test_loader = self.test_loader['caption']
            test_results = self.caption_evaluate(caption_test_loader)
            if self.args.gpu == 0:
                for score_name, score in test_results.items():
                    wandb_log_dict[f'Caption/Test/{score_name}'] = score

        if 'mmt' in self.args.tasks:
            # MMT
            mmt_test2016_loader = self.test_loader['mmt_test2016']
            mmt_test2017_loader = self.test_loader['mmt_test2017']
            mmt_test2018_loader = self.test_loader['mmt_test2018']
            for loader in [mmt_test2016_loader, mmt_test2017_loader, mmt_test2018_loader]:
                split = loader.dataset.source
                dump_path = os.path.join(self.args.output, f'submit_{split}_raw.txt')
                test_results = self.mmt_evaluate(loader, dump_path=dump_path)
                if self.args.gpu == 0:
                    for score_name, score in test_results.items():
                        wandb_log_dict[f'MMT/{split}/{score_name}'] = score
                    log_str += f'{split} set results\n'
                    log_str += pformat(test_results)

        if self.verbose:
            print(log_str)
            wandb.log(wandb_log_dict, step=self.args.epochs)
            wandb.run.summary['finished'] = True

        if self.args.distributed:
            dist.destroy_process_group()

    def vqa_predict(self, loader, dump_path=None):
        self.model.eval()
        with torch.no_grad():
            quesid2ans = {}

            gen_kwargs = {}
            gen_kwargs['num_beams'] = 1

            for i, batch in enumerate(tqdm(loader, ncols=150, desc="VQA Validation", disable=not self.verbose)):

                if self.args.distributed:
                    results = self.model.module.test_step(batch, **gen_kwargs)
                else:
                    results = self.model.test_step(batch, **gen_kwargs)

                pred_ans = results['pred_ans']
                ques_ids = batch['question_ids']

                for qid, ans in zip(ques_ids, pred_ans):
                    quesid2ans[qid] = ans

            if self.args.distributed:
                # convert to list
                list_qid_ans = list(quesid2ans.items())
                # collect
                all_qid_ans = self.collect_results_gpu(list_qid_ans, len(loader.dataset))
                # convert back to dict
                if self.args.gpu == 0:
                    quesid2ans = dict(all_qid_ans)

            if self.args.gpu == 0 and dump_path is not None:
                loader.evaluator.dump_result(quesid2ans, dump_path)
            return quesid2ans

    def vqa_evaluate(self, loader, dump_path=None):
        quesid2ans = self.vqa_predict(loader, dump_path)

        if self.args.gpu == 0:
            evaluator = loader.evaluator
            acc_dict = evaluator.evaluate_raw(quesid2ans)

            topk_score = evaluator.evaluate(quesid2ans)
            acc_dict['topk_score'] = topk_score

            return acc_dict

    def gqa_predict(self, loader, dump_path=None):
        self.model.eval()
        with torch.no_grad():
            quesid2ans = {}

            gen_kwargs = {}
            gen_kwargs['num_beams'] = 1

            if self.verbose:
                pbar = tqdm(total=len(loader), ncols=150, desc="GQA Validation")

            for i, batch in enumerate(loader):

                if self.args.distributed:
                    results = self.model.module.test_step(batch, **gen_kwargs)
                else:
                    results = self.model.test_step(batch, **gen_kwargs)

                pred_ans = results['pred_ans']
                ques_ids = batch['question_ids']

                for qid, ans in zip(ques_ids, pred_ans):
                    quesid2ans[qid] = ans

                if self.verbose:
                    pbar.update(1)

            if self.args.distributed:
                # convert to list
                list_qid_ans = list(quesid2ans.items())
                # collect
                all_qid_ans = self.collect_results_gpu(list_qid_ans, len(loader.dataset))
                # convert back to dict
                if self.args.gpu == 0:
                    quesid2ans = dict(all_qid_ans)

            if self.args.gpu == 0 and dump_path is not None:
                print('\nsave dump at', dump_path)
                loader.evaluator.dump_result(quesid2ans, dump_path)
            return quesid2ans

    def gqa_evaluate(self, loader, dump_path=None):
        quesid2ans = self.gqa_predict(loader, dump_path)

        if self.args.gpu == 0:
            evaluator = loader.evaluator
            return evaluator.evaluate(quesid2ans)

    def nlvr_predict(self, loader, dump_path=None):
        """
        Predict the answers to questions in a data split.
        :param eval_tuple: The data tuple to be evaluated.
        :param dump: The path of saved file to dump results.
        :return: A dict of question_id to answer.
        """
        self.model.eval()
        with torch.no_grad():
            quesid2ans = {}
            for i, batch in enumerate(tqdm(loader, ncols=150, desc="NLVR Prediction", disable=not self.verbose)):

                if self.args.distributed:
                    results = self.model.module.test_step(batch)
                else:
                    results = self.model.test_step(batch)

                pred_ans = results['pred_ans_id']
                ques_ids = batch['question_ids']

                for qid, ans in zip(ques_ids, pred_ans):
                    quesid2ans[qid] = ans

            if self.args.distributed:
                # convert to list
                list_qid_ans = list(quesid2ans.items())
                # collect
                all_qid_ans = self.collect_results_gpu(list_qid_ans, len(loader.dataset))
                # convert back to dict
                if self.args.gpu == 0:
                    quesid2ans = dict(all_qid_ans)

            if self.args.gpu == 0 and dump_path is not None:
                loader.evaluator.dump_result(quesid2ans, dump_path)
            return quesid2ans

    def nlvr_evaluate(self, loader, dump_path=None):
        quesid2ans = self.nlvr_predict(loader, dump_path)

        if self.args.gpu == 0:
            evaluator = loader.evaluator
            return evaluator.evaluate(quesid2ans)

    def refcoco_evaluate(self, loader):
        self.model.eval()
        with torch.no_grad():

            n_correct = 0
            n_total = 0

            if self.verbose:
                iterator = tqdm(loader, ncols=150, desc="RefCOCOg Validation")
            else:
                iterator = loader

            for i, batch in enumerate(iterator):

                if self.args.distributed:
                    results = self.model.module.test_step(batch)
                else:
                    results = self.model.test_step(batch)

                correct = results['correct']
                n_correct += sum(correct)
                n_total += len(correct)

        if self.args.distributed:
            numbers = torch.tensor([n_correct, n_total], device=self.args.gpu)
            dist.all_reduce(numbers, op=dist.ReduceOp.SUM)
            n_correct, n_total = numbers.tolist()

        score_dict = {}
        score_dict['n_correct'] = n_correct
        score_dict['n_total'] = n_total

        return score_dict

    def vcr_evaluate(self, loader):
        self.model.eval()
        with torch.no_grad():

            score_dict = {}
            Q_A_results = 0
            QA_R_results = 0
            Q_AR_results = 0
            n_total = 0

            if self.verbose:
                iterator = tqdm(loader, ncols=150, desc="VCR Validation")
            else:
                iterator = loader

            for i, batch in enumerate(iterator):

                if self.args.distributed:
                    results = self.model.module.valid_step(batch)
                else:
                    results = self.model.vaid_step(batch)

                qa_pred = results['qa_pred']
                qar_pred = results['qar_pred']

                a_labels = batch['answer_labels']
                r_labels = batch['rationale_labels']

                Q_A_correct = a_labels == qa_pred
                QA_R_correct = r_labels == qar_pred
                Q_AR_correct = Q_A_correct & QA_R_correct

                Q_A_results += sum(Q_A_correct)
                QA_R_results += sum(QA_R_correct)
                Q_AR_results += sum(Q_AR_correct)
                n_total += len(qa_pred)

        if self.args.distributed:
            numbers = torch.tensor([Q_A_results, QA_R_results, Q_AR_results, n_total], device=self.args.gpu)
            dist.all_reduce(numbers, op=dist.ReduceOp.SUM)
            Q_A_results, QA_R_results, Q_AR_results, n_total = numbers.tolist()

        score_dict = {}
        score_dict['Q_A'] = Q_A_results
        score_dict['QA_R'] = QA_R_results
        score_dict['Q_AR'] = Q_AR_results
        score_dict['n_total'] = n_total

        return score_dict

    def caption_predict(self, loader, dump_path=None):
        self.model.eval()
        with torch.no_grad():

            predictions = []
            targets = []

            gen_kwargs = {}
            gen_kwargs['num_beams'] = self.args.num_beams
            gen_kwargs['max_length'] = self.args.gen_max_length

            for i, batch in enumerate(tqdm(loader, ncols=150, desc="Caption Prediction", disable=not self.verbose)):

                if self.args.distributed:
                    results = self.model.module.test_step(
                        batch,
                        **gen_kwargs)
                else:
                    results = self.model.test_step(
                        batch,
                        **gen_kwargs)

                predictions.extend(results['pred'])

                if 'targets' in batch:
                    targets.extend(batch['targets'])

            if self.args.distributed:
                size = len(loader.dataset)
                predictions = self.collect_results_gpu(predictions, size)
                targets = self.collect_results_gpu(targets, size)

            results = {
                'predictions': predictions,
                'targets': targets
            }

            return results

    def caption_evaluate(self, loader, dump_path=None):
        results = self.caption_predict(loader, dump_path)

        if self.args.gpu == 0:
            predictions = results['predictions']
            if dump_path is None:
                targets = results['targets']
                evaluator = loader.evaluator
                eval_results = evaluator.evaluate(predictions, targets)
                return eval_results

    def mmt_predict(self, loader, dump_path=None):
        self.model.eval()
        with torch.no_grad():

            predictions = []
            targets = [[]]

            gen_kwargs = {}
            gen_kwargs['num_beams'] = self.args.num_beams
            gen_kwargs['max_length'] = self.args.gen_max_length

            for i, batch in enumerate(tqdm(loader, ncols=150, desc=f"MMT Prediction {loader.dataset.source}", disable=not self.verbose)):

                if self.args.distributed:
                    results = self.model.module.test_step(
                        batch,
                        **gen_kwargs)
                else:
                    results = self.model.test_step(
                        batch,
                        **gen_kwargs)

                predictions.extend(results['pred'])

                targets[0].extend(batch['target_text'])

            if self.args.distributed:
                size = len(loader.dataset)
                predictions = self.collect_results_gpu(predictions, size)
                targets[0] = self.collect_results_gpu(targets[0], size)

            if self.args.gpu == 0:
                assert len(predictions) == len(
                    targets[0]), (len(predictions), len(targets[0]))
                assert len(targets) == 1

            results = {
                'predictions': predictions,
                'targets': targets
            }

            if self.args.gpu == 0 and dump_path is not None:
                print('Dumping prediction')
                with open(dump_path, 'w') as f:
                    for i, pred in enumerate(predictions):
                        f.write(pred.lower().strip())
                        if i+1 < len(predictions):
                            f.write('\n')

            return results

    def mmt_evaluate(self, loader, dump_path=None):
        results = self.mmt_predict(loader, dump_path)

        if self.args.gpu == 0:
            evaluator = loader.evaluator
            predictions = results['predictions']
            targets = results['targets']
            eval_results = evaluator.evaluate(predictions, targets)
            return eval_results

    def cls_predict(self, loader, dump_path=None):
        self.model.eval()
        with torch.no_grad():
            quesid2ans = {}

            gen_kwargs = {}
            gen_kwargs['num_beams'] = 1

            for i, batch in enumerate(tqdm(loader, ncols=150, desc="VQA Validation", disable=not self.verbose)):

                if self.args.distributed:
                    results = self.model.module.test_step(batch, **gen_kwargs)
                else:
                    results = self.model.test_step(batch, **gen_kwargs)

                pred_ans = results['pred_ans']
                ques_ids = batch['question_ids']

                for qid, ans in zip(ques_ids, pred_ans):
                    quesid2ans[qid] = ans

            if self.args.distributed:
                # convert to list
                list_qid_ans = list(quesid2ans.items())
                # collect
                all_qid_ans = self.collect_results_gpu(list_qid_ans, len(loader.dataset))
                # convert back to dict
                if self.args.gpu == 0:
                    quesid2ans = dict(all_qid_ans)

            if self.args.gpu == 0 and dump_path is not None:
                loader.evaluator.dump_result(quesid2ans, dump_path)
            return quesid2ans

    def cls_compute_acc(self, quesid2ans, gts_for_data):
        correct = 0
        for k, pred in quesid2ans.items():
            gt = gts_for_data[k]

            if gt == pred:
                correct += 1

        return correct / len(quesid2ans)

    def cls_evaluate(self, loader, dump_path=None):
        quesid2ans = self.cls_predict(loader, dump_path)

        if self.args.gpu == 0:
            acc_dict = {}
            gts_for_data = loader.gts_for_data

            acc_dict = {
                'overall': self.cls_compute_acc(quesid2ans, gts_for_data)
            }

            return acc_dict

    def collect_results_gpu(self, result_part: list, size: int) -> Optional[list]:
        """Collect results under gpu mode.

        On gpu mode, this function will encode results to gpu tensors and use gpu
        communication for results collection.

        Args:
            result_part (list): Result list containing result parts
                to be collected.
            size (int): Size of the results, commonly equal to length of
                the results.

        Returns:
            list: The collected results.
        """
        rank, world_size = dist.get_rank(), dist.get_world_size()
        # dump result part to tensor with pickle
        part_tensor = torch.tensor(
            bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
        # gather all result part tensor shape
        shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
        shape_list = [shape_tensor.clone() for _ in range(world_size)]
        dist.all_gather(shape_list, shape_tensor)
        # padding result part tensor to max length
        shape_max = torch.tensor(shape_list).max()
        part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
        part_send[:shape_tensor[0]] = part_tensor
        part_recv_list = [
            part_tensor.new_zeros(shape_max) for _ in range(world_size)
        ]
        # gather all result part
        dist.all_gather(part_recv_list, part_send)

        if rank == 0:
            part_list = []
            for recv, shape in zip(part_recv_list, shape_list):
                part_result = pickle.loads(recv[:shape[0]].cpu().numpy().tobytes())
                # When data is severely insufficient, an empty part_result
                # on a certain gpu could makes the overall outputs empty.
                if part_result:
                    part_list.append(part_result)
            # sort the results
            ordered_results = []
            for res in zip(*part_list):
                ordered_results.extend(list(res))
            # the dataloader may pad some samples
            ordered_results = ordered_results[:size]
            return ordered_results


def main_worker(gpu, args):
    # GPU is assigned
    args.gpu = gpu
    args.rank = gpu
    print(f'Process Launching at GPU {gpu}')

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend='nccl')

    print(f"args.feature_type {args.feature_type}")
    # use different type of inputs features
    if args.feature_type == "butd":
        import caption_data
        import gqa_data
        import mmt_data
        import nlvr_data
        import refcoco_data
        import vcr_data
        import vqa_data

    elif args.feature_type.startswith("raw"):
        feature_type = args.feature_type.split("_")[-1]

        if args.vis_pooling_output:
            feat_dim_dict = {
                "RN50": 1024,
                "RN101": 512,
                "RN50x4": 640,
            }
        else:
            feat_dim_dict = {
                "RN50": 2048,
                "RN101": 2048,
                "RN50x4": 2560,
                "ViT": 768
            }
        args.feat_dim = feat_dim_dict[feature_type]

        import caption_raw_data as caption_data
        import gqa_raw_data as gqa_data
        import nlvr_raw_data as nlvr_data
        import vqa_raw_data as vqa_data

        ## I haven't added these tasks
        # import vcr_data
        # import mmt_data
        # import refcoco_data

    else:
        feat_dim_dict = {
            "RN50": 2048,
            "RN101": 2048,
            "RN50x4": 2560,
            "ViT": 768
        }
        args.feat_dim = feat_dim_dict[args.feature_type]
        import caption_clip_data as caption_data
        import classification_clip_data as cls_data
        import gqa_clip_data as gqa_data
        import nlvr_clip_data as nlvr_data
        import vqa_clip_data as vqa_data

    vqa_args = deepcopy(args)
    vqa_args.max_text_length = 20

    gqa_args = deepcopy(args)
    gqa_args.batch_size = int(args.batch_size * 100 / 60) # 100
    gqa_args.max_text_length = 20

    nlvr_args = deepcopy(args)
    nlvr_args.batch_size = int(args.batch_size * 20 / 60)

    refcoco_args = deepcopy(args)
    refcoco_args.batch_size = 80
    refcoco_args.max_text_length = 30

    vcr_args = deepcopy(args)
    vcr_args.batch_size = 3
    vcr_args.max_text_length = 100

    caption_args = deepcopy(args)
    caption_args.batch_size = int(args.batch_size * 50 / 60)
    caption_args.max_text_length = 40
    caption_args.gen_max_length = 40

    mmt_args = deepcopy(args)
    mmt_args.batch_size = 20
    mmt_args.max_text_length = 40
    mmt_args.gen_max_length = 40

    cls_args = deepcopy(args)
    cls_args.max_text_length = 20

    if args.use_tasks_prompts:
        vqa_args.prompt = "vqa: "
        gqa_args.prompt = "gpa: "
        nlvr_args.prompt = "nlvr: "
        refcoco_args.prompt = "visual grounding: "
        vcr_args.prompt = ""
        caption_args.prompt = "caption: "
        mmt_args.prompt = "translate English to German: "
        cls_args.prompt = "cls: "
    else:
        vqa_args.prompt = ""
        gqa_args.prompt = ""
        nlvr_args.prompt = ""
        refcoco_args.prompt = ""
        vcr_args.prompt = ""
        caption_args.prompt = ""
        mmt_args.prompt = ""
        cls_args.prompt = ""

    train_loaders = []

    if args.epochs > 0:
        if 'vqa' in args.tasks:
            print(f'Building VQA train loader at GPU {gpu}')
            vqa_train_loader = vqa_data.get_loader(
                vqa_args,
                split='karpathy_train', mode='train', batch_size=vqa_args.batch_size,
                distributed=args.distributed, gpu=args.gpu,
                workers=args.num_workers,
                topk=args.train_topk,
            )
            train_loaders.append(vqa_train_loader)
        if 'gqa' in args.tasks:
            print(f'Building GQA train loader at GPU {gpu}')
            gqa_train_loader = gqa_data.get_loader(
                gqa_args,
                split='train,valid', mode='train', batch_size=gqa_args.batch_size,
                distributed=args.distributed, gpu=args.gpu,
                workers=args.num_workers,
                topk=args.train_topk,
            )
            train_loaders.append(gqa_train_loader)

        if 'nlvr' in args.tasks:
            print(f'Building NLVR train loader at GPU {gpu}')
            nlvr_train_loader = nlvr_data.get_loader(
                nlvr_args,
                split='train', mode='train', batch_size=nlvr_args.batch_size,
                distributed=args.distributed, gpu=args.gpu,
                workers=args.num_workers,
                topk=args.train_topk,
            )
            train_loaders.append(nlvr_train_loader)
        if 'refcoco' in args.tasks:
            print(f'Building RefCOCO train loader at GPU {gpu}')
            refcoco_train_loader = refcoco_data.get_loader(
                refcoco_args,
                split='train', mode='train', batch_size=refcoco_args.batch_size,
                distributed=args.distributed, gpu=args.gpu,
                workers=args.num_workers,
                topk=args.train_topk,
            )
            train_loaders.append(refcoco_train_loader)

        if 'vcr' in args.tasks:
            print(f'Building VCR train loader at GPU {gpu}')
            vcr_train_loader = vcr_data.get_loader(
                vcr_args,
                split='train', mode='train', batch_size=vcr_args.batch_size,
                distributed=args.distributed, gpu=args.gpu,
                workers=args.num_workers,
                topk=args.train_topk,
            )
            train_loaders.append(vcr_train_loader)
        if 'caption' in args.tasks:
            print(f'Building COCO Caption train loader at GPU {gpu}')
            caption_train_loader = caption_data.get_loader(
                caption_args,
                split='karpathy_train', mode='train', batch_size=caption_args.batch_size,
                distributed=args.distributed, gpu=args.gpu,
                workers=args.num_workers,
                topk=args.train_topk,
            )
            train_loaders.append(caption_train_loader)
        if 'mmt' in args.tasks:
            print(f'Building MMT train loader at GPU {gpu}')
            mmt_train_loader = mmt_data.get_loader(
                mmt_args,
                split='train', mode='train', batch_size=mmt_args.batch_size,
                distributed=args.distributed, gpu=args.gpu,
                workers=args.num_workers,
                topk=args.train_topk,
            )
            train_loaders.append(mmt_train_loader)

        if 'cls' in args.tasks:
            print(f'Building CLS train loader at GPU {gpu}')
            cls_train_loader = cls_data.get_loader(
                cls_args,
                split='train', mode='train', batch_size=cls_args.batch_size,
                distributed=args.distributed, gpu=args.gpu,
                workers=args.num_workers,
                topk=args.train_topk,
            )
            train_loaders.append(cls_train_loader)

    train_loader = multitask_data.MultitaskLoader(
        train_loaders,
        sampling=args.multitask_sampling,
        verbose=gpu==0)

    val_num_workers = 4
    # Validation set
    val_loader = {}
    if args.epochs > 0:
        if 'vqa' in args.tasks:
            print(f'Building VQA val loader at GPU {gpu}')
            vqa_val_loader = vqa_data.get_loader(
                vqa_args,
                split='karpathy_val', mode='val', batch_size=vqa_args.batch_size,
                distributed=args.distributed, gpu=args.gpu,
                workers=val_num_workers,
                topk=args.valid_topk,
            )
            val_loader['vqa'] = vqa_val_loader
        if 'gqa' in args.tasks:
            print(f'Building GQA val loader at GPU {gpu}')
            gqa_val_loader = gqa_data.get_loader(
                gqa_args,
                split='testdev', mode='val', batch_size=gqa_args.batch_size,
                distributed=args.distributed, gpu=args.gpu,
                workers=val_num_workers,
                topk=args.valid_topk,
            )
            val_loader['gqa'] = gqa_val_loader
        if 'nlvr' in args.tasks:
            print(f'Building NLVR val loader at GPU {gpu}')
            nlvr_val_loader = nlvr_data.get_loader(
                nlvr_args,
                split='valid', mode='val', batch_size=nlvr_args.batch_size,
                distributed=args.distributed, gpu=args.gpu,
                workers=val_num_workers,
                # topk=args.valid_topk,
            )
            val_loader['nlvr'] = nlvr_val_loader
        if 'vcr' in args.tasks:
            print(f'Building VCR val loader at GPU {gpu}')
            vcr_val_loader = vcr_data.get_loader(
                vcr_args,
                split='val', mode='val', batch_size=vcr_args.batch_size,
                distributed=args.distributed, gpu=args.gpu,
                workers=val_num_workers,
                topk=args.valid_topk,
            )
            val_loader['vcr'] = vcr_val_loader
        if 'refcoco' in args.tasks:
            print(f'Building RefCOCOg val loader at GPU {gpu}')
            refcoco_val_loader = refcoco_data.get_loader(
                refcoco_args,
                split='val', mode='val', batch_size=refcoco_args.batch_size,
                distributed=args.distributed, gpu=args.gpu,
                workers=val_num_workers,
                topk=args.valid_topk,
            )
            val_loader['refcoco'] = refcoco_val_loader
        if 'caption' in args.tasks:
            print(f'Building COCO Caption val loader at GPU {gpu}')
            caption_val_loader = caption_data.get_loader(
                caption_args,
                split='karpathy_val', mode='val', batch_size=caption_args.batch_size,
                distributed=args.distributed, gpu=args.gpu,
                workers=val_num_workers,
                topk=args.valid_topk,
            )
            val_loader['caption'] = caption_val_loader
        if 'mmt' in args.tasks:
            print(f'Building MMT val loader at GPU {gpu}')
            mmt_val_loader = mmt_data.get_loader(
                mmt_args,
                split='val', mode='val', batch_size=mmt_args.batch_size,
                distributed=args.distributed, gpu=args.gpu,
                workers=val_num_workers,
                topk=args.valid_topk,
            )
            val_loader['mmt'] = mmt_val_loader

        if 'cls' in args.tasks:
            print(f'Building CLS val loader at GPU {gpu}')
            cls_val_loader = cls_data.get_loader(
                cls_args,
                split='val', mode='val', batch_size=cls_args.batch_size,
                distributed=args.distributed, gpu=args.gpu,
                workers=val_num_workers,
                topk=args.valid_topk,
            )
            val_loader['cls'] = cls_val_loader

    # Test set
    test_loader = {}
    if 'vqa' in args.tasks:
        print(f'Building VQA test loader at GPU {gpu}')
        vqa_test_loader = vqa_data.get_loader(
            vqa_args,
            split='karpathy_test', mode='val', batch_size=vqa_args.batch_size,
            distributed=args.distributed, gpu=args.gpu,
            workers=val_num_workers,
            topk=args.valid_topk,
        )
        test_loader['vqa'] = vqa_test_loader

        if args.testing:
            vqa_submit_test_loader = vqa_data.get_loader(
                vqa_args,
                split='test_4', mode='val', batch_size=vqa_args.batch_size,
                distributed=args.distributed, gpu=args.gpu,
                workers=val_num_workers,
                topk=args.valid_topk,
            )
            test_loader['vqa_submit'] = vqa_submit_test_loader

    if 'gqa' in args.tasks and args.testing:
        print(f'Building GQA val loader at GPU {gpu}')
        gqa_val_loader = gqa_data.get_loader(
            args,
            split='submit', mode='val', batch_size=nlvr_args.batch_size,
            distributed=args.distributed, gpu=args.gpu,
            workers=val_num_workers,
            topk=args.valid_topk,
        )
        test_loader['gqa'] = gqa_val_loader

    if 'nlvr' in args.tasks:
        print(f'Building NLVR test loader at GPU {gpu}')
        nlvr_test_loader = nlvr_data.get_loader(
            nlvr_args,
            split='test', mode='val', batch_size=nlvr_args.batch_size,
            distributed=args.distributed, gpu=args.gpu,
            workers=val_num_workers,
            # topk=args.valid_topk,
        )
        test_loader['nlvr'] = nlvr_test_loader

    if 'refcoco' in args.tasks:
        print(f'Building RefCOCOg test loader at GPU {gpu}')
        refcoco_test_loader = refcoco_data.get_loader(
            refcoco_args,
            split='test', mode='val', batch_size=refcoco_args.batch_size,
            distributed=args.distributed, gpu=args.gpu,
            workers=val_num_workers,
            topk=args.valid_topk,
        )
        test_loader['refcoco'] = refcoco_test_loader

    if 'caption' in args.tasks:
        print(f'Building COCO Caption test loader at GPU {gpu}')
        caption_test_loader = caption_data.get_loader(
            caption_args,
            split='karpathy_test', mode='val', batch_size=caption_args.batch_size,
            distributed=args.distributed, gpu=args.gpu,
            workers=val_num_workers,
            topk=args.valid_topk,
        )
        test_loader['caption'] = caption_test_loader
    if 'mmt' in args.tasks:
        print(f'Building MMT test2016 loader at GPU {gpu}')
        mmt_test2016_loader = mmt_data.get_loader(
            mmt_args,
            split='test_2016_flickr', mode='val', batch_size=mmt_args.batch_size,
            distributed=args.distributed, gpu=args.gpu,
            workers=val_num_workers,
            topk=args.valid_topk,
        )
        test_loader['mmt_test2016'] = mmt_test2016_loader

        print(f'Building MMT test2017 loader at GPU {gpu}')
        mmt_test2017_loader = mmt_data.get_loader(
            mmt_args,
            split='test_2017_flickr', mode='val', batch_size=mmt_args.batch_size,
            distributed=args.distributed, gpu=args.gpu,
            workers=val_num_workers,
            topk=args.valid_topk,
        )
        test_loader['mmt_test2017'] = mmt_test2017_loader

        print(f'Building MMT test2018 loader at GPU {gpu}')
        mmt_test2018_loader = mmt_data.get_loader(
            mmt_args,
            split='test_2018_flickr', mode='val', batch_size=mmt_args.batch_size,
            distributed=args.distributed, gpu=args.gpu,
            workers=val_num_workers,
            topk=args.valid_topk,
        )
        test_loader['mmt_test2018'] = mmt_test2018_loader

    trainer = Trainer(args, train_loader, val_loader, test_loader, train=True)
    trainer.train()

if __name__ == "__main__":
    cudnn.benchmark = True
    args = parse_args()
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node
    path_print(args.local_rank)
    if args.local_rank in [0, -1]:
        print(args)

        comments = []
        if args.load is not None:
            ckpt_str = "_".join(args.load.split('/')[-3:])
            comments.append(ckpt_str)
        if args.comment != '':
            comments.append(args.comment)
        comment = '_'.join(comments)

        from datetime import datetime
        current_time = datetime.now().strftime('%b%d_%H-%M')
        run_name = f'{current_time}_GPU{args.world_size}'
        if len(comments) > 0:
            run_name += f'_{comment}'

        if args.run_name == "":
            args.run_name = run_name

    # if args.distributed:
    main_worker(args.local_rank, args)
