# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import json
import logging
import os
import pickle
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from typing import Optional

import multitask_data
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import video.how2qa_data as how2qa_data
import video.tvc_data as tvc_data
import video.tvqa_data as tvqa_data
import video.yc2c_data as yc2c_data
import wandb
from adapters import AdapterController, MetaLayersAdapterController
from ddp_fix import ddp_forward
from param import parse_args
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from trainer_base import TrainerBase
from utils import LossMeter, set_global_logging_level
from video.video_model import VLBartVideo, VLT5Video
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
            model_class = VLT5Video
        elif 'bart' in args.backbone:
            model_class = VLBartVideo

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

            self.model.vis_encoder = self.vis_encoder


        self.freeze_whole_model() # freeze whole parameters first
        self.unfreeze_parameters() # unfreeze selected parameters

        self.percent_updated_parameters = self.print_trainable_params_percentage(self.model)

        # Load Checkpoint
        self.start_epoch = None
        if args.load is not None:
            print("load model weight!")
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
            tvqa_loss_meter = LossMeter()
            tvc_loss_meter = LossMeter()
            # best_eval_loss = 9595.
            quesid2ans = {}
            best_tvqa_valid = 0.
            best_tvqa_epoch = 0

            # how2qa
            best_how2qa_valid = 0
            best_how2qa_epoch = 0

            # tvc
            best_tvc_valid = 0
            best_tvc_epoch = 0

            # yc2c
            best_yc2c_valid = 0
            best_yc2c_epoch = 0

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
                'tvqa': 0,
                'how2qa': 0,
                'tvc': 0,
                'yc2c': 0,
            }

            for step_i, batch in enumerate(self.train_loader):
                task = batch['task']
                task_counter[task] += 1
                batch['log_train_accuracy'] = self.args.log_train_accuracy

                # self.optim.zero_grad()
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

                if self.verbose:
                    if task == 'tvqa':
                        tvqa_loss_meter.update(loss.item())
                    elif task == 'tvc':
                        tvc_loss_meter.update(loss.item())

                    desc_str = f'Epoch {epoch} | LR {lr:.6f}'

                    desc_str += f" |"
                    if 'tvqa' in self.args.tasks:
                        desc_str += f" TVQA {task_counter['tvqa']}"
                    if 'how2qa' in self.args.tasks:
                        desc_str += f" How2QA {task_counter['how2qa']}"
                    if 'tvc' in self.args.tasks:
                        desc_str += f" TVC {task_counter['tvc']}"
                    if 'yc2c' in self.args.tasks:
                        desc_str += f" YC2C {task_counter['yc2c']}"

                    if len(tvqa_loss_meter) > 0:
                        desc_str += f' | TVQA Loss {tvqa_loss_meter.val:4f}'
                    if len(tvc_loss_meter) > 0:
                        desc_str += f' | TVC Loss {tvc_loss_meter.val:.3f}'

                    pbar.set_description(desc_str)
                    pbar.update(1)

            if self.verbose:
                pbar.close()

            # Validation
            log_str = ''
            wandb_log_dict = {}

            if 'tvqa' in self.args.tasks:
                # TVQA
                tvqa_val_loader = self.val_loader['tvqa']
                score_dict = self.qa_evaluate(tvqa_val_loader)
                if self.args.gpu == 0:
                    valid_score = score_dict['all_type_accuracy'] * 100.
                    if valid_score > best_tvqa_valid or epoch == 0:
                        best_tvqa_valid = valid_score
                        best_tvqa_epoch = epoch
                        # self.save("VQA_BEST")
                    log_str += f"TVQA"
                    log_str += "\nEpoch %d: Valid Raw %0.2f" % (epoch, valid_score)
                    log_str += "\nEpoch %d: Best Raw %0.2f\n" % (best_tvqa_epoch, best_tvqa_valid)
                    wandb_log_dict['VQA/Valid/score'] = valid_score
            if 'how2qa' in self.args.tasks:
                # How2QA
                how2qa_val_loader = self.val_loader['how2qa']
                score_dict = self.qa_evaluate(how2qa_val_loader)
                if self.args.gpu == 0:
                    valid_score = score_dict['all_type_accuracy'] * 100.
                    if valid_score > best_how2qa_valid or epoch == 0:
                        best_how2qa_valid = valid_score
                        best_how2qa_epoch = epoch
                    wandb_log_dict['How2QA/Valid/Acc'] = valid_score
                    log_str += f"How2QA"
                    log_str += "\nEpoch %d: Valid %0.2f" % (epoch, valid_score)
                    log_str += "\nEpoch %d: Best %0.2f\n" % (best_how2qa_epoch, best_how2qa_valid)
            if 'tvc' in self.args.tasks:
                # TVC
                tvc_val_loader = self.val_loader['tvc']
                valid_results = self.caption_evaluate(tvc_val_loader)
                if self.args.gpu == 0:
                    valid_score = valid_results['CIDEr'] * 100
                    if valid_score > best_tvc_valid or epoch == 0:
                        best_tvc_valid = valid_score
                        best_tvc_epoch = epoch
                    for score_name, score in valid_results.items():
                        wandb_log_dict[f'TVC/Valid/{score_name}'] = score * 100
                    log_str += f"TVC"
                    log_str += "\nEpoch %d: Valid CIDEr %0.2f" % (epoch, valid_score)
                    log_str += "\nEpoch %d: Best %0.2f\n" % (best_tvc_epoch, best_tvc_valid)
            if 'yc2c' in self.args.tasks:
                # YC2C
                yc2c_val_loader = self.val_loader['yc2c']
                valid_results = self.caption_evaluate(yc2c_val_loader)
                if self.args.gpu == 0:
                    valid_score = valid_results['CIDEr'] * 100
                    if valid_score > best_yc2c_valid or epoch == 0:
                        best_yc2c_valid = valid_score
                        best_yc2c_epoch = epoch
                    for score_name, score in valid_results.items():
                        wandb_log_dict[f'YC2C/Valid/{score_name}'] = score * 100
                    log_str += f"YC2C"
                    log_str += "\nEpoch %d: Valid CIDEr %0.2f" % (epoch, valid_score)
                    log_str += "\nEpoch %d: Best %0.2f\n" % (best_yc2c_epoch, best_yc2c_valid)

            if self.verbose:
                wandb.log(wandb_log_dict, step=epoch)
                print(log_str)

        if self.args.epochs > 0: # model is trained
            self.save("LAST")

         # test only
        if self.args.epochs == 0:
            if self.verbose:
                print("start generating test submission file")
                if not os.path.isdir(self.args.output):
                    os.makedirs(self.args.output, exist_ok=True)
            
            if 'tvqa' in self.args.tasks:
                # TVQA
                tvqa_test_loader = self.test_loader['tvqa']

                predictions = self.qa_predict(tvqa_test_loader)

                if self.args.gpu == 0:
                    dump_results = {}
                    for pred in predictions:
                        try:
                            p = int(pred['answer'][1])
                        except:
                            p = 0
                        
                        dump_results[pred['question_id']] = p

                    with open(os.path.join(self.args.output, "tvqa_test_predictions.json"), "w") as f:
                        json.dump(dump_results, f)

            if 'how2qa' in self.args.tasks:
                # How2QA
                how2qa_test_loader = self.test_loader['how2qa']

                predictions = self.qa_predict(how2qa_test_loader)

                if self.args.gpu == 0:
                    dump_results = {}
                    for pred in predictions:
                        try:
                            p = int(pred['answer'][1])
                        except:
                            p = 0
                        
                        dump_results[pred['question_id']] = p

                    with open(os.path.join(self.args.output, "how2qa_test_public_predictions.json"), "w") as f:
                        json.dump(dump_results, f)

            if 'tvc' in self.args.tasks:
                # TVC
                tvc_test_loader = self.test_loader['tvc']

                predictions = self.caption_predict(tvc_test_loader)

                if self.args.gpu == 0:
                    preds = predictions["predictions"]
                    tss = predictions["tss"]
                    video_ids = predictions["video_ids"]
                    clip_ids = predictions["clip_ids"]

                    dump_results = []

                    for pred, ts, video_id, clip_id in zip(preds, tss, video_ids, clip_ids):
                        
                        result = {
                            "vid_name": video_id,
                            "clip_id": clip_id,
                            "ts": ts,
                            "descs": [{"desc": pred}]
                        }
                        
                        dump_results.append(result)

                    with open(os.path.join(self.args.output, "tvc_test_predictions.jsonl"), "w") as f:
                        for entry in dump_results:
                            json.dump(entry, f)
                            f.write('\n')

            if 'yc2c' in self.args.tasks:
                # YC2C
                yc2c_test_loader = self.test_loader['yc2c']

                predictions = self.caption_predict(yc2c_test_loader)

                if self.args.gpu == 0:
                    preds = predictions["predictions"]
                    video_ids = predictions["video_ids"]
                    clip_ids = predictions["clip_ids"]

                    dump_results = []

                    for pred, video_id, clip_id in zip(preds, video_ids, clip_ids):
                        
                        result = {
                            "vid_name": video_id,
                            "clip_id": clip_id,
                            "descs": [{"desc": pred}]
                        }
                        
                        dump_results.append(result)

                    with open(os.path.join(self.args.output, "yc2c_test_predictions.jsonl"), "w") as f:
                        for entry in dump_results:
                            json.dump(entry, f)
                            f.write('\n')

        if self.verbose:
            wandb.run.summary['finished'] = True

        if self.args.distributed:
            dist.destroy_process_group()

    def qa_predict(self, loader, dump_path=None):
        self.model.eval()
        with torch.no_grad():
            quesid2ans = []
            for i, batch in enumerate(tqdm(loader, ncols=150, disable=not self.verbose)):
                if self.args.distributed:
                    results = self.model.module.test_step(batch)
                else:
                    results = self.model.test_step(batch)

                pred_ans = results['pred_ans']
                ques_ids = batch['question_ids']

                for pred, ques in zip(pred_ans, ques_ids):
                    quesid2ans.append(
                        {"question_id": ques, "answer": pred}
                    )

        if self.args.distributed:
            quesid2ans = self.collect_results_gpu(quesid2ans, len(loader.dataset))

        return quesid2ans

    def qa_evaluate(self, loader, dump_path=None):
        quesid2ans = self.qa_predict(loader, dump_path)

        if self.args.gpu == 0:
            evaluator = loader.evaluator
            corrects, type_count = evaluator.eval(quesid2ans)
            acc_dict = evaluator.output(corrects, type_count)

            return acc_dict

    def caption_predict(self, loader, dump_path=None):
        """
        Predict the answers to questions in a data split.
        :param eval_tuple: The data tuple to be evaluated.
        :param dump: The path of saved file to dump results.
        :return: A dict of question_id to answer.
        """
        self.model.eval()
        with torch.no_grad():

            predictions = []
            targets = []
            tss = []
            video_ids = []
            clip_ids = []

            gen_kwargs = {}
            gen_kwargs['num_beams'] = self.args.num_beams
            gen_kwargs['max_length'] = self.args.gen_max_length

            for i, batch in enumerate(tqdm(loader, ncols=120, desc="Prediction", disable=not self.verbose)):

                if self.args.distributed:
                    results = self.model.module.test_step(
                        batch,
                        **gen_kwargs)
                else:
                    results = self.model.test_step(
                        batch,
                        **gen_kwargs)

                predictions.extend(results['pred_ans'])

                if 'answers' in batch:
                    targets.extend(batch['answers'])

                if 'tss' in batch:
                    tss.extend(batch['tss'])

                if 'video_ids' in batch:
                    video_ids.extend(batch['video_ids'])

                if 'question_ids' in batch:
                    clip_ids.extend(batch['question_ids'])

            if self.args.distributed:
                size = len(loader.dataset)
                predictions = self.collect_results_gpu(predictions, size)
                targets = self.collect_results_gpu(targets, size)
                tss = self.collect_results_gpu(tss, size)
                video_ids = self.collect_results_gpu(video_ids, size)
                clip_ids = self.collect_results_gpu(clip_ids, size)

            results = {
                'predictions': predictions,
                'targets': targets,
                'tss': tss,
                'video_ids': video_ids,
                'clip_ids': clip_ids,
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
        dist.init_process_group(backend='nccl', timeout=timedelta(hours=1))

    args.feat_dim = 512

    tvqa_args = deepcopy(args)
    how2qa_args = deepcopy(args)
    tvc_args = deepcopy(args)
    yc2c_args = deepcopy(args)

    if args.use_tasks_prompts:
        tvqa_args.prompt = "tvqa: "
        how2qa_args.prompt = "how2qa: "
        tvc_args.prompt = "tvc: "
        yc2c_args.prompt = "yc2c: "
    else:
        tvqa_args.prompt = ""
        how2qa_args.prompt = ""
        tvc_args.prompt = ""
        yc2c_args.prompt = ""

    train_loaders = []

    if args.epochs > 0:
        if 'tvqa' in args.tasks:
            print(f'Building TVQA train loader at GPU {gpu}')
            tvqa_train_loader = tvqa_data.get_loader(
                tvqa_args,
                split='train', mode='train', batch_size=tvqa_args.batch_size,
                distributed=args.distributed, gpu=args.gpu,
                workers=args.num_workers,
                topk=args.train_topk,
            )
            train_loaders.append(tvqa_train_loader)
            # print(f'VQA train loader len: {len(vqa_train_loader)}')
        if 'how2qa' in args.tasks:
            print(f'Building How2QA train loader at GPU {gpu}')
            how2qa_train_loader = how2qa_data.get_loader(
                how2qa_args,
                split='train_release', mode='train', batch_size=how2qa_args.batch_size,
                distributed=args.distributed, gpu=args.gpu,
                workers=args.num_workers,
                topk=args.train_topk,
            )
            train_loaders.append(how2qa_train_loader)
            # print(f'GQA train loader len: {len(gqa_train_loader)}')

        if 'tvc' in args.tasks:
            print(f'Building TVC train loader at GPU {gpu}')
            tvc_train_loader = tvc_data.get_loader(
                tvc_args,
                split='train_release', mode='train', batch_size=tvc_args.batch_size,
                distributed=args.distributed, gpu=args.gpu,
                workers=args.num_workers,
                topk=args.train_topk,
            )
            train_loaders.append(tvc_train_loader)
            # print(f'NLVR train loader len: {len(nlvr_train_loader)}')
        if 'yc2c' in args.tasks:
            print(f'Building TC2C train loader at GPU {gpu}')
            yc2c_train_loader = yc2c_data.get_loader(
                yc2c_args,
                split='train_release', mode='train', batch_size=yc2c_args.batch_size,
                distributed=args.distributed, gpu=args.gpu,
                workers=args.num_workers,
                topk=args.train_topk,
            )
            train_loaders.append(yc2c_train_loader)
            # print(f'RefCOCO train loader len: {len(refcoco_train_loader)}')

    train_loader = multitask_data.MultitaskLoader(
        train_loaders,
        sampling=args.multitask_sampling,
        verbose=gpu==0)

    val_num_workers = 4
    # Validation set
    val_loader = {}
    if args.epochs > 0:
        valid_batch_size = args.valid_batch_size or args.batch_size
        if 'tvqa' in args.tasks:
            print(f'Building TVQA val loader at GPU {gpu}')
            tvqa_val_loader = tvqa_data.get_loader(
                tvqa_args,
                split='val', mode='val', batch_size=valid_batch_size,
                distributed=args.distributed, gpu=args.gpu,
                workers=val_num_workers,
                topk=args.valid_topk,
            )
            val_loader['tvqa'] = tvqa_val_loader
        if 'how2qa' in args.tasks:
            print(f'Building How2QA val loader at GPU {gpu}')
            how2qa_val_loader = how2qa_data.get_loader(
                how2qa_args,
                split='val_release', mode='val', batch_size=valid_batch_size,
                distributed=args.distributed, gpu=args.gpu,
                workers=val_num_workers,
                topk=args.valid_topk,
            )
            val_loader['how2qa'] = how2qa_val_loader
        if 'tvc' in args.tasks:
            print(f'Building TVC val loader at GPU {gpu}')
            tvc_val_loader = tvc_data.get_loader(
                tvc_args,
                split='val_release', mode='val', batch_size=valid_batch_size,
                distributed=args.distributed, gpu=args.gpu,
                workers=val_num_workers,
                topk=args.valid_topk,
            )
            val_loader['tvc'] = tvc_val_loader
        if 'yc2c' in args.tasks:
            print(f'Building YC2C val loader at GPU {gpu}')
            yc2c_val_loader = yc2c_data.get_loader(
                yc2c_args,
                split='val_release', mode='val', batch_size=valid_batch_size,
                distributed=args.distributed, gpu=args.gpu,
                workers=val_num_workers,
                topk=args.valid_topk,
            )
            val_loader['yc2c'] = yc2c_val_loader

    test_loader = None

    test_loader = {}
    if args.epochs == 0: # test only
        valid_batch_size = args.valid_batch_size or args.batch_size
        if 'tvqa' in args.tasks:
            print(f'Building TVQA test loader at GPU {gpu}')
            tvqa_test_loader = tvqa_data.get_loader(
                tvqa_args,
                split='test_public,test_release', mode='val', batch_size=valid_batch_size,
                distributed=args.distributed, gpu=args.gpu,
                workers=val_num_workers,
                topk=args.valid_topk,
            )
            test_loader['tvqa'] = tvqa_test_loader
        if 'how2qa' in args.tasks:
            print(f'Building How2QA test loader at GPU {gpu}')
            how2qa_test_loader = how2qa_data.get_loader(
                how2qa_args,
                split='test_public_release', mode='val', batch_size=valid_batch_size,
                distributed=args.distributed, gpu=args.gpu,
                workers=val_num_workers,
                topk=args.valid_topk,
            )
            test_loader['how2qa'] = how2qa_test_loader
        if 'tvc' in args.tasks:
            print(f'Building TVC test loader at GPU {gpu}')
            tvc_test_loader = tvc_data.get_loader(
                tvc_args,
                split='test_release', mode='val', batch_size=valid_batch_size,
                distributed=args.distributed, gpu=args.gpu,
                workers=val_num_workers,
                topk=args.valid_topk,
            )
            test_loader['tvc'] = tvc_test_loader
        if 'yc2c' in args.tasks:
            print(f'Building YC2C test loader at GPU {gpu}')
            yc2c_test_loader = yc2c_data.get_loader(
                yc2c_args,
                split='test_release', mode='val', batch_size=valid_batch_size,
                distributed=args.distributed, gpu=args.gpu,
                workers=val_num_workers,
                topk=args.valid_topk,
            )
            test_loader['yc2c'] = yc2c_test_loader

    trainer = Trainer(args, train_loader, val_loader, test_loader, train=True)
    trainer.train()

if __name__ == "__main__":
    cudnn.benchmark = True
    args = parse_args()
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node
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
