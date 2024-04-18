<h1 align="center">
    <p>DoRA: Weight-Decomposed Low-Rank Adaptation</p>
</h1>

<h1 align="center"> 
    <img src="./dora.png" width="600">
</h1>

This repo contains the official implementation of [DoRA](https://arxiv.org/abs/2402.09353) and the code for reproducing the results of the four experiments reported in the paper.

DoRA: Weight-Decomposed Low-Rank Adaptation

Shih-Yang Liu, Chien-Yi Wang, Hongxu Yin, Pavlo Molchanov, Yu-Chiang Frank Wang, Kwang-Ting Cheng, Min-Hung Chen

Paper: https://arxiv.org/abs/2402.09353

DoRA decomposes the pre-trained weight into two components, magnitude and direction, for fine-tuning, specifically employing LoRA for directional updates to efficiently minimize the number of trainable parameters. By employing DoRA, we enhance both
the learning capacity and training stability of
LoRA while avoiding any additional inference
overhead. DoRA consistently outperforms LoRA
on fine-tuning LLaMA, LLaVA, and VL-BART
on various downstream tasks, such as commonsense reasoning, visual instruction tuning, and
image/video-text understanding

## Repository Structure
This repo contains four directories:

`./commonsense_reasoning` contains the code to finetune LLaMA-7B/13B using DoRA on the commonsense reasoning tasks. This directory is modified based on [LLM-Adapter](https://github.com/AGI-Edgerunners/LLM-Adapters).

`./instruction_tuning` contains the code to finetune LLaMA-7B and LLaMA2-7B using DoRA and DVoRA (DoRA+VeRA) with the cleaned Alpaca instruction tuning dataset. This directory is modified based on [VeRA](https://openreview.net/attachment?id=NjNfLdxr3A&name=supplementary_material).

`./image_video_text_understanding` contains the code to finetune VL-BART using DoRA for the image/video-text understanding tasks. This directory is modified based on [VL-Adapter](https://github.com/ylsung/VL_adapter).

`./visual_instruction_tuning` contains the code to finetune LLaVA-1.5-7B on the visual instruction tuning tasks with DoRA. This directory is modified based on [LLaVA](https://github.com/haotian-liu/LLaVA).

## Quick Start on your own tasks
### HuggingFace PEFT
DoRA is now supported by the Huggingface PEFT package. You can install the PEFT package using
```
pip install git+https://github.com/huggingface/peft.git -q
```
After PEFT is installed, you can simply set the use_dora argument of LoRAConfig to True for applying DoRA.
An example could be as follows:
```
from peft import LoraConfig, get_peft_model

# Initialize DoRA configuration
config = LoraConfig(
    r=8, 
    lora_alpha=16, 
    target_modules=[
      "q_proj",
      "k_proj",
      "v_proj",
      "o_proj",
      "gate_proj",
      "up_proj",
      "down_proj",
    ],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    use_dora=True
)
```
Please refer to the official [documentation](https://huggingface.co/docs/peft/en/package_reference/lora) for more details.

### HuggingFace Diffuser
You can also toy with DoRA on finetuning Diffusion Model. A good starting point would be this [collab notebook](https://colab.research.google.com/drive/134mt7bCMKtCYyYzETfEGKXT1J6J50ydT?usp=sharing#scrollTo=23d6bb49-3469-4e23-baf5-25b2344b599d) from [Linoy Tsaban](https://twitter.com/linoy_tsaban).

## Contact
Shih-Yang Liu: [shihyangl@nvidia.com](shihyangl@nvidia.com) or [sliuau@connect.ust.hk](sliuau@connect.ust.hk)

## Licenses
Copyright © 2024, NVIDIA Corporation. All rights reserved.

This work is made available under the NVIDIA Source Code License-NC. Refer to LICENSE to view a copy of the complete license.

## Citation
If you find DoRA useful, please cite it by using the following BibTeX entry.
```bibtex
@article{liu2024dora,
  title={DoRA: Weight-Decomposed Low-Rank Adaptation},
  author={Liu, Shih-Yang and Wang, Chien-Yi and Yin, Hongxu and Molchanov, Pavlo and Wang, Yu-Chiang Frank and Cheng, Kwang-Ting and Chen, Min-Hung},
  journal={arXiv preprint arXiv:2402.09353},
  year={2024}
}
```