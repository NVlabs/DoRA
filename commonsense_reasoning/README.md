<!---
Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Finetuning LLaMA on commonsense reasoning tasks using DoRA

This directory includes the DoRA implementation and guidelines for replicating the results outlined in our paper.

## Setup
1. Install dependencies
```bash
conda create -n dora_llama python=3.10
conda activate dora_llama
pip install -r requirements.txt
```

## Datasets
1. Download the complete commonsense datasets from [here](https://github.com/AGI-Edgerunners/LLM-Adapters/tree/main/dataset) and download the commonsense 170k finetuning dataset from [here](https://github.com/AGI-Edgerunners/LLM-Adapters/blob/main/ft-training_set/commonsense_170k.json), then organize the data as follows
```bash
# Store the complete commonsense datasets
./dataset
# rest of the files
./experiment
./peft
# Finetuning commonsense dataset
./commonsense_170k.json
...
```

## Code Structure

Refer to `./peft/src/peft/tuners/dora.py` for the implementation of DoRA.

Refer to `./finetune.py` for finetuning LLaMA using DoRA.

Refer to `./commonsense_evaluate.py` for the evaluation of the finetuned model.

## Finetuning and Evaluation

### Finetuning (`./7B_Dora.sh`)
This file contains the code to finetune LLaMA-7B using DoRA. User can specify different DoRA configuration for finetuning. To be specific, the first argument denotes the rank r, the second argument specifies the corresponding alpha, the third argument indicates the destination for saving the fine-tuned model, and the last argument determines the GPU to use.
 
An example could be:
```
sh 7B_Dora.sh 32 64 ./finetuned_result/dora_r32 0
```

### Finetuning (`./7B_Dora_qkv.sh`)
This file contains the code to finetune LLaMA-7B using DoRA but with more customizability, that is user can further specify which modules to only finetune the magnitude component of DoRA by changing `--Wdecompose_target_modules`, please refer to Sec. 5.6 in the paper for more details.

An example could be:
```
sh 7B_Dora_qkv.sh 32 64 ./finetuned_result/dora_qkv_r32 0
```

### Evaluation

You can directly download the finetuned DoRA weights from [google drive](https://drive.google.com/drive/folders/1tFVtNcpfwdCLQTrHpP-1LJiq5jH3reUc?usp=sharing) and evaluate them with `7B_Dora_eval.sh` as describe below to reproduce the result reported in the paper.

This file contains the code to evaluate LLaMA-7B finetuned with DoRA on the eight commonsense reasoning tasks. The first argument is the address of the DoRA weight, the second argument specifies where you would like to save the evaluation result, and the last argument determines which GPU to use.

An example could be:
```
sh 7B_Dora_eval.sh ./finetuned_result/dora_r32 ./finetuned_result/dora_r32 0
```

## DoRA Result
| Model                 |  BoolQ  |  PIQA  |  SIQA  |  HellaSwag  |  WinoGrande  |  ARC-e  |  ARC-c  |  OBQA  |  Average  |
|-----------------------|---------|--------|--------|-------------|--------------|---------|---------|--------|-----------|
| LLaMA-7B-Prefix	      |   64.3  |  76.8  |  73.9  |  42.1       |  72.1        |  72.9   |  54.0   |  60.6  |  64.6     |
| LLaMA-7B-Series	      |   63.0  |  79.2  |  76.3  |  67.9       |  75.7        |  74.5   |  57.1   |  72.4  |  70.8     |
| LLaMA-7B-Parallel	    |   67.9  |  76.4  |  78.8  |  69.8       |  78.9        |  73.7   |  57.3   |  75.2  |  72.3     |
| LLaMA-7B-LoRA	(r32)	        |   68.9  |  80.7  |  77.4  |  78.1       |  78.8        |  77.8   |  61.3   |  74.8  |  74.7     |
| LLaMA-7B-DoRA (r16)	        |   70.0  | 82.6  | 79.7  | 83.2  | 80.6  | 80.6  | 65.4  | 77.6  | **77.5**    |
| LLaMA-7B-DoRA	(r32)	        |   68.5  | 82.9  | 79.6  | 84.8  | 80.8  | 81.4  | 65.8  | 81.0  | **78.1**    |
| LLaMA-13B-Prefix	    |   65.3  |  75.4  |  72.1  |  55.2       |  68.6        |  79.5   |  62.9   |  68.0  |  68.4     |
| LLaMA-13B-Series	    |   71.8  |  83.0  |  79.2  |  88.1       |  82.4        |  82.5   |  67.3   |  81.8  |  79.5     |
| LLaMA-13B-Parallel	  |   72.5  |  84.8  |  79.8  |  92.1       |  84.7        |  84.2   |  71.2   |  82.4  |  81.4     |
| LLaMA-13B-LoRA	(r32)      |   72.1  |  83.5  |  80.5  |  90.5       |  83.7        |  82.8   |  68.3   |  82.4  |  80.5     |
| LLaMA-13B-DoRA	(r16)      |   72.5  | 85.3  | 79.9  | 90.1  | 82.9  | 82.7  | 69.7  | 83.6  | **80.8**     |
| LLaMA-13B-DoRA	(r32)      |   72.4 | 84.9 | 81.5 | 92.4 | 84.2 | 84.2 | 69.6 | 82.8 | **81.5**     |

## Acknowledgement
We greatly appreciate the contributions of two remarkable repositories: [LLM-Adapter](https://github.com/AGI-Edgerunners/LLM-Adapters), [PEFT](https://github.com/huggingface/peft). These projects have significantly benefited our work.


