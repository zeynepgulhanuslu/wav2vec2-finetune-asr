
# Wav2vec2 Fine Tuning with Custom Dataset

This project contains Wav2vec2 asr model finetuning with custom kaldi dataset.


## Installation

Tested on : version python 3.8

```bash
conda create -n wav2vec-env python=3.8
```
    
## Required packages
```bash
pip install -r requirements.txt
```


## Usage/Examples

First create data for wav2vec2 training from kaldi data directory. 

```bash  
cd dataloader
python convert_kaldi_data.py --kaldi_dir /kaldi_dir/train/ --out_file train.csv
python convert_kaldi_data.py --kaldi_dir /kaldi_dir/test/ --out_file test.csv

```




Then dataset can be used ```finetune_with_hg.py``` or 
```finetune_with_flash.py ``` for creating a model.

```finetune_with_flash``` is finetuning model with lightning-flash.

```finetune_with_hg``` is finetuning model with huggingface.


```bash  
cd training
python finetune_with_flash.py 
--train train.csv 
--test test.csv 
--epochs 20 
--num_nodes 1 
--gpus 1 
--batch_size 1 
--model_file asr_flash.pt
```

```bash  
finetune_with_hg.py 
--train train.csv 
--test test.csv 
--vocab vocab.json 
--num_proc 4 
--out_dir tr-huggingface-finetuned/
```

For inference:

```bash  
predict_with_flash.py 
--model asr_flash.pt
--test_file test.wav
--gpus 1
```