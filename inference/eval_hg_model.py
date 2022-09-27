import argparse
import os

import torch
from datasets import Audio
from transformers import AutoProcessor, AutoModelForCTC

from dataloader.convert_kaldi_data import get_dataset
from training.finetune_with_hg import replace_hatted_characters, remove_special_characters


def prepare_dataset(batch):
    audio = batch["audio"]
    # batched output is "un-batched"
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    with processor.as_target_processor():
        batch["labels"] = processor(batch["sentence"]).input_ids
    return batch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, required=True, help='model directory')
    parser.add_argument('--test', type=str, required=True, help='test csv file')
    parser.add_argument('--num_proc', type=int, required=True, help='num process counts')
    parser.add_argument('--out_file', type=str, required=True, help='out file for wer information')

    args = parser.parse_args()
    model_dir = args.model
    test_file = args.test
    out_file = args.out_file
    num_process = args.num_proc

    model = AutoModelForCTC.from_pretrained(model_dir)
    processor = AutoProcessor.from_pretrained(model_dir)

    print('creating test dataset')
    test_dataset = get_dataset(test_file)
    test_dataset = test_dataset.map(replace_hatted_characters)
    test_dataset = test_dataset.map(remove_special_characters)
    test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=16_000))

    test_dataset = test_dataset.map(prepare_dataset, remove_columns=test_dataset.column_names,
                                    num_proc=num_process, keep_in_memory=True)

    parent_dir = os.path.dirname(out_file)
    if not os.path.exists(parent_dir):
        os.mkdir(parent_dir)

    f_o = open(out_file, 'w', encoding='utf-8')
    for x in range(len(test_dataset)):
        input_dict = processor(test_dataset[x]["input_values"], sampling_rate=16_000, return_tensors="pt", padding=True)

        logits = model(input_dict.input_values).logits

        pred_ids = torch.argmax(logits, dim=-1)[0]

        transcript = processor(test_dataset[x]["sentence"])

        f_o.write("\n" + "Prediction:" + processor.decode(pred_ids) +
                  "\n" + "Reference:" + transcript)
