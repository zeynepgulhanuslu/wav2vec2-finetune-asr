import argparse

import torch
from datasets import load_dataset, Audio
from transformers import AutoProcessor, AutoModelForCTC

from dataloader.convert_kaldi_data import get_dataset
from training.finetune_with_hg import remove_special_characters, replace_hatted_characters


def prepare_dataset(batch):
    audio = batch["audio"]
    # batched output is "un-batched"
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    with processor.as_target_processor():
        batch["labels"] = processor(batch["sentence"]).input_ids
    return batch


if __name__ == '__main__':
    model_path = "patrickvonplaten/wav2vec2-large-xls-r-300m-turkish-colab"
    parser = argparse.ArgumentParser()

    parser.add_argument('--test', type=str, required=True, help='test csv data file')
    args = parser.parse_args()
    test_file = args.test
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForCTC.from_pretrained(model_path)

    model = model.to('cpu')
    print(model)
    test_dataset = get_dataset(test_file)
    sentence = test_dataset[0]["sentence"].lower()
    test_dataset = test_dataset.map(remove_special_characters)
    test_dataset = test_dataset.map(replace_hatted_characters)
    test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=16_000))
    test_dataset = test_dataset.map(prepare_dataset, remove_columns=test_dataset.column_names,
                                    num_proc=4, keep_in_memory=True)

    input_dict = processor(test_dataset[0]["input_values"], sampling_rate=16_000, return_tensors="pt", padding=True)

    logits = model(input_dict.input_values.to("cpu")).logits

    pred_ids = torch.argmax(logits, dim=-1)[0]

    print("Prediction:")
    print(processor.decode(pred_ids))

    print("\nReference:")
    print(sentence)
