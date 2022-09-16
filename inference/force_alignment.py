import torch
from datasets import load_dataset, Audio
from transformers import AutoProcessor, AutoModelForCTC

from training.finetune_with_hg import remove_special_characters, replace_hatted_characters, prepare_dataset

if __name__ == '__main__':
    model_path = "patrickvonplaten/wav2vec2-large-xls-r-300m-turkish-colab"

    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForCTC.from_pretrained(model_path)

    model = model.to('cpu')
    print(model.summary())
    common_voice_test = load_dataset("common_voice", "tr", split="test")
    common_voice_test = common_voice_test.remove_columns(
        ["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])
    common_voice_test = common_voice_test.map(remove_special_characters)
    common_voice_test = common_voice_test.map(replace_hatted_characters)
    common_voice_test = common_voice_test.cast_column("audio", Audio(sampling_rate=16_000))
    common_voice_test = common_voice_test.map(prepare_dataset, remove_columns=common_voice_test.column_names,
                                              num_proc=4, keep_in_memory=True)

    input_dict = processor(common_voice_test[0]["input_values"], return_tensors="pt", padding=True)

    logits = model(input_dict.input_values.to("cpu")).logits

    pred_ids = torch.argmax(logits, dim=-1)[0]

    common_voice_test_transcription = load_dataset("common_voice", "tr", data_dir="./cv-corpus-6.1-2020-12-11",
                                                   split="test")
    print("Prediction:")
    print(processor.decode(pred_ids))

    print("\nReference:")
    print(common_voice_test_transcription[0]["sentence"].lower())