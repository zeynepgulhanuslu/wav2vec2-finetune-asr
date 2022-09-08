import argparse
import os

import flash
from flash.audio import SpeechRecognitionData, SpeechRecognition

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', type=str, required=True, help='train json data file')
    parser.add_argument('--test', type=str, required=True, help='test json data file')
    parser.add_argument('--epochs', type=int, required=True, help='epoch count')
    parser.add_argument('--num_nodes', type=int, required=True, help='num nodes')
    parser.add_argument('--gpus', type=int, required=True, help='gpu counts')
    parser.add_argument('--model_file', type=str, required=True, help='output model file')

    args = parser.parse_args()
    train_json_file = args.train
    test_json_file = args.test
    model_file = args.model_file

    datamodule = SpeechRecognitionData.from_json(
        input_field="file",
        target_field="text",
        train_file=train_json_file,
        test_file=test_json_file,
        batch_size=4)

    model = SpeechRecognition(backbone="patrickvonplaten/wav2vec2-large-xls-r-300m-turkish-colab")

    trainer = flash.Trainer(max_epochs=args.epochs, gpus=args.gpus, num_nodes=args.num_nodes)
    trainer.finetune(model, datamodule=datamodule, strategy='no_freeze')
    # Save Checkpoint
    trainer.save_checkpoint(model_file)

    print(model)
