import os 
import argparse
import logging

from config import(
    DATA_DIR,
    WORKING_DIR,
    FILE_NAME,
    TRAIN_PATH,
    TEST_PATH,
    tokenizer,
    model)

from utils import(
    merge_txt_files,
    build_train_test_files,
    load_dataset)

from transformers import (
    TextDataset,
    DataCollatorForLanguageModeling,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

logger = logging.getLogger(__name__)


if "all_data.txt" not in os.listdir(WORKING_DIR):
    
    files = [file for file in os.listdir(DATA_DIR) if file.endswith(".txt")]
    logging.info("Merging all txt files into all_data.txt...")
    merge_txt_files(files)
    
else:
    logging.info("All_data.txt file already exist...")
    

if "train_dataset.txt" not in os.listdir(WORKING_DIR):
    
    logging.info("Building train test files...")
    build_train_test_files(FILE_NAME)
else:
    logging.info("train_dataset.txt or test_dataset.txt already exist...")


def main():
    
    arg_parser = argparse.ArgumentParser(description="Fine Tuning Turkish GPT-2 on Ataturk's Book")
    arg_parser.add_argument("--output_dir", required = True,
                            help="The output directory")
    arg_parser.add_argument("--overwrite_output_dir", required=False, default = True,
                            help="Overrite the contents of the output directory")
    arg_parser.add_argument("--num_train_epochs", default=1,
                            help="Number of training epochs")
    arg_parser.add_argument("--per_device_train_batch_size", default=32,
                            help="Batch size for training")
    arg_parser.add_argument("--per_device_eval_batch_size", default=64,
                            help="Batch size for evaluation")
    arg_parser.add_argument("--eval_steps", default=400, 
                            help="Number of update steps between two evaluations.")
    arg_parser.add_argument("--save_steps", default=800,
                            help="after # steps model is saved ")
    arg_parser.add_argument("--warmup_steps", default=500,
                            help="number of warmup steps for learning rate scheduler ")
    arg_parser.add_argument("--prediction_loss_only", default=True)
    
    args = arg_parser.parse_args()
    args = vars(args)
    
    logger.info(f"Loading train and test dataset from {TRAIN_PATH,TEST_PATH}")
    train_dataset,test_dataset,data_collator = load_dataset(TRAIN_PATH,TEST_PATH,tokenizer,block_size=128)
    
    training_args = TrainingArguments(
        output_dir                 = args['output_dir'],
        overwrite_output_dir       = args['overwrite_output_dir'], 
        num_train_epochs           = args['num_train_epochs'], 
        per_device_train_batch_size= args['per_device_train_batch_size'], 
        per_device_eval_batch_size = args['per_device_eval_batch_size'],  
        eval_steps                 = args['eval_steps'], 
        save_steps                 = args['save_steps'], 
        warmup_steps               = args['warmup_steps'],
        prediction_loss_only       = args['prediction_loss_only'],
        )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )
    logger.info("Training is starting...")
    trainer.train()
    logger.info("Saving the trained model to the {}".format(args["output_dir"]))
    trainer.save_model()
    

if __name__ == "__main__":
    main()