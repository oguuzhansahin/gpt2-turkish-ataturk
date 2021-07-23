import pathlib
import os
from pathlib import Path
from transformers import(    
    AutoModelWithLMHead,
    AutoTokenizer
)

WORKING_DIR = pathlib.Path().resolve()
DATA_DIR = Path(WORKING_DIR,"data")
MODEL_NAME = "redrussianarmy/gpt2-turkish-cased"
FILE_NAME = "all_data.txt"
TRAIN_PATH = "train_dataset.txt"
TEST_PATH = "test_dataset.txt"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelWithLMHead.from_pretrained(MODEL_NAME)
