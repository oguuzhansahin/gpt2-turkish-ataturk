import pathlib
import os
import argparse
from transformers import pipeline
from pathlib import Path
from config import WORKING_DIR,MODEL_NAME

OUTPUT_DIR = Path(WORKING_DIR,[file for file in os.listdir(WORKING_DIR) if file.startswith("gpt2-")][0])

arg_parser = argparse.ArgumentParser(description="Fine Tuning Turkish GPT-2 on Ataturk's Book")
arg_parser.add_argument("--text", required = True,
                          help="The output directory")
args = arg_parser.parse_args()
args = vars(args)

text = args['text']

pipe = pipeline('text-generation', model=str(OUTPUT_DIR),
                 tokenizer=MODEL_NAME, config={'max_length':800})   
text = pipe(text)[0]["generated_text"]
print("Generated text: ", text)