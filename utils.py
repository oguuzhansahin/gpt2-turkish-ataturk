from config import(
    WORKING_DIR,
    DATA_DIR)

from transformers import TextDataset,DataCollatorForLanguageModeling

def merge_txt_files(files):
    
    all_data = ""
    
    for file_name in files:
        
        data = open(str(DATA_DIR) + "\\" + file_name,"r",encoding="utf-8") \
                                                                .read() \
                                                                .strip()\
                                                                .split("\n")
        cleaned_data = " ".join(data)
        cleaned_data = cleaned_data.replace("Ģ","ş")
        cleaned_data = cleaned_data.replace("Ġ","i")
        cleaned_data = cleaned_data.replace("â","a")
        all_data += cleaned_data
        
        with open("all_data.txt","w",encoding="utf-8") as f:
                f.write(all_data)
                
def build_train_test_files(file_name):
    
    data = open(file_name, "r", encoding = "utf-8") \
                                            .read() \
                                            .split()   
    TRAIN_SIZE  = int(len(data) * 0.75)
           
    train = " ".join(data[:TRAIN_SIZE])
    test  = " ".join(data[TRAIN_SIZE:])
    
    with open("train_dataset.txt","w",encoding="utf-8") as f:
        f.write(train)
    
    with open("test_dataset.txt","w",encoding="utf-8") as f:
        f.write(test)

def load_dataset(train_path,test_path,tokenizer,block_size):
    train_dataset = TextDataset(
          tokenizer=tokenizer,
          file_path=train_path,
          block_size=block_size)
     
    test_dataset = TextDataset(
          tokenizer=tokenizer,
          file_path=test_path,
          block_size=block_size)   
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )
    return train_dataset,test_dataset,data_collator