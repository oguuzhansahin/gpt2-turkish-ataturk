<h3 align="center">
    <p>Fine-tuning Turkish GPT-2 on Ataturk's data</p>
</h3>

<p align="center">
  <img src="img/ataturk.jpg" />
</p>

## Quick tour

Requirements

```
transformers==4.2.2
```

Model's file can be downloaded by [this link](https://drive.google.com/drive/folders/1QDNYdnoNHlm7y90aVQc9JGBMLfra_0us?usp=sharing) 


To fine-tune Turkish GPT-2 on your custom dataset, you can run main.py with the following arguments. Be aware that output directory has to be start with "gpt2-".

```
python3 main.py \
--output_dir "YOUR_OUTPUT_PATH" \
--num_train_epochs=1 \
--per_device_train_batch_size=32 \
--per_device_eval_batch_size=64 \
--eval_steps = 400 \
--save_steps=800 \
--warmup_steps=500 \
--overwrite_output_dir True \
--prediction_loss_only True      
```

After fine-tuning you can test your model by using test.py file.

```
python3 test.py \
--text "Türkiye Cumhuriyeti'nin istikbali" 

# Output
Türkiye Cumhuriyeti'nin istikbali, bağımsızlığı uğrunda asla taviz vermeyeceğimizin bir kez daha hatırlatılması gerekir.
```

Or if you want to generate advance text, you can basically use the following code:


```python
>>> from config import tokenizer
>>> from transformers import AutoModelWithLMHead
>>> from test import OUTPUT_DIR

>>> model = AutoModelWithLMHead.from_pretrained(str(OUTPUT_DIR))

# Tokenize input text and return as pytorch tensor.
>>> inputs = tokenizer("Türkiye Cumhuriyeti'nin istikbali" , return_tensors="pt")
>>> sample_outputs = model.generate(inputs.input_ids,
                                pad_token_id=50256,
                                do_sample=True, 
                                max_length=50, # put the token number you want
                                top_k=40,
                                num_return_sequences=1)
>>> for i, sample_output in enumerate(sample_outputs):
        print(">> Generated text {}\\\\
    \\\\
    {}".format(i+1, tokenizer.decode(sample_output.tolist())))

```
