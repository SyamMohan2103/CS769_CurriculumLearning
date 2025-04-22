from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig, DataCollatorForLanguageModeling, pipeline
import torch
from tqdm import tqdm
import os

from peft import LoraConfig, get_peft_model
from torch.nn.utils.rnn import pad_sequence

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


dataset = load_dataset("openlifescienceai/medmcqa", split='train')

ans_to_idx_map = {"A":0, "B":1, "C":2, "D":3}
idx_to_ans_map = {0:"A", 1:"B", 2:"C", 3:"D"}

def format_example(example):
    instruction = "Answer the following multiple-choice question by giving the most appropriate response. The answer should be one of [A, B, C, D]."
    options = f"A. {example['opa']} B. {example['opb']} C. {example['opc']} D. {example['opd']}"
    prompt = f"{instruction}\nQuestion: {example['question']}\n{options}\nAnswer: {idx_to_ans_map[example['cop']]}"
    
    return {"text": prompt}

dataset = dataset.map(format_example)

print(len(dataset))
print(dataset[0]['text'])

model_id = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

# def tokenize_fn(example):
#     return tokenizer(example["text"], padding="max_length", truncation=True, max_length=512)

def tokenize_fn(example):
    tokenized = tokenizer(example["text"], padding="max_length", truncation=True, max_length=512)
    return {"input_ids": tokenized["input_ids"]}

tokenized = dataset.map(tokenize_fn)

class DataCollatorForAnswerToken:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        input_ids = [torch.tensor(f["input_ids"]) for f in features]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)

        labels = input_ids.clone().fill_(-100)

        # For each sequence, find last non-padding token and unmask it
        for i in range(input_ids.size(0)):
            non_pad_indices = (input_ids[i] != self.tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
            if len(non_pad_indices) > 0:
                last_token_idx = non_pad_indices[-1]
                labels[i, last_token_idx] = input_ids[i, last_token_idx]

        return {
            "input_ids": input_ids,
            "labels": labels,
        }


# add Quant_Lora
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,               # 4-bit quantization
    bnb_4bit_use_double_quant=True,  # double quantization for stability
    bnb_4bit_compute_dtype="bfloat16",  # or "float16"
    bnb_4bit_quant_type="nf4",       # best trade-off for LLaMA
)

# device_map={'':torch.cuda.current_device()}
# {'':torch.cuda.current_device()}, 
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto", quantization_config=bnb_config)
# model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

# add lora
lora_config = LoraConfig(
    r=64,                        # LoRA rank
    lora_alpha=16,               # Alpha scaling factor
    target_modules=["q_proj", "v_proj"],  # target modules in transformer blocks
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()

args = TrainingArguments(
    # output_dir="./llama3-medmcqa-baseline",
    output_dir="./llama3-qlora-medmcqa", # LoRA
    per_device_train_batch_size=3, # 5 : 15gb, 4 - 13gb, 3 - 11gb
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    logging_steps=10,
    save_strategy="epoch",
    fp16=False,
    bf16=True,
    report_to="none",
    # max_steps=15
)

trainer = Trainer(
    # model=model, 
    model=peft_model, # LoRA
    args=args,
    train_dataset=tokenized,
    #train_batch_size = 8,
    # data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    data_collator=DataCollatorForAnswerToken(tokenizer=tokenizer),
)

trainer.train()