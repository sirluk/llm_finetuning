import random
import functools
import csv
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from skmultilearn.model_selection import iterative_train_test_split
from datasets import Dataset, DatasetDict
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)


def tokenize_examples(examples, tokenizer):
    tokenized_inputs = tokenizer(examples['text'])
    tokenized_inputs['labels'] = examples['labels']
    return tokenized_inputs


def collate_fn(batch):
    d = {k: [dic[k] for dic in batch] for k in batch[0] if k in ['input_ids', 'attention_mask', 'labels']}
    d['input_ids'] = torch.nn.utils.rnn.pad_sequence([torch.tensor(x) for x in d['input_ids']], batch_first=True, padding_value=tokenizer.pad_token_id)
    d['attention_mask'] = torch.nn.utils.rnn.pad_sequence([torch.tensor(x) for x in d['attention_mask']], batch_first=True, padding_value=0)
    d['labels'] = torch.tensor(d['labels'])
    return d


def compute_metrics(p):
    predictions, labels = p
    f1_micro = f1_score(labels, predictions > 0, average = 'micro')
    f1_macro = f1_score(labels, predictions > 0, average = 'macro')
    f1_weighted = f1_score(labels, predictions > 0, average = 'weighted')
    return {
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }


class CustomTrainer(Trainer):
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # compute custom loss
        loss = F.binary_cross_entropy_with_logits(logits, labels.to(torch.float32))
        return (loss, outputs) if return_outputs else loss


random.seed(0)

# load data
with open('train.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile, delimiter=','))
    header_row = data.pop(0)

random.shuffle(data)
 
idx, text, labels = list(zip(*[(int(row[0]), f'Title: {row[1].strip()}\n\nAbstract: {row[2].strip()}', row[3:]) for row in data]))
labels = np.array(labels, dtype=int)

# stratified train test split for multilable ds
row_ids = np.arange(len(labels))
train_idx, y_train, val_idx, y_val = iterative_train_test_split(row_ids[:,np.newaxis], labels, test_size = 0.1)
x_train = [text[i] for i in train_idx.flatten()]
x_val = [text[i] for i in val_idx.flatten()]

# create dataset
ds = DatasetDict({
    'train': Dataset.from_dict({'text': x_train, 'labels': y_train}),
    'val': Dataset.from_dict({'text': x_val, 'labels': y_val})
})

# tokenizer dataset
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenize_ds = functools.partial(tokenize_examples, tokenizer=tokenizer)
tokenized_ds = ds.map(tokenize_ds, batched=True)

# set pad token
tokenizer.pad_token = tokenizer.eos_token

# qunatization config
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# lora config
lora_config = LoraConfig(
    r=16,
    lora_alpha=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLS"
)

# load model
model = AutoModelForSequenceClassification.from_pretrained("mistralai/Mistral-7B-v0.1", quantization_config=quantization_config, num_labels=labels.shape[1])
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model.config.pad_token_id = tokenizer.pad_token_id

# define training args
training_args = TrainingArguments(
    output_dir="mistral7b-lora-classification",
    learning_rate=1e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    use_cpu=False
)

# train
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["val"],
    tokenizer=tokenizer,
    data_collator=collate_fn,
    compute_metrics=compute_metrics
)

trainer.train()

# save model
peft_model_id = "results"
trainer.model.save_pretrained(peft_model_id)
tokenizer.save_pretrained(peft_model_id)
