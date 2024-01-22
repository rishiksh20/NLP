import argparse
import random
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForMultipleChoice
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
from transformers import AutoModelForMultipleChoice
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
import evaluate
import warnings
warnings.filterwarnings("ignore")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

SEED = 595
set_seed(SEED)
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        labels = [feature.pop('labels') for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])

        flattened_features = [[{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features]
        flattened_features = sum(flattened_features, [])
        
        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        
        batch = {k: v.view(batch_size, num_choices, -1).to(device) for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.int64).to(device)
        return batch
    

def get_dataset(dataset):
    dataset = load_dataset(dataset,"ARC-Challenge")
    train_dataset = dataset['train'].filter(lambda r: len(r['choices']['label'])==4)
    test_dataset = dataset['test'].filter(lambda r: len(r['choices']['label'])==4)
    val_dataset = dataset['validation'].filter(lambda r: len(r['choices']['label'])==4)
    return train_dataset, test_dataset, val_dataset

def load_data(tokenizer, params):
    train_data,test_data,val_data = get_dataset(params.dataset)

    def tokenize_function( examples):
        qs = [[context] * 4 for context in examples["question"]]
        ans=[]
        for example in examples["choices"]:
            ans.append([answers for answers in example["text"]])            
        qs = sum(qs, [])
        ans = sum(ans, [])
        tokenized_examples = tokenizer(qs, ans, truncation=True)
        return {k: [v[i:i+4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}

    def label2num(example):
        label_map = {"A": 0, "B": 1, "C": 2, "D": 3,"1":0,"2":1,"3":2,"4":3}
        example['labels'] = [label_map[ch] for ch in example['labels']]
        return example

    tokenized_train = train_data.map(tokenize_function, batched=True)
    tokenized_test = test_data.map(tokenize_function, batched=True)
    tokenized_val = val_data.map(tokenize_function, batched=True)

    tokenized_train = tokenized_train.rename_column("answerKey", "labels")
    tokenized_test = tokenized_test.rename_column("answerKey", "labels")
    tokenized_val = tokenized_val.rename_column("answerKey", "labels")

    accepted_keys = ["input_ids", "attention_mask", "labels"]
    for key in tokenized_train.features.keys():
        if key not in accepted_keys:
            tokenized_train = tokenized_train.remove_columns(key)
    tokenized_train.set_format("torch")

    for key in tokenized_test.features.keys():
        if key not in accepted_keys:
            tokenized_test = tokenized_test.remove_columns(key)
    tokenized_test.set_format("torch")

    for key in tokenized_val.features.keys():
        if key not in accepted_keys:
            tokenized_val = tokenized_val.remove_columns(key)
    tokenized_val.set_format("torch")

    tokenized_train=tokenized_train.map(label2num)
    tokenized_test=tokenized_test.map(label2num)
    tokenized_val=tokenized_val.map(label2num)

    batch_size = params.batch_size
    
    data_collator = DataCollatorForMultipleChoice(tokenizer)
    train_dataset = tokenized_train.shuffle(seed=SEED).select(range(1000))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=data_collator)
    eval_dataset = tokenized_val
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=data_collator)
    test_dataset = tokenized_test
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=data_collator)

    return train_dataloader, eval_dataloader, test_dataloader


def finetune(model, train_dataloader, eval_dataloader, params):
    num_epochs = params.num_epochs
    learning_rate = params.learning_rate
    num_training_steps = num_epochs * len(train_dataloader)
    optimizer = AdamW(model.parameters(), lr = learning_rate)

    lr_scheduler = get_scheduler(
        name="linear", 
        optimizer=optimizer, 
        num_warmup_steps=0, 
        num_training_steps=num_training_steps
    )

    metric = evaluate.load("accuracy")
    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(num_epochs):
        model.train()
        for batch in train_dataloader:
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
        
        model.eval()
        for batch in eval_dataloader:
            with torch.no_grad():
                outputs = model(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])
    
        score = metric.compute()
        print('Validation Accuracy:', score['accuracy'])

    return model


def test(model, test_dataloader, prediction_save='predictions.torch'):
    metric = evaluate.load("accuracy")
    model.eval()
    all_predictions = []

    for batch in test_dataloader:
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        all_predictions.extend(list(predictions))
        metric.add_batch(predictions=predictions, references=batch["labels"])

    score = metric.compute()
    print('Test Accuracy:', score)
    torch.save(all_predictions, prediction_save)


def main(params):
    tokenizer = AutoTokenizer.from_pretrained(params.model)
    train_dataloader, eval_dataloader, test_dataloader = load_data(tokenizer, params)
    model = AutoModelForMultipleChoice.from_pretrained(params.model)
    model.to(device)
    model = finetune(model, train_dataloader, eval_dataloader, params)
    test(model, test_dataloader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune Language Model")
    parser.add_argument("--dataset", type=str, default="ai2_arc")
    parser.add_argument("--model", type=str, default="bert-base-cased")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    params, unknown = parser.parse_known_args()
    main(params)
