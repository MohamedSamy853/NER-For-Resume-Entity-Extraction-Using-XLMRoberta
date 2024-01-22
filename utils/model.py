import os , pickle
import logging
import argparse
import sys
import numpy as np
import typing as t
import pandas as pd
import torch
import evaluate
from datasets import load_from_disk  , DatasetDict
from transformers import XLMRobertaTokenizerFast 
from transformers import XLMRobertaForTokenClassification 
from transformers import Trainer , TrainingArguments
from transformers import DataCollatorForTokenClassification
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

os.system(f"{sys.executable} -m pip install seqeval")
checkpoint = 'xlm-roberta-base'

# Create and configure logger
logging.basicConfig(filename="./logs/model.log",
                    format='%(asctime)s %(message)s',
                    filemode='w' , 
                    level=logging.INFO)

# Creating an object
logger = logging.getLogger()

logger.info('start logging ...')
with open("./tags.pkl" , "rb") as f:
        tags = pickle.load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"
metric = evaluate.load("seqeval")
def load_label2id_and_id2label()->t.Tuple[t.Dict , t.Dict]:
    with open("./label2id.pkl", 'rb') as f:
        label2id = pickle.load(f)
    with open("./id2label.pkl" , 'rb') as f:
        id2label = pickle.load(f)
        
    return label2id , id2label
def load_data(path:str)->DatasetDict:
    dataset = load_from_disk(path)
    return dataset

def load_model(checkpoint , label2id , id2label):

    model = XLMRobertaForTokenClassification.from_pretrained(
        checkpoint , 
        id2label=id2label,
        label2id=label2id
        )
    return model.to(device)
def get_param_optimizr(model):
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "gamma", "beta"]
    optimizer_grouped_parameters = [
        {
            "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay_rate": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay_rate": 0.0,
            },
        ]
    return optimizer_grouped_parameters

def load_tokenizer(checkpoint):
    return XLMRobertaTokenizerFast.from_pretrained(checkpoint)


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[tags[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [tags[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }


if __name__ == '__main__':
    
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("--epochs" , type=int , default=5 )
    arg_parse.add_argument("--lr" , type=float , default=5e-5)
    arg_parse.add_argument('--input' , type=str , default="./data/tokenized_data")
    arg_parse.add_argument('--output' , type=str , default="./model/saved_model")
    args_ = arg_parse.parse_args()
    logger.info("load datasets ...")
    tokenized_datasets = load_data(args_.input)
    logger.info("datasets is loaded ...")
    
    logger.info("load tokenizer ...")
    tokenizer = load_tokenizer(checkpoint)
    logger.info("tokenizer is loaded .")
    label2id , id2label = load_label2id_and_id2label()
    O_id = label2id['O']
    label2id.__delitem__("O")
    id2label.__delitem__(O_id)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer , return_tensors='pt') 
    logger.info("load model ...")
    model = load_model(checkpoint , label2id , id2label)
    logger.info("model is loaded.")
    logger.info("get model param optimizers")
    optimizer_grouped_parameters = get_param_optimizr(model)
    optim = AdamW(params=optimizer_grouped_parameters , 
                  lr = args_.lr )
    
    num_warm_up = int(0.001 * len(tokenized_datasets["train"]) * args_.epochs)
    total_steps = int(args_.epochs * len(tokenized_datasets["train"]))
    optim_schedule = get_linear_schedule_with_warmup(optim ,num_warmup_steps = num_warm_up , 
                                                     num_training_steps = total_steps)
    
    args = TrainingArguments(
    "./model/bert-finetuned-ner-resume",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=args_.lr,
    num_train_epochs=args_.epochs,
    optim = 'adamw_hf' ,
    push_to_hub=False,

    disable_tqdm=False,
    
    )
    
    
    trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    optimizers=(optim , optim_schedule)
    )
    logger.info("start training ")
    trainer.train()
    logger.info("training  completed .")

    if not os.path.exists(args_.output):
        os.makedirs(args_.output)
    trainer.save_model(args_.output)
    logger.info("model is saved .")
    
    
    
