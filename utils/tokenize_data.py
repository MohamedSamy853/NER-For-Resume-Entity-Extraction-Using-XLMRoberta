import os , pickle
import logging
import argparse
import typing as t
import torch
from datasets import load_from_disk , Dataset  , DatasetDict
from transformers import XLMRobertaTokenizerFast

checkpoint = 'xlm-roberta-base'

# Create and configure logger
logging.basicConfig(filename="./logs/tokenize.log",
                    format='%(asctime)s %(message)s',
                    filemode='w' ,
                    level=logging.INFO)

# Creating an object
logger = logging.getLogger()

logger.info('start logging ...')


def load_label2id_and_id2label()->t.Tuple[t.Dict , t.Dict]:
    with open("./label2id.pkl", 'rb') as f:
        label2id = pickle.load(f)
    with open("./id2label.pkl" , 'rb') as f:
        id2label = pickle.load(f)
        
    return label2id , id2label
def load_data(path:str)->DatasetDict:
    dataset = load_from_disk(path)
    return dataset

   

def align_labels_with_tokens(word_ids :torch.Tensor , labels :t.List | torch.Tensor)->t.List:
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id !=current_word:
            current_word = word_id
            label = 'UNK' if word_id is None else labels[word_id]
            new_labels.append(label2id[label])
        elif word_id is None:
            new_labels.append(label2id["UNK"])
        else :
            label = labels[word_id]
            if label.startswith("B"):
                label = label.replace("B", "I")
            new_labels.append(label2id[label])
    return new_labels
def tokenize_and_align_labels(examples:Dataset):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    all_labels = examples["ner-tags"]
    new_labels = align_labels_with_tokens(tokenized_inputs.word_ids() ,all_labels )

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs

def load_tokenizer()->XLMRobertaTokenizerFast:
    return XLMRobertaTokenizerFast.from_pretrained(checkpoint , 
                                                   )
    
if __name__ == '__main__':
    print(os.getcwd())
    parse =argparse.ArgumentParser()
    parse.add_argument('--input' , type=str , default='./data/preprocessed_data')
    parse.add_argument('--output', type=str , default='./data/tokenized_data')
    parsesr = parse.parse_args()
    logger.info("loading tokenizer ...")
    tokenizer = load_tokenizer()
    logger.info("tokeizer is loaded .")
    label2id , id2label = load_label2id_and_id2label()
    
    label2id['O'] = -100
    label2id['UNK'] = -100
    id2label[-100] = 'UNK'
    logger.info("loading dataset ...")
    dataset = load_data(path=parsesr.input)
    print(dataset)
    logger.info("dataset is loaded succesuflly!")

    if not os.path.isdir(parsesr.output):
        os.makedirs(parsesr.output)
    logger.info("tokenization data in progress ...")
    tokenized_datasets = dataset.map(
        tokenize_and_align_labels,
        batched=False,
        remove_columns=dataset["train"].column_names,
    )
    logger.info("tokenization data is completed.")
    tokenized_datasets.save_to_disk(parsesr.output)
    logger.info(f"tokenized data is saved to {parsesr.output}")
    


