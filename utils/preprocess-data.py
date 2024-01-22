import os
import argparse
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import logging
from datasets import Dataset , DatasetDict
from pydantic import BaseModel , validator
import typing as t 

class Config(BaseModel):
    test_size:float
    input_path:str
    output_path :str
    
    @validator('test_size')
    def check_test_size(cls, value):
        if value >= 1 or value <= 0:
            raise ValueError("test size must be in ]0 ,1[")
        else:
            return value
    

# Create and configure logger


logging.basicConfig(filename="./logs/perprocess.log",
                    format='%(asctime)s %(message)s',
                    filemode='w',
                    level=logging.INFO)

# Creating an object
logger = logging.getLogger()

logger.info('start logging ...')

#load data from csv and peperer it 
def load_mapping_dict(path:str = '.\mapping.pkl')->t.Dict:
    
    with open(path , 'rb') as f:
        mapping = pickle.load(f)
    return mapping
def map_labels(df :pd.DataFrame, mapping:t.Dict , col :str) ->pd.DataFrame:
    df[col] = df[col].apply(lambda x :eval(x)) #to convert str to list
    df[col] = df[col].apply(lambda x : [mapping[i] for i in x])
    return df


if __name__ == '__main__':
    
    parse_args = argparse.ArgumentParser()
    parse_args.add_argument('--test-size' , default=0.1 , type=int ,help='size of test data ')
    parse_args.add_argument('--output' , default='./data/preprocessed_data/' , type=str)
    parse_args.add_argument('--input' , default='./data/prepered_data/prepered_data.csv' , type=str)
    args = parse_args.parse_args()
    config = Config(test_size=args.test_size , input_path=args.input , output_path=args.output)
    
    mapping = load_mapping_dict()

    logger.info('mappaing file is loaded succesuffly!')
    logger.info(f'num class of mapping file is {len(mapping)}')
    
    tags = sorted([*mapping.values()] )
    label2id ={label : i for i , label in enumerate(tags)}
    id2label = {i : label for label , i in label2id.items()}
    with open("./label2id.pkl" ,'wb') as f:
        pickle.dump(label2id , f)
    with open("./id2label.pkl", 'wb') as f:
        pickle.dump(id2label , f)
    with open("./tags.pkl" , "wb") as f:
        pickle.dump(tags , f)
    

    #load data 
    logger.info("start loading csc data ...")

    data = pd.read_csv(config.input_path)

    logger.info("data is loaded suceesuffly!")
    logger.info(f"example of data \n {data.head().values}")
    logger.info(f"data shape {data.shape}")
    data['tokens'] = data['content'].str.split()
    #get only used columns
    df = data[['tokens', 'ner-tags']].copy()
 
    df = map_labels(df , mapping , 'ner-tags')

    #split data 
    train , validation = train_test_split(df , test_size=config.test_size, random_state=42 , shuffle=True)
    logger.info(f"Data is split to train , validataion")
    logger.info(f"train size is {train.shape}")
    logger.info(f"validation size is {validation.shape}")
    train.to_parquet("./data/prepered_data/train.parquet", index=False )
    validation.to_parquet("./data/prepered_data/test.parquet" , index=False )
    train_dataset = Dataset.from_parquet("./data/prepered_data/train.parquet")
    validation_dataset = Dataset.from_parquet("./data/prepered_data/test.parquet")

    dataset = DatasetDict(
        {"train":train_dataset , 
        "validation" : validation_dataset}
    )

    if not os.path.exists(config.output_path):
        os.makedirs(config.output_path)
        
    dataset.save_to_disk(config.output_path)
    logger.info(f"data is saved at {config.output_path}")







