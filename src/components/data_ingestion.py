import pandas as pd
from src.logger import logging
import os
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join("artifacts", "train.csv")
    test_data_path = os.path.join("artifacts", "test.csv")
    modelling_data_path = os.path.join("artifacts", "modelling_data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logging.info("before data ingestion")
            df1 = pd.read_csv("/Users/lukasmetz/Desktop/Lukas/Code/FullProjects/BostonHousePrice/src/notebook/data/train.csv")
            df2 = pd.read_csv("/Users/lukasmetz/Desktop/Lukas/Code/FullProjects/BostonHousePrice/src/notebook/data/train.csv")
            df = pd.concat([df1,df2], axis=0)

            #splitting in train and test data
            train_set, test_set=train_test_split(df,test_size=0.2,random_state=42)
            

            # if there exist no folder artifacts it creates one 
            os.makedirs(os.path.dirname(self.ingestion_config.modelling_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.modelling_data_path, index=False,header=True)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False,header=True)


            logging.info("after data ingestion")

        except:
            pass


# only for testing    
if __name__=="__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()