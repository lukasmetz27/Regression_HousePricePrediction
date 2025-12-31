import pandas as pd
import numpy as np
import os
from dataclasses import dataclass
from src.logger import logging


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def remove_outliers_zscore(self, df, numeric_cols, threshold=3.0):
        z_scores = np.abs(
            (df[numeric_cols] - df[numeric_cols].mean()) /
            df[numeric_cols].std()
        )
        mask = (z_scores < threshold).all(axis=1)
        return df[mask]


    def get_data_transformer_object(self):
        try:
            numerical_columns = [
                "CRIM",      # Kriminalitätsrate
                "ZN",        # Anteil Wohnfläche
                "INDUS",     # Industrieanteil
                "NOX",       # Luftverschmutzung
                "RM",        # Zimmeranzahl
                "AGE",       # Anteil alter Gebäude
                "DIS",       # Distanz zu Arbeitszentren
                "RAD",       # Highway-Zugänglichkeit (Index)
                "TAX",       # Grundsteuer
                "PTRATIO",   # Schüler-Lehrer-Verhältnis
                "B",         # demographischer Index
                "LSTAT"      # Anteil niedriger sozialer Status
            ]

        except:
            pass



    def initiate_data_transformation(self,train_path, test_path):
        try:
            df_train = pd.read_csv(train_path)
            df_test = pd.read_csv(test_path)
            logging.info("read train test data completed")
            
            numerical_columns = [
                "CRIM",      # Kriminalitätsrate
                "ZN",        # Anteil Wohnfläche
                "INDUS",     # Industrieanteil
                "NOX",       # Luftverschmutzung
                "RM",        # Zimmeranzahl
                "AGE",       # Anteil alter Gebäude
                "DIS",       # Distanz zu Arbeitszentren
                "RAD",       # Highway-Zugänglichkeit (Index)
                "TAX",       # Grundsteuer
                "PTRATIO",   # Schüler-Lehrer-Verhältnis
                "B",         # demographischer Index
                "LSTAT"      # Anteil niedriger sozialer Status
            ]

            df_train = self.remove_outliers_zscore(
                df_train,
                numeric_cols=numerical_columns,
                threshold=3.0
                )
            
            logging.info("outliers removed")



        except:
            pass