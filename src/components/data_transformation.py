import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

import os
from dataclasses import dataclass
from src.logger import logging
from src.utils import save_object


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

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())
                ]
            )

            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns)
                ]
            )

            return preprocessor

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

            preprocessing_obj=self.get_data_transformer_object()
            target_column_name="Price"

            input_feature_train_df=df_train.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=df_train[target_column_name]

            input_feature_test_df=df_test.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=df_test[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)            

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            #raise CustomException(e,sys)
            pass