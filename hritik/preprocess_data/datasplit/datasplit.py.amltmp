import os
import argparse
import pandas as pd
import numpy as np

from azureml.core import Run
run = Run.get_context()

from sklearn.model_selection import train_test_split
 
parser = argparse.ArgumentParser("split")
 
parser.add_argument("--split_data", type=str, help="Spliting raw data into train test and valid",dest = "data")

args = parser.parse_args()
 
dataframe = run.input_datasets["raw_data"].to_pandas_dataframe()

train, raw_test = train_test_split(dataframe,test_size=0.18, random_state=42)
valid,test  = train_test_split(raw_test,test_size=0.4, random_state=42)

 
os.makedirs(args.data, exist_ok=True)

train.to_csv(args.data+"/train.csv", index=False)
test.to_csv(args.data+"/test.csv", index=False)
valid.to_csv(args.data+"/valid.csv", index=False)
