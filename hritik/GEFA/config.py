import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data-path",type=str,dest="data_path",help="Davis Directory")
parser.add_argument("--model_path",type=str,dest="model",help="Trained model output")
parser.add_argument("--data_type",type=int,dest="data_type",help="datasetType")
parser.add_argument("--out_path",type=str,dest="output_path",help="datasetType")
parser.add_argument("--generated",type=str,dest="generated",help="datasetType")
args = parser.parse_args()

l = ['davis','pred']

import os
if(l[args.data_type] == "davis"):
    os.makedirs(args.model, exist_ok=True)

data_path = args.data_path + '/'
model_path = args.model

if(l[args.data_type] == "pred"):
    os.makedirs(args.output_path, exist_ok=True)
    output_path = args.output_path
    generated_path = args.generated

    

is_seq_in_graph = True          # set True to use sequential data of proteins
is_con_in_graph = True          # set True to use contact map data
is_profile_in_graph = True      # set True to use profile data
is_emb_in_graph = True          # set True to use embedding features from TAPE-proteins, False to use one-hot encoded data

NUM_EPOCHS = 100               # Number of epochs for which the model is trained
TRAIN_BATCH_SIZE = 128          # Number of samples for one training batch
TEST_BATCH_SIZE = 256           # Number of samples for one testing batch
PRED_BATCH_SIZE = 256           # Number of samples for one prediction batch. Required only for running predict.py
run_model = 0                   # 0 for GEFA, 1 for GLFA
cuda = 0                        # GPU used
setting = 0                     # Setting number, 0, 1, 2 or 3
LR = 0.0005                     # Learning Rate
dataset = l[args.data_type]               # dataset used for training
to_plot_drug = False             # set True to plot graphs of drugs from smiles
to_plot_prot = False            # set True to plot graphs of proteins from smiles
mode = 0                        # 0 for training, 1 for evaluating trained model on test data
from_resume = False              # To resume previous train

# path to the model on which predictions will be made. required only while making predictions.
pred_model_path = model_path +'/saved_model'+ "/setting_1/model_GEFA_davis_emb_seq_con_pf_setting_1_70_779.model"
