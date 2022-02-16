# load general packages and functions
import csv
import sys
import os

# load program-specific functions
import parameters.load as load
from tools.atom_types import get_atom_types
from tools.max_n_nodes import get_max_n_atoms
from tools.formal_charges import get_formal_charges

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--input-data",type=str,dest="in_data",help="Input splitted data")
parser.add_argument("--job-dir",type=str,dest="job_dir",help="Job Directory")
parser.add_argument("--train_dir",type=str,dest="train_dir",help="Train Directory")
parser.add_argument("--test_dir",type=str,dest="test_dir",help="Test Directory")
parser.add_argument("--valid_dir",type=str,dest="valid_dir",help="Valid Directory")
parser.add_argument("--data_path",type=str,dest="data_path",help="Fine-Tune Directory")
parser.add_argument("--trained",type=str,dest="trained_model",help="Trained Model Dir.")
args = parser.parse_args()

atoms =  get_atom_types(args.in_data + "/train.csv")
nodes =  get_max_n_atoms(args.in_data + "/train.csv")
formal = get_formal_charges(args.in_data + "/train.csv")

os.makedirs(args.job_dir, exist_ok=True)

# default parameters defined below

# general job parameters

params_dict = {
    "atom_types": atoms,
    "formal_charge": formal,
    "max_n_nodes": nodes,
    "imp_H": [0, 1, 2, 3],
    "chirality": ["None", "R", "S"],
    "job_type": "learn",
    
    "dataset_dir": args.trained_model + '/', #model_dir,
    "train": args.train_dir + '/',
    "test": args.test_dir + '/',
    "valid": args.valid_dir + '/',
    "job_dir" : args.job_dir + '/',
    "data_path": args.data_path + '/qsar/',

    "restart": False,
    "model": "GGNN",
    
    "sample_every": 50,
    "init_lr": 1e-4,
    "min_rel_lr": 1e-2,
    "max_rel_lr": 1,

    "group_size": 5000,
    "generation_epoch": 50,  
    "n_samples": 100,
    "n_workers": 0,
    
    "sigma": 20,
    "alpha": 0.5,
    "score_type": "activity", #"reduce", "augment", "qed", "activity",
    
    "use_aromatic_bonds": False,
    "use_canon": True,
    "use_chirality": False,
    "use_explicit_H": False,
    "ignore_H": True,
}

# model common hyperparameters
model_common_hp_dict = {
    "batch_size": 100,
    "gen_batch_size": 10,
    "block_size": 100000,
    "epochs": 100,
    "init_lr": 1e-4,
    "min_rel_lr": 5e-2,
    "max_rel_lr": 1,
    "sigma": 1,
    "weights_initialization": "uniform",
    "weight_decay": 0.0,
    "alpha": 0.5,
}

# get the model before loading model-specific hyperparameters

model = "GGNN"  # default model

model_common_hp_dict["model"] = model


# model-specific hyperparameters (implementation-specific)
if model_common_hp_dict["model"] == "GGNN":

    model_specific_hp_dict = {
        "enn_depth": 4,
        "enn_dropout_p": 0.0,
        "enn_hidden_dim": 250,
        "mlp1_depth": 4,
        "mlp1_dropout_p": 0.0,
        "mlp1_hidden_dim": 500,
        "mlp2_depth": 4,
        "mlp2_dropout_p": 0.0,
        "mlp2_hidden_dim": 500,
        "gather_att_depth": 4,
        "gather_att_dropout_p": 0.0,
        "gather_att_hidden_dim": 250,
        "gather_emb_depth": 4,
        "gather_emb_dropout_p": 0.0,
        "gather_emb_hidden_dim": 250,
        "gather_width": 100,
        "hidden_node_features": 100,
        "message_passes": 3,
        "message_size": 100,
    }


# join dictionaries
params_dict.update(model_common_hp_dict)
params_dict.update(model_specific_hp_dict)
