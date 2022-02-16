# load general packages and functions
import csv
import sys
import os

# load program-specific functions
import parameters.load as load
from tools.atom_types import get_atom_types
from tools.max_n_nodes import get_max_n_atoms
from tools.formal_charges import get_formal_charges

# default model parameters, hyperparameters, and settings are defined here
# recommended not to modify the default settings here, but rather create
# an input file with the modified parameters in a new job directory (see README)
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--input-data",type=str,dest="in_data",help="Input splitted data")
parser.add_argument("--job-dir",type=str,dest="job_dir",help="Job Directory")
parser.add_argument("--train_dir",type=str,dest="train_dir",help="Train Directory")
parser.add_argument("--test_dir",type=str,dest="test_dir",help="Test Directory")
parser.add_argument("--valid_dir",type=str,dest="valid_dir",help="Valid Directory")
args = parser.parse_args()

atoms =  get_atom_types(args.in_data + "/train.csv")
nodes =  get_max_n_atoms(args.in_data + "/train.csv")
formal = get_formal_charges(args.in_data + "/train.csv")

os.makedirs(args.job_dir, exist_ok=True)
# general job parameters
params_dict = {
    "atom_types": atoms,
    "formal_charge": formal,
    "max_n_nodes": nodes,
    "imp_H": [0, 1, 2, 3],
    "chirality": ["None", "R", "S"],
    "group_size": 5000,

    "generation_epoch": 50,
    "n_samples": 100,  #5000,
    "n_workers": 2,
    "restart": False,
    "job_type": "train",
    "sample_every": 50,
    "use_aromatic_bonds": False,
    "use_canon": True,
    "use_chirality": False,
    "use_explicit_H": False,
    "ignore_H": True,
    "train": args.train_dir + '/',
    "test": args.test_dir + '/',
    "valid": args.valid_dir + '/',
    "job_dir" : args.job_dir + '/'
}

# model common hyperparameters
model_common_hp_dict = {
    "batch_size": 100,
    "block_size": 100000,
    "epochs": 100,
    "init_lr": 1e-4,
    "min_rel_lr": 5e-2,
    "max_rel_lr": 1,
    "weights_initialization": "uniform",
    "weight_decay": 0.0,
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
