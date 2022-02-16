import csv
import sys
import os
from tools.atom_types import get_atom_types
from tools.max_n_nodes import get_max_n_atoms
from tools.formal_charges import get_formal_charges


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--job-dir",type=str,default='./job_dir',dest="job_dir",help="Job Directory")
parser.add_argument("--input-data",type=str,dest="in_data",help="Input splitted data for preprocessing")
parser.add_argument("--output-data",type=str,dest="out_data",help="Where to save preprocessed data")
args = parser.parse_args()

os.makedirs(args.in_data, exist_ok=True)
os.makedirs(args.out_data, exist_ok=True)

atoms =  get_atom_types(args.in_data + "/train.csv")
nodes =  get_max_n_atoms(args.in_data + "/train.csv")
formal = get_formal_charges(args.in_data + "/train.csv")

params_dict = {
    "job_type": "preprocess",
    "atom_types": atoms,
    "formal_charge": formal,
    "max_n_nodes": nodes,
    "imp_H": [0, 1, 2, 3],
    "chirality": ["None", "R", "S"],
    "group_size": 5000,
    "use_aromatic_bonds": False,
    "use_canon": True,
    "use_chirality": False,
    "use_explicit_H": False,
    "ignore_H": True,
    "restart": False,
    "job_dir": args.job_dir
    
}

directory = {"input_directory": args.in_data,
"output_directory": args.out_data}

params_dict.update(directory)
