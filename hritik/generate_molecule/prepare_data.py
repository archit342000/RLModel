import pandas as pd
import sys
import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--pred",type=str,dest="pred",help="Input splitted data")
parser.add_argument("--data",type=str,dest="data",help="Job Directory")
parser.add_argument("--generation",type=str,dest="generation",help="Directory for generated graphs")
args = parser.parse_args()

os.makedirs(args.data, exist_ok=True)

# load smiles
compound_iso_smiles = []
for file in os.listdir(args.generation):
    
    if file.endswith(".smi"):
        with open(args.generation+'/'+file, 'r') as f:
            data = f.read()

        temp = data.split('\n')[1:]
        for i in range(len(temp)):
            temp[i] = temp[i].split(' ')[0]

        valid_file = file.split('.')[0] + ".valid"
        with open(args.generation+'/'+valid_file, 'r') as f:
            data = f.read()
        
        valid = data.split('\n')

        for i in range(len(valid)):
            if(valid[i] == '1.0' and len(temp[i])>4):
                compound_iso_smiles.append(temp[i].upper())

compound_iso_smiles = list(set(compound_iso_smiles))


# laod targets
pdbs = []
pdbs_seqs = []
for seq in os.listdir(args.pred+'/pred'+'/sequences'):
    temp = ""
    with open(args.pred+ '/pred' + '/sequences/' + seq, 'r') as file:
        data = file.read()

    pdbs.append(data.split('\n')[0])
    for string in data.split('\n')[1:]:
        temp += string
    pdbs_seqs.append(temp)

# create dataset
compound_iso_smiles_temp = []
pdbs_temp = []
pdbs_seqs_temp = []
num_targets = len(pdbs)
num_smiles = len(compound_iso_smiles)
temp = []
for i in range(num_targets):
    compound_iso_smiles_temp += compound_iso_smiles
    
    temp = [pdbs[i][1:]]
    temp = temp * num_smiles
    pdbs_temp += temp
    
    temp = [pdbs_seqs[i]]
    temp = temp * num_smiles
    pdbs_seqs_temp += temp


pdbs = pdbs_temp
pdbs_seqs = pdbs_seqs_temp
compound_iso_smiles = compound_iso_smiles_temp
del(temp)
del(compound_iso_smiles_temp)
del(pdbs_temp)
del(pdbs_seqs_temp)


dict = {'compound_iso_smiles': compound_iso_smiles, 'target_name': pdbs, 'target_sequence': pdbs_seqs}
df = pd.DataFrame(dict)
df.to_csv(args.data+'/'+"davis"+'.csv')
