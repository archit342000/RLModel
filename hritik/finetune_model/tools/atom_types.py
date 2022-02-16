import os
import rdkit
import numpy as np
import pandas as pd

#from azureml.core import Run
#run = Run.get_context()

import parameters.load as load

def get_atom_types(path):
    molecule = load.molecules(path)
    atom_types = list()
    for mol in molecule:
        for atom in mol.GetAtoms():
            atom_types.append(atom.GetAtomicNum())

    set_of_atom_types = set(atom_types)
    atom_types_sorted = list(set_of_atom_types)
    atom_types_sorted.sort()

    return [rdkit.Chem.Atom(atom).GetSymbol() for atom in atom_types_sorted]
