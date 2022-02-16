import numpy as np
import rdkit
import pandas as pd
import os

#from azureml.core import Run
#run = Run.get_context()

import parameters.load as load


def get_formal_charges(path):
    molecule = load.molecules(path)
    formal_charges = list()
    for mol in molecule:
        for atom in mol.GetAtoms():
            formal_charges.append(atom.GetFormalCharge())

    set_of_formal_charges = set(formal_charges)
    formal_charges_sorted = list(set_of_formal_charges)
    formal_charges_sorted.sort()

    return formal_charges_sorted
