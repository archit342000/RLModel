import os
import rdkit
import numpy as np
import pandas as pd

#from azureml.core import Run
#run = Run.get_context()

import parameters.load as load


def get_max_n_atoms(path):
    molecule = load.molecules(path)
    max_n_atoms = 0
    for mol in molecule:
        n_atoms = mol.GetNumAtoms()
        if n_atoms > max_n_atoms:
            max_n_atoms = n_atoms

    return max_n_atoms
