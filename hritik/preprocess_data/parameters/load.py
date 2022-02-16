import os
import rdkit
import numpy as np
import pandas as pd
from rdkit.Chem.rdmolfiles import SmilesMolSupplier

#from azureml.core import Run
#run = Run.get_context()

def molecules(path):

    molecule_set = SmilesMolSupplier(path, sanitize=True, nameColumn=-1, titleLine=True)

    return molecule_set
