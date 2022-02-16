# load general packages and functions
import csv
from rdkit.Chem.rdmolfiles import SmilesMolSupplier

# functions for loading SMILES and model type



def molecules(path):

    # read file
    molecule_set = SmilesMolSupplier(path, sanitize=True, nameColumn=-1, titleLine=True)

    return molecule_set


def which_model(input_csv_path):
    """ Gets the type of model to use by reading it from CSV (in "input.csv").
    """
    with open(input_csv_path, "r") as csv_file:

        params_reader = csv.reader(csv_file, delimiter=";")

        for key, value in params_reader:
            if key == "model":
                return value  # string describing model e.g. "GGNN"

