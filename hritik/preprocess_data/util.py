# load general packages and functions
import csv
import numpy as np
import torch
import rdkit

# load program-specific functions
from parameters.constants import constants as C

# contains miscellaneous useful functions



def get_feature_vector_indices():
    """ Gets the indices of the different segments of the feature vector. The
    indices are analogous to the lengths of the various segments.

    Returns:
      idc (list) : Contains the indices of the different one-hot encoded
        segments used in the feature vector representations of nodes in
        `MolecularGraph`s. These segments are, in order, atom type, formal
        charge, number of implicit Hs, and chirality.
    """
    idc = [C.n_atom_types, C.n_formal_charge]

    # indices corresponding to implicit H's and chirality are optional (below)
    if not C.use_explicit_H and not C.ignore_H:
        idc.append(C.n_imp_H)

    if C.use_chirality:
        idc.append(C.n_chirality)

    return np.cumsum(idc).tolist()


def normalize_evaluation_metrics(prop_dict, epoch_key):
    """ Normalizes histograms in `props_dict` (see below) and converts them
    to `list`s (from `torch.Tensor`s) and rounds the elements. This is done for
    clarity when saving the histograms to CSV.

    Returns:
      norm_n_nodes_hist (torch.Tensor) : Normalized histogram of the number of
        nodes per molecule.
      norm_atom_type_hist (torch.Tensor) : Normalized histogram of the atom
        types present in the molecules.
      norm_charge_hist (torch.Tensor) : Normalized histogram of the formal
        charges present in the molecules.
      norm_numh_hist (torch.Tensor) : Normalized histogram of the number of
        implicit hydrogens present in the molecules.
      norm_n_edges_hist (torch.Tensor) : Normalized histogram of the number of
        edges per node in the molecules.
      norm_edge_feature_hist (torch.Tensor) : Normalized histogram of the
        edge features (types of bonds) present in the molecules.
      norm_chirality_hist (torch.Tensor) : Normalized histogram of the
        chiral centers present in the molecules.
    """
    # compute histograms for non-optional features
    norm_n_nodes_hist = [
        round(i, 2) for i in
        norm(prop_dict[(epoch_key, "n_nodes_hist")]).tolist()
    ]

    norm_atom_type_hist = [
        round(i, 2) for i in
        norm(prop_dict[(epoch_key, "atom_type_hist")]).tolist()
    ]

    norm_charge_hist = [
        round(i, 2) for i in
        norm(prop_dict[(epoch_key, "formal_charge_hist")]).tolist()
    ]

    norm_n_edges_hist = [
        round(i, 2) for i in
        norm(prop_dict[(epoch_key, "n_edges_hist")]).tolist()
    ]

    norm_edge_feature_hist = [
        round(i, 2) for i in
        norm(prop_dict[(epoch_key, "edge_feature_hist")]).tolist()
    ]

    # compute histograms for optional features
    if not C.use_explicit_H and not C.ignore_H:
        norm_numh_hist = [
            round(i, 2) for i in
            norm(prop_dict[(epoch_key, "numh_hist")]).tolist()
        ]
    else:
        norm_numh_hist = [0] * len(C.imp_H)

    if C.use_chirality:
        norm_chirality_hist = [
            round(i, 2) for i in
            norm(prop_dict[(epoch_key, "chirality_hist")]).tolist()
        ]
    else:
        norm_chirality_hist = [1, 0, 0]

    return (
        norm_n_nodes_hist,
        norm_atom_type_hist,
        norm_charge_hist,
        norm_numh_hist,
        norm_n_edges_hist,
        norm_edge_feature_hist,
        norm_chirality_hist,
    )


def norm(list_of_nums):
    """ Normalizes input `list_of_nums` (`list` of `float`s or `int`s)
    """
    try:
        norm_list_of_nums = list_of_nums / sum(list_of_nums)
    except:  # occurs if divide by zero
        norm_list_of_nums = list_of_nums

    return norm_list_of_nums


def one_of_k_encoding(x, allowable_set):
    """ Returns the one-of-k encoding of a value `x` having a range of possible
    values in `allowable_set`.

    Args:
      x (str, int) : Value to be one-hot encoded.
      allowable_set (list) : `list` of all possible values.

    Returns:
      one_hot_generator (generator) : One-hot encoding. A generator of `int`s.
    """
    if x not in set(allowable_set):  # use set for speedup over list
        raise Exception(
            f"Input {x} not in allowable set {allowable_set}. "
            f"Add {x} to allowable set in `features.py` and run again."
        )

    one_hot_generator = (int(x == s) for s in allowable_set)

    return one_hot_generator



def read_last_molecule_idx(restart_file_path):
    """ Reads the index of the last preprocessed molecule from a file called
    "index.restart" located in the same directory as the data.
    """
    with open(restart_file_path + "index.restart", "r") as txt_file:

        last_molecule_idx = txt_file.read()

    return int(last_molecule_idx)

def suppress_warnings():
    """ Suppresses unimportant warnings for a cleaner readout.
    """
    from rdkit import RDLogger
    from warnings import filterwarnings

    RDLogger.logger().setLevel(RDLogger.CRITICAL)
    filterwarnings(action="ignore", category=UserWarning)
    filterwarnings(action="ignore", category=FutureWarning)
    # could instead suppress ALL warnings with:
    # `filterwarnings(action="ignore")`
    # but choosing not to do this


def turn_off_empty_axes(n_plots_y, n_plots_x, ax):
    """ Turns off empty axes in a `n_plots_y` by `n_plots_x` grid of plots.

    Args:
      n_plots_y (int) : See above.
      n_plots_x (int) : See above.
      ax (matplotlib.axes) : Matplotlib object containing grid of plots.
    """
    for vi in range(n_plots_y):
        for vj in range(n_plots_x):
            # if nothing plotted on ax, it will contain `inf`
            # in axes lims, so clean up (turn off)
            if "inf" in str(ax[vi, vj].dataLim):
                ax[vi, vj].axis("off")

    return ax



def write_last_molecule_idx(last_molecule_idx, restart_file_path):
    """ Writes the index of the last preprocessed molecule (`last_molecule_idx`)
    to a file called "index.restart" to be located in the same directory as
    the data.
    """
    with open(restart_file_path + "index.restart", "w") as txt_file:
        txt_file.write(str(last_molecule_idx))


def write_job_parameters(params):
    """ Writes job parameters/hyperparameters in `params` (`namedtuple`) to
    CSV..
    """
    dict_path = params.job_dir + "params.csv"

    with open(dict_path, "w") as csv_file:

        writer = csv.writer(csv_file, delimiter=";")
        i = 0
        for key, value in enumerate(params._fields):
            writer.writerow([value, params[key]])
            i += 1


def write_preprocessing_parameters(params):
    """ Writes job parameters/hyperparameters in `params` (`namedtuple`) to
    CSV, so that parameters used during preprocessing can be referenced later.
    """
    dict_path = params.output_directory + "/preprocessing_params.csv"
    keys_to_write = ["atom_types",
                     "formal_charge",
                     "imp_H",
                     "chirality",
                     "group_size",
                     "max_n_nodes",
                     "use_aromatic_bonds",
                     "use_chirality",
                     "use_explicit_H",
                     "ignore_H"]

    with open(dict_path, "w") as csv_file:

        writer = csv.writer(csv_file, delimiter=";")
        for key, value in enumerate(params._fields):
            if value in keys_to_write:
                writer.writerow([value, params[key]])


def write_ts_properties(ts_properties_dict):
    """ Writes the training set properties in `ts_properties_dict` to CSV.
    """
    training_set = C.training_set_out  # path to "train.smi"
    dict_path = f"{training_set[:-4]}.csv"

    with open(dict_path, "w") as csv_file:

        csv_writer = csv.writer(csv_file, delimiter=";")
        for key, value in ts_properties_dict.items():
            if "validity_tensor" in key:
                # skip writing the validity tensor here because it is really
                # long, instead it gets its own file elsewhere
                continue
            elif type(value) == np.ndarray:
                csv_writer.writerow([key, list(value)])
            elif type(value) == torch.Tensor:
                try:
                    csv_writer.writerow([key, float(value)])
                except ValueError:
                    csv_writer.writerow([key, [float(i) for i in value]])
            else:
                csv_writer.writerow([key, value])
