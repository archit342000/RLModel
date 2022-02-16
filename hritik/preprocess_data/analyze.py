# load general packages and functions
import numpy as np
import time
import torch
import rdkit

#from azureml.core import Run
#run = Run.get_context()

# load program-specific functions
from parameters.constants import constants as C
import util

# functions for evaluating sets of structures, including
# sets of training, validation, and generation structures


def get_edge_feature_distribution(molecular_graphs):
 
    # initialize histogram
    edge_feature_hist = torch.zeros(C.n_edge_features, device="cuda")

    for molecular_graph in molecular_graphs:

        edges = molecular_graph.edge_features

        for edge in range(C.n_edge_features):
            try:  # `GenerationGraph`s
                edge_feature_hist[edge] += torch.sum(edges[:, :, edge])/2
            except TypeError:  # `PreprocessingGraph`s
                edge_feature_hist[edge] += np.sum(edges[:, :, edge])/2

    return edge_feature_hist


def get_fraction_unique(molecular_graphs):

    smiles_list = []

    for molecular_graph in molecular_graphs:

        smiles = molecular_graph.get_smiles()
        smiles_list.append(smiles)

    smiles_set = set(smiles_list)
    try:
        smiles_set.remove(None)  # remove placeholder for invalid SMILES
    except KeyError:  # no invalid SMILES in set!
        pass

    n_repeats = len(smiles_set)

    try:
        fraction_unique = n_repeats / len(smiles_list)
    except (ValueError, ZeroDivisionError):
        fraction_unique = 0

    return fraction_unique


def get_fraction_valid(molecular_graphs, termination):

    n_invalid = 0  # start counting
    n_valid_and_properly_terminated = 0  # start counting
    n_graphs = len(molecular_graphs)

    for idx, molecular_graph in enumerate(molecular_graphs):

        mol = molecular_graph.get_molecule()

        # determine if valid
        try:
            rdkit.Chem.SanitizeMol(mol)
            n_valid_and_properly_terminated += termination[idx]
        except:  # invalid molecule
            n_invalid += 1

    fraction_valid = (n_graphs - n_invalid) / n_graphs

    if 1 in termination:
        fraction_valid_properly_terminated = n_valid_and_properly_terminated / sum(termination)
    else:
        fraction_valid_properly_terminated = 0.0

    fraction_properly_terminated = sum(termination)/len(termination)

    return fraction_valid, fraction_valid_properly_terminated, fraction_properly_terminated


def get_molecular_properties(molecules, epoch_key, termination=None):

    # get the distribution of the number of atoms per graph
    n_nodes_hist, avg_n_nodes = get_n_nodes_distribution(molecular_graphs=molecules)

    # get the distributions of node features (e.g. atom types) in the graphs
    atom_type_hist, formal_charge_hist, numh_hist, chirality_hist = get_node_feature_distribution(molecular_graphs=molecules)

    # get the distribution of the number of edges per node and the average
    # number of edges per graph
    n_edges_hist, avg_n_edges = get_n_edges_distribution(molecular_graphs=molecules,
                                                         n_edges_to_bin=10)

    # get the distribution of bond types present in the graphs
    edge_feature_hist = get_edge_feature_distribution(molecular_graphs=molecules)

    # get the fraction of unique molecules in the input graphs
    fraction_unique = get_fraction_unique(molecular_graphs=molecules)

    if epoch_key == "Training set":
        fraction_valid = 1.0
        fraction_valid_pt = 1.0
        fraction_pt = 1.0
    else:
        # get the fraction of valid molecules in the graphs
        (
            fraction_valid,     # fraction valid
            fraction_valid_pt,  # fraction valid and properly terminated
            fraction_pt         # fraction properly terminated
        ) = get_fraction_valid(molecular_graphs=molecules, termination=termination)

    properties_dict = {
        (epoch_key, "n_nodes_hist"): n_nodes_hist,
        (epoch_key, "avg_n_nodes"): avg_n_nodes,
        (epoch_key, "atom_type_hist"): atom_type_hist,
        (epoch_key, "formal_charge_hist"): formal_charge_hist,
        (epoch_key, "n_edges_hist"): n_edges_hist,
        (epoch_key, "avg_n_edges"): avg_n_edges,
        (epoch_key, "edge_feature_hist"): edge_feature_hist,
        (epoch_key, "fraction_unique"): fraction_unique,
        (epoch_key, "fraction_valid"): fraction_valid,
        (epoch_key, "fraction_valid_properly_terminated"): fraction_valid_pt,
        (epoch_key, "fraction_properly_terminated"): fraction_pt,
        (epoch_key, "numh_hist"): numh_hist,
        (epoch_key, "chirality_hist"): chirality_hist
    }

    return properties_dict

def evaluate_training_set(preprocessing_graphs):
    """ Computes molecular properties for structures in training set.

    Args:
      training_graphs (list) : Contains `PreprocessingGraph`s.

    Returns:
      ts_prop_dict (dict) : Dictionary of training set molecular properties.
    """
    ts_prop_dict = get_molecular_properties(molecules=preprocessing_graphs,
                                            epoch_key="Training set")
    return ts_prop_dict

def combine_ts_properties(prev_properties, next_properties, weight_next):

    # convert any CUDA (torch.Tensor)s to CPU
    for dictionary in [prev_properties, next_properties]:
        for key, value in dictionary.items():
            try:
                if value.is_cuda:
                    dictionary[key] = value.cpu()
            except AttributeError:
                pass

    # `weight_prev` says how much to weight the properties of the old structures
    # when calculating the average with the new structures
    weight_prev = C.group_size

    # bundle properties in a tuple for some readibility
    bundle_properties = (prev_properties, next_properties, weight_prev, weight_next)

    # take a weighted average of the "old properties" with the "new properties"
    n_nodes_hist = weighted_average(b=bundle_properties, key="n_nodes_hist")
    avg_n_nodes = weighted_average(b=bundle_properties, key="avg_n_nodes")
    atom_type_hist = weighted_average(b=bundle_properties, key="atom_type_hist")
    formal_charge_hist = weighted_average(b=bundle_properties, key="formal_charge_hist")
    n_edges_hist = weighted_average(b=bundle_properties, key="n_edges_hist")
    avg_n_edges = weighted_average(b=bundle_properties, key="avg_n_edges")
    edge_feature_hist = weighted_average(b=bundle_properties, key="edge_feature_hist")
    fraction_unique = weighted_average(b=bundle_properties, key="fraction_unique")
    fraction_valid = weighted_average(b=bundle_properties, key="fraction_valid")
    numh_hist = weighted_average(b=bundle_properties, key="numh_hist")
    chirality_hist = weighted_average(b=bundle_properties, key="chirality_hist")

    # return the weighted averages in a new dictionary
    ts_properties = {
        ("Training set", "n_nodes_hist"): n_nodes_hist,
        ("Training set", "avg_n_nodes"): avg_n_nodes,
        ("Training set", "atom_type_hist"): atom_type_hist,
        ("Training set", "formal_charge_hist"): formal_charge_hist,
        ("Training set", "n_edges_hist"): n_edges_hist,
        ("Training set", "avg_n_edges"): avg_n_edges,
        ("Training set", "edge_feature_hist"): edge_feature_hist,
        ("Training set", "fraction_unique"): fraction_unique,
        ("Training set", "fraction_valid"): fraction_valid,
        ("Training set", "numh_hist"): numh_hist,
        ("Training set", "chirality_hist"): chirality_hist
    }

    return ts_properties


def weighted_average(b, key):

    (p, n, wp, wn) = b

    weighted_average = np.around((
        np.array(p[("Training set", key)]) * wp
        + np.array(n[("Training set", key)]) * wn
    ) / (wp + wn), decimals=3)

    return weighted_average


def get_n_edges_distribution(molecular_graphs, n_edges_to_bin=10):

    # initialize and populate histogram (last bin is for # num edges > `n_edges_to_bin`)
    n_edges_histogram = torch.zeros(n_edges_to_bin, device="cuda")

    for molecular_graph in molecular_graphs:

        edges = molecular_graph.edge_features

        for vi in range(molecular_graph.n_nodes):

            n_edges = 0
            for bond_type in range(C.n_edge_features):
                try:
                    n_edges += int(torch.sum(edges[vi, :, bond_type]))
                except TypeError:  # if edges is `np.ndarray`
                    n_edges += int(np.sum(edges[vi, :, bond_type]))

            if n_edges > n_edges_to_bin:
                n_edges = n_edges_to_bin
            n_edges_histogram[n_edges - 1] += 1

    # compute average number of edges per node
    sum_n_edges = 0
    for n_edges, count in enumerate(n_edges_histogram, start=1):
        sum_n_edges += n_edges * count

    try:
        avg_n_edges = sum_n_edges / torch.sum(n_edges_histogram, dim=0)
    except ValueError:
        avg_n_edges = 0

    return n_edges_histogram, avg_n_edges


def get_n_nodes_distribution(molecular_graphs):

    # initialize and populate histogram
    n_nodes_histogram = torch.zeros(C.max_n_nodes + 1, device="cuda")

    for molecular_graph in molecular_graphs:
        n_nodes = molecular_graph.n_nodes
        n_nodes_histogram[n_nodes] += 1

    # compute the average number of nodes per graph
    sum_n_nodes = 0
    for key, count in enumerate(n_nodes_histogram):
        n_nodes = key
        sum_n_nodes += n_nodes * count

    avg_n_nodes = sum_n_nodes / len(molecular_graphs)

    return n_nodes_histogram, avg_n_nodes


def get_node_feature_distribution(molecular_graphs):

    # sum up all node feature vectors to get an un-normalized histogram
    if type(molecular_graphs[0].node_features) == torch.Tensor:
        nodes_hist = torch.zeros(C.n_node_features, device="cuda")
    else:
        nodes_hist = np.zeros(C.n_node_features)

    # loop over all the node feature matrices of the input `TrainingGraph`s
    for molecular_graph in molecular_graphs:
        try:
            nodes_hist += torch.sum(molecular_graph.node_features, dim=0)
        except TypeError:
            nodes_hist += np.sum(molecular_graph.node_features, axis=0)

    idc = util.get_feature_vector_indices()  # **note: "idc" == "indices"

    # split up `nodes_hist` into atom types hist, formal charge hist, etc
    # `atom_type_histogram` and `formal_charge_histogram` are calculated by
    # default, and if specified, also `numh_histogram` and `chirality_histogram`
    atom_type_histogram = nodes_hist[:idc[0]]

    formal_charge_histogram = nodes_hist[idc[0]:idc[1]]

    if not C.use_explicit_H and not C.ignore_H:
        numh_histogram = nodes_hist[idc[1]:idc[2]]
    else:
        numh_histogram = [0] * C.n_imp_H

    if C.use_chirality:
        correction = int(not C.use_explicit_H and not C.ignore_H)
        chirality_histogram = nodes_hist[idc[1 + correction]:idc[2 + correction]]
    else:
        chirality_histogram = [0] * C.n_chirality

    return (atom_type_histogram, formal_charge_histogram, numh_histogram, chirality_histogram)
