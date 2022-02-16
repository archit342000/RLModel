# load general packages and functions

import csv
import sys
import os
import rdkit
import numpy as np
from collections import namedtuple
from rdkit.Chem.rdchem import BondType
from parameters.config import params_dict as params_dict



def get_feature_dimensions(parameters):
    n_atom_types = len(parameters["atom_types"])
    n_formal_charge = len(parameters["formal_charge"])
    n_numh = int(
        not parameters["use_explicit_H"]
        and not parameters["ignore_H"]
    ) * len(parameters["imp_H"])
    n_chirality = int(parameters["use_chirality"]) * len(parameters["chirality"])

    return n_atom_types, n_formal_charge, n_numh, n_chirality


def get_tensor_dimensions(n_atom_types,
                          n_formal_charge,
                          n_num_h,
                          n_chirality,
                          n_node_features,
                          n_edge_features,
                          parameters):
    max_nodes = parameters["max_n_nodes"]

    # define the matrix dimensions as `list`s
    # first for the graph reps...
    dim_nodes = [max_nodes, n_node_features]

    dim_edges = [max_nodes, max_nodes, n_edge_features]

    # ... then for the APDs
    if parameters["use_chirality"]:
        if parameters["use_explicit_H"] or parameters["ignore_H"]:
            dim_f_add = [
                parameters["max_n_nodes"],
                n_atom_types,
                n_formal_charge,
                n_chirality,
                n_edge_features,
            ]
        else:
            dim_f_add = [
                parameters["max_n_nodes"],
                n_atom_types,
                n_formal_charge,
                n_num_h,
                n_chirality,
                n_edge_features,
            ]
    else:
        if parameters["use_explicit_H"] or parameters["ignore_H"]:
            dim_f_add = [
                parameters["max_n_nodes"],
                n_atom_types,
                n_formal_charge,
                n_edge_features,
            ]
        else:
            dim_f_add = [
                parameters["max_n_nodes"],
                n_atom_types,
                n_formal_charge,
                n_num_h,
                n_edge_features,
            ]

    dim_f_conn = [parameters["max_n_nodes"], n_edge_features]

    dim_f_term = 1

    return dim_nodes, dim_edges, dim_f_add, dim_f_conn, dim_f_term


def load_params(input_csv_path):
    params_to_override_dict = {}
    with open(input_csv_path, "r") as csv_file:

        params_reader = csv.reader(csv_file, delimiter=";")

        for key, value in params_reader:
            try:
                params_to_override_dict[key] = eval(value)
            except NameError:  # `value` is a `str`
                params_to_override_dict[key] = value
            except SyntaxError:  # to avoid "unexpected `EOF`"
                params_to_override_dict[key] = value

    return params_to_override_dict


def override_params(all_params):
    input_csv_path = all_params["job_dir"] + "input.csv"

    # check if there exists and `input.csv` in working directory
    if os.path.exists(input_csv_path):
        # override default values for parameters in `input.csv`
        params_to_override_dict = load_params(input_csv_path)
        for key, value in params_to_override_dict.items():
            all_params[key] = value

    return all_params


def collect_global_constants(parameters, job_dir):
    # first override any arguments from `input.csv`:
    parameters["job_dir"] = job_dir
    parameters= override_params(all_params=parameters)

    # then calculate any global constants below:
    if parameters["use_explicit_H"] and parameters["ignore_H"]:
        raise ValueError(
            f"Cannot use explicit H's and ignore H's "
            f"at the same time. Please fix flags."
        )

    # define edge feature (rdkit `GetBondType()` result -> `int`) constants
    bondtype_to_int = {BondType.SINGLE: 0, BondType.DOUBLE: 1, BondType.TRIPLE: 2}

    if parameters["use_aromatic_bonds"]:
        bondtype_to_int[BondType.AROMATIC] = 3

    int_to_bondtype = dict(map(reversed, bondtype_to_int.items()))

    n_edge_features = len(bondtype_to_int)

    # define node feature constants
    n_atom_types, n_formal_charge, n_imp_H, n_chirality = get_feature_dimensions(parameters)

    n_node_features = n_atom_types + n_formal_charge + n_imp_H + n_chirality

    # define matrix dimensions
    dim_nodes, dim_edges, dim_f_add, dim_f_conn, dim_f_term = get_tensor_dimensions(
        n_atom_types,
        n_formal_charge,
        n_imp_H,
        n_chirality,
        n_node_features,
        n_edge_features,
        parameters,
    )

    dim_f_add_p0 = np.prod(dim_f_add[:])
    dim_f_add_p1 = np.prod(dim_f_add[1:])
    dim_f_conn_p0 = np.prod(dim_f_conn[:])
    dim_f_conn_p1 = np.prod(dim_f_conn[1:])

    # create a dictionary of global constants, and add `job_dir` to it; this
    # will ultimately be converted to a `namedtuple`
    constants_dict = {
        "big_negative": -1e6,
        "big_positive": 1e6,
        "bondtype_to_int": bondtype_to_int,
        "int_to_bondtype": int_to_bondtype,
        "n_edge_features": n_edge_features,
        "n_atom_types": n_atom_types,
        "n_formal_charge": n_formal_charge,
        "n_imp_H": n_imp_H,
        "n_chirality": n_chirality,
        "n_node_features": n_node_features,
        "dim_nodes": dim_nodes,
        "dim_edges": dim_edges,
        "dim_f_add": dim_f_add,
        "dim_f_conn": dim_f_conn,
        "dim_f_term": dim_f_term,
        "dim_f_add_p0": dim_f_add_p0,
        "dim_f_add_p1": dim_f_add_p1,
        "dim_f_conn_p0": dim_f_conn_p0,
        "dim_f_conn_p1": dim_f_conn_p1,
    }

    # join with `features.args_dict`
    constants_dict.update(parameters)

    # define path to dataset splits
    constants_dict["test_set"] = parameters["input_directory"] + "/test.csv"
    constants_dict["training_set"] = parameters["input_directory"] + "/train.csv"
    constants_dict["validation_set"] = parameters["input_directory"] + "/valid.csv"
    constants_dict["test_set_out"] = parameters["output_directory"] + "/test.csv"
    constants_dict["training_set_out"] = parameters["output_directory"] + "/train.csv"
    constants_dict["validation_set_out"] = parameters["output_directory"] + "/valid.csv"

    # check (if a job is not a preprocessing job) that parameters  match those for
    # the original preprocessing job
    if constants_dict["job_type"] != "preprocess":
        print(
            "* Running job using HDF datasets located at "
            + parameters["output_directory"],
            flush=True,
        )
        print(
            "* Checking that the relevant parameters match "
            "those used in preprocessing the dataset.",
            flush=True,
        )

        # load preprocessing parameters for comparison (if they exist already)
        csv_file = parameters["input_directory"] + "/preprocessing_params.csv"
        params_to_check = load_params(input_csv_path=csv_file)

        for key, value in params_to_check.items():
            if key in constants_dict.keys() and value != constants_dict[key]:
                raise ValueError(
                    f"Check that training job parameters match those used in "
                    f"preprocessing. {key} does not match."
                )

        # if above error never raised, then all relevant parameters match! :)
        print("-- Job parameters match preprocessing parameters.", flush=True)

    # convert `CONSTANTS` dictionary into a namedtuple (immutable + cleaner)
    Constants = namedtuple("CONSTANTS", sorted(constants_dict))
    constants = Constants(**constants_dict)

    return constants
constants = collect_global_constants(parameters=params_dict,job_dir=params_dict["job_dir"])
