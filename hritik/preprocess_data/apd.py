# load general packages and functions
import torch
import copy

#from azureml.core import Run
#run = Run.get_context()

# load program-specific functions
from parameters.constants import constants as C

# defines functions related to the action probability distributions (APDs)



def get_decoding_route_length(molecular_graph):

    return molecular_graph.get_n_edges() + 2


def get_decoding_route_state(molecular_graph, subgraph_idx):

    molecular_graph = copy.deepcopy(molecular_graph)

    if subgraph_idx != 0:
        # find which subgraph is indicated by the index by progressively
        # truncating the input molecular graph
        for _ in range(1, subgraph_idx):
            molecular_graph.truncate_graph()

        # get the APD before the last truncation (since APD says how to get to
        # the *next* graph, need to truncate once more after obtaining APD)
        decoding_APD = molecular_graph.get_decoding_APD()
        molecular_graph.truncate_graph()

        X, E = molecular_graph.get_graph_state()

    elif subgraph_idx == 0:
        # return the first subgraph
        decoding_APD = molecular_graph.get_final_decoding_APD()

        X, E = molecular_graph.get_graph_state()

    else:
        raise ValueError("`subgraph_idx` not a valid value.")

    decoding_graph = [X, E]

    return decoding_graph, decoding_APD


def split_APD_vector(APD_output, as_vec):

    # get the number of elements which should be in each APD component
    f_add_elems = int(C.dim_f_add_p0)
    f_conn_elems = int(C.dim_f_conn_p0)

    # split up the target vector into three and reshape
    f_add, f_conn_and_term = torch.split(APD_output, f_add_elems, dim=1)
    f_conn, f_term = torch.split(f_conn_and_term, f_conn_elems, dim=1)

    if as_vec:  # output as vectors, without reshaping
        return f_add, f_conn, f_term

    else:  # reshape the APDs from vectors into proper tensors
        f_add = f_add.view(C.dim_f_add)
        f_conn = f_conn.view(C.dim_f_conn)
        f_term = f_term.view(C.dim_f_term)

        return f_add, f_conn, f_term
