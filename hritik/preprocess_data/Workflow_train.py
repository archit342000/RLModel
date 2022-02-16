# load general packages and functions
import time
import torch
import os

from azureml.core import Run
run = Run.get_context()

# load program-specific functions
import preprocessing as prep


# defines `Workflow` class

# set default torch dtype
torch.set_default_dtype(torch.float32)

class Workflow:
    """ Single `Workflow` class split up into different functions for
      1) preprocessing various molecular datasets

    The preprocessing step reads a set of molecules and generates training data
    for each molecule in HDF file format, consisting of decoding routes and
    APDs.
    """
    def __init__(self, constants):

        self.start_time = time.time()

        self.C = constants

        self.train_h5_path = self.C.training_set_out[:-3] + "h5"


    def print_time_elapsed(self):
        """ Prints elapsed time since input `start_time`.
        """
        stop_time = time.time()
        elapsed_time = stop_time - self.start_time
        print(f"-- time elapsed: {elapsed_time:.5f} s", flush=True)

    def preprocess_train_data(self):
        """ Converts training dataset to HDF file format.
        """
        print("* Preprocessing training data.", flush=True)
        prep.create_HDF_file(self.C.training_set,self.C.training_set_out, is_training_set=True)

        self.print_time_elapsed()


    def preprocess_phase(self):
        """ Preprocesses all the datasets (validation, training, and testing).
        """
        if not self.C.restart:  # start preprocessing job from scratch
            if (os.path.exists(self.train_h5_path)):
                raise OSError(
                    f"There currently exist(s) pre-created *.h5 file(s) in the "
                    f"dataset directory. If you would like to proceed with "
                    f"creating new ones, please delete them and rerun the "
                    f"program. Otherwise, check your input file."
                )
            self.preprocess_train_data()
