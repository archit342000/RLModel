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

    def __init__(self, constants):

        self.start_time = time.time()

        self.C = constants

        # define path variables for various datasets
        self.test_h5_path = self.C.test_set_out[:-3] + "h5"

    def print_time_elapsed(self):
        """ Prints elapsed time since input `start_time`.
        """
        stop_time = time.time()
        elapsed_time = stop_time - self.start_time
        print(f"-- time elapsed: {elapsed_time:.5f} s", flush=True)


    def preprocess_test_data(self):
        """ Converts test dataset to HDF file format.
        """
        print("*Preprocessing test data.", flush=True)
        prep.create_HDF_file(self.C.test_set,self.C.test_set_out)

        self.print_time_elapsed()



    def preprocess_phase(self):
        """ Preprocesses all the datasets (validation, training, and testing).
        """
        if not self.C.restart:  # start preprocessing job from scratch
            if (os.path.exists(self.test_h5_path)):
                raise OSError(
                    f"There currently exist(s) pre-created *.h5 file(s) in the "
                    f"dataset directory. If you would like to proceed with "
                    f"creating new ones, please delete them and rerun the "
                    f"program. Otherwise, check your input file."
                )
            self.preprocess_test_data()
