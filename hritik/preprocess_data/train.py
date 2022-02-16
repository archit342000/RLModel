# load general packages and functions
import datetime

# load program-specific functions
import util

from azureml.core import Run
run = Run.get_context()


util.suppress_warnings()
from parameters.constants import constants as C
from Workflow_train import Workflow

def main():
    """ Defines the type of job (preprocessing)
    runs it, and writes the job parameters used.
    """
    # fix date/time
    _ = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    workflow = Workflow(constants=C)

    job_type = C.job_type
    print(f"* Run mode: '{job_type}'", flush=True)

    if job_type == "preprocess":
        # write preprocessing parameters
        util.write_preprocessing_parameters(params=C)

        # preprocess all datasets
        workflow.preprocess_phase()

    else:
        return NotImplementedError("Not a valid `job_type`.")


if __name__ == "__main__":
    main()
