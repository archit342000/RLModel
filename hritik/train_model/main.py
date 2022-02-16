# load general packages and functions
import datetime

# load program-specific functions
import util

util.suppress_warnings()
from parameters.constants import constants as C
from Workflow import Workflow

# defines and runs the job



def main():
    """ Defines the type of job (preprocessing, training, generation, testing, multiple validation, or computation of the validation loss), 
    runs it, and writes the job parameters used.
    """
    # fix date/time
    _ = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    workflow = Workflow(constants=C)

    job_type = C.job_type
    print(f"* Run mode: '{job_type}'", flush=True)


    if job_type == "train":
        # write training parameters
        util.write_job_parameters(params=C)

        # train model and generate graphs
        workflow.training_phase()

    else:
        return NotImplementedError("Not a valid `job_type`.")


if __name__ == "__main__":
    main()
