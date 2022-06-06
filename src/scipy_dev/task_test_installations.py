import pytask
from scipy_dev.auxiliary import is_installed


for executable in ["marp", "decktape"]:

    @pytask.mark.task
    def task_test_executable_is_installed(executable=executable):
        if not is_installed(executable=executable):
            raise ValueError(f"{executable} is not installed.")
