import logging
import random

import numpy as np

log = logging.getLogger(__name__)


def fix_seed(seed: int = 0) -> None:
    """fix random seed.

    Parameters
    ----------
    seed : int
        seed

    Returns
    -------
    None

    """
    random.seed(seed)
    np.random.seed(seed)


def adjust_qiskit_logging_level(logging_level: int = logging.WARNING) -> None:
    """
    adjust qiskit logging level.
    If we use hydra, loging_level will be INFO, and we will get
    too much log from qiskit. So we suppress these logging with this function.
    """
    target_qiskit_logger = ["qiskit.providers.ibmq", "qiskit.transpiler", "qiskit.compiler", "qiskit.execute_function"]
    for logger in target_qiskit_logger:
        logging.getLogger(logger).setLevel(logging_level)
