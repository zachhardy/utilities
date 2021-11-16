import numpy as np
from numpy import ndarray

from typing import TYPE_CHECKING, List
if TYPE_CHECKING:
    from . import SimulationReader


def create_simulation_matrix(self: "SimulationReader",
                             variables: List[str] = None) -> ndarray:
    """
    Create a simulation matrix.

    Parameters
    ----------
    variables : List[str], default None
        The variables to stack. The default behavior stacks
        only the groupwise scalar flux values.

    Returns
    -------
    ndarray (n_snapshots, varies)
    """
    if variables is None:
        variables = ['flux_m0']
    elif isinstance(variables, str):
        variables = [variables]

    for v, var in enumerate(variables):
        tmp = self.get_variable_by_key(var)
        matrix = tmp if v == 0 else np.hstack((matrix, tmp))
    return matrix


def create_simulation_vector(self: 'SimulationReader',
                             variables: List[str] = None) -> ndarray:
    """
    Create a simulation matrix.

    Parameters
    ----------
    variables : List[str], default None
        The variables to stack. The default behavior stacks
        only the groupwise scalar flux values.

    Returns
    -------
    ndarray (n_snapshots * varies,)
    """
    data = self.create_simulation_matrix(variables)
    return data.reshape(data.size, 1)








