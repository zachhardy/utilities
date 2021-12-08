import os
import numpy as np

from numpy import ndarray
from typing import List

from .. import SimulationReader


class DatasetReader:
    """
    A class for reading and handling multiple simulations.
    """
    def __init__(self, path: str) -> None:
        if not os.path.isdir(path):
            raise NotADirectoryError(
                'The provided path is not a valid directory.')

        self.path: str = os.path.abspath(path)
        self.simulations: List[SimulationReader] = []
        self.parameters: ndarray = []

    @property
    def n_simulations(self) -> int:
        return len(self.simulations)

    @property
    def n_parameters(self) -> int:
        return self.parameters.shape[1]

    @property
    def n_snapshots(self) -> int:
        return self.simulations[0].n_snapshots

    def read_dataset(self) -> None:
        self.clear()

        # Get sorted file list
        entries = sorted(os.listdir(self.path))

        # Loop over simulations
        for simulation_num, simulation in enumerate(entries):
            path = os.path.join(self.path, simulation)
            if os.path.isdir(path) and 'reference' not in path:
                self.simulations.append(self.read_simulation(path))
                self.simulations[-1].read_simulation_data()
            elif simulation == 'params.txt':
                params = np.loadtxt(path)
                self.parameters = params.reshape(self.n_simulations, -1)

    def read_simulation(self, path: str) -> SimulationReader:
        cls_name = self.__class__.__name__
        raise NotImplementedError(
            f'{cls_name}.read_simulation must be implemented.')

    def create_dataset_matrix(self, variables: List[str] = None) -> ndarray:
        """
        Create a matrix whose columns contain stacked
        simulation results.

        Parameters
        ----------
        variables : List[str], default None
            The variables to stack. This varies based on the
            physics modules.

        Returns
        -------
        ndarray (n_simulations, n_snapshots * varies)
        """
        for n, simulation in enumerate(self.simulations):
            tmp = simulation.create_simulation_vector(variables)
            matrix = tmp if n == 0 else np.hstack((matrix, tmp))
        return matrix.T

    def unstack_simulation_vector(self, vector: ndarray) -> ndarray:
        """
        Unstack simulation vectors into snapshot matrices.

        Parameters
        ----------
        vector : ndarray (varies, n_snapshots * varies)
            A set of simulation vectors, where each row is an
            independent simulation.

        Returns
        -------
        ndarray (varies, n_snapshots, varies)
        """
        if vector.ndim == 1:
            vector = np.atleast_2d(vector)
        shape = (vector.shape[0], self.n_snapshots, -1)
        return vector.reshape(shape)

    def clear(self) -> None:
        self.__init__(self.path)
