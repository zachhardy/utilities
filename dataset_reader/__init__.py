import os
import numpy as np

from numpy import ndarray
from typing import List

from pyPDEs.utilities import Vector
from simulation_reader import SimulationReader


class DatasetReader:
    """
    A class for reading and handling multiple simulations.
    """

    def __init__(self, path: str) -> None:
        if not os.path.isdir(path):
            raise NotADirectoryError(
                'The provided path is not a valid directory.')

        self.path: str = os.path.abspath(path)
        self.simulations : List[SimulationReader] = []
        self.parameters: ndarray = []

    def read_dataset(self) -> None:
        """
        Read a set of parameterized simulation results.
        """
        self.clear()

        # Get sorted file list
        entries = sorted(os.listdir(self.path))

        for simulation_num, simulation in enumerate(entries):
            path = os.path.join(self.path, simulation)
            if os.path.isdir(path) and 'reference' not in path:
                self.simulations.append(SimulationReader(path))
                self.simulations[-1].read_simulation_data()
            elif simulation == 'params.txt':
                params = np.loadtxt(path)
                self.parameters = params.reshape(self.n_simulations, -1)


    def create_dataset_matrix(self, variables: List[str] = None) -> ndarray:
        """
        Create a matrix for POD reduced order modeling.

        Parameters
        ----------
        variables : List[str], default None
            The variables to stack. The default behavior stacks
            only the groupwise scalar flux values.

        Returns
        -------
        ndarray (n_snapshots * varies, n_simulations)
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
        return vector.reshape(vector.shape[0], self.n_snapshots, -1)

    @property
    def n_simulations(self) -> int:
        return len(self.simulations)

    @property
    def n_parameters(self) -> int:
        return self.parameters.shape[1]

    @property
    def dim(self) -> int:
        return self.simulations[0].dim

    @property
    def n_snapshots(self) -> int:
        return self.simulations[0].n_snapshots

    @property
    def n_cells(self) -> int:
        return self.simulations[0].n_cells

    @property
    def n_nodes(self) -> int:
        return self.simulations[0].n_nodes

    @property
    def nodes_per_cell(self) -> int:
        return self.simulations[0].nodes_per_cell

    @property
    def n_moments(self) -> int:
        return self.simulations[0].n_moments

    @property
    def n_groups(self) -> int:
        return self.simulations[0].n_groups

    @property
    def n_precursors(self) -> int:
        return self.simulations[0].n_precursors

    @property
    def max_precursors(self) -> int:
        return self.simulations[0].max_precursors

    @property
    def n_materials(self) -> int:
        return self.simulations[0].n_materials

    @property
    def times(self) -> List[float]:
        return self.simulations[0].times

    @property
    def nodes(self) -> List[Vector]:
        return self.simulations[0].nodes

    @property
    def centroids(self) -> List[Vector]:
        return self.simulations[0].centroids

    @property
    def material_ids(self) -> List[int]:
        return self.simulations[0].material_ids

    def clear(self) -> None:
        self.__init__(self.path)
