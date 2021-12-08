import os

import numpy as np
from numpy import ndarray
from typing import List

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from ...base import SimulationReader


class PRKESimulationReader(SimulationReader):

    from ._plotters import (plot_powers,
                            plot_precursors,
                            plot_fuel_temperatures,
                            plot_coolant_temperatures,
                            plot_results)

    def __init__(self, path: str) -> None:
        super().__init__(path)

        self.n_snapshots: int = 0
        self.n_precursors: int = 0

        self.times: ndarray = []

        self.powers: ndarray = []
        self.fuel_temperatures: ndarray = []
        self.coolant_temperatures: ndarray = []
        self.precursors: ndarray = []

    def initialize_storage(self) -> None:
        """
        Initialize storage based on macro-quantities.
        """
        T, J = self.n_snapshots, self.n_precursors

        self.times = np.empty(T, dtype=float)
        self.powers = np.empty(T, dtype=float)
        self.fuel_temperatures = np.empty(T, dtype=float)
        self.coolant_temperatures = np.empty(T, dtype=float)
        self.precursors = np.empty((T, J), dtype=float)

    def read_simulation_data(self) -> None:
        """
        Parse the bindary files of a simulation.
        """
        self.clear()

        # Get sorted file list
        files = sorted(os.listdir(self.path))
        self.n_snapshots = len(files)

        # Loop over files
        for snapshot_num, snapshot in enumerate(files):
            path = os.path.join(self.path, snapshot)

            # Open and read snapshot file
            with open(path, mode='rb') as f:
                f.read(499) # skip header

                step = self.read_uint64_t(f)
                n_precursors = self.read_uint64_t(f)

                # Set macro-data
                if snapshot_num == 0:
                    self.n_precursors = n_precursors
                    self.initialize_storage()

                # Check for compatibility
                else:
                    assert n_precursors == self.n_precursors

                # Set snapshot data
                self.times[step] = self.read_double(f)
                self.powers[step] = self.read_double(f)
                self.fuel_temperatures[step] = self.read_double(f)
                self.coolant_temperatures[step] = self.read_double(f)
                for j in range(n_precursors):
                    precursor = self.read_unsigned_int(f)
                    self.precursors[step, precursor] = self.read_double(f)

    def get_variable_by_key(self, key: str) -> ndarray:
        if key == 'power':
            return self.powers.reshape(-1, 1)
        elif key == 'fuel_temperature':
            return self.fuel_temperatures.reshape(-1, 1)
        elif key == 'coolant_temperature':
            return self.coolant_temperatures.reshape(-1, 1)
        elif 'precursors' in key:
            if key == 'precursors':
                return self.precursors
            elif 'j' in key:
                j = int(key[key.find('j') + 1])
                return self.precursors[:, j].reshape(-1, 1)

    def create_simulation_matrix(
            self, variables: List[str] = None) -> ndarray:
        """
        Create a simulation matrix.

        Parameters
        ----------
        variables : List[str], default None
            The variables to stack. The default behavior
            stacks all unknowns

        Returns
        -------
        ndarray (n_snapshots, varies)
        """
        if isinstance(variables, str):
            variables = [variables]
        elif variables is None:
            variables = ['power', 'precursors',
                         'fuel_temperature',
                         'coolant_temperature']

        for v, var in enumerate(variables):
            tmp = self.get_variable_by_key(var)
            matrix = tmp if v == 0 else np.hstack((matrix, tmp))
        return matrix
