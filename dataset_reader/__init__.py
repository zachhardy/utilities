import os
import numpy as np

from numpy import ndarray
from typing import List

from simulation_reader import SimulationReader


class DatasetReader:
    """A class for reading and handling multiple simulations.
    """

    def __init__(self, path: str) -> None:
        if not os.path.isdir(path):
            raise NotADirectoryError(
                "The provided path is not a valid directory.")

        self.path: str = os.path.abspath(path)
        self.simulations : List[SimulationReader] = []

    def read_dataset(self) -> None:
        self.clear()

        # Get sorted file list
        entries = sorted(os.listdir(self.path))

        for simulation_num, simulation in enumerate(entries):
            path = os.path.join(self.path, simulation)
            if os.path.isdir(path):
                self.simulations.append(SimulationReader(path))
                self.simulations[-1].read_simulation_data()
            else:
                pass

    def clear(self) -> None:
        self.__init__(self.path)
