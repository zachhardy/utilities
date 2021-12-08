import os
import numpy as np

from typing import List

from pyPDEs.utilities import Vector

from ...base import DatasetReader
from ..simulation_reader import PRKESimulationReader


class PRKEDatasetReader(DatasetReader):
    """
    A class for reading and handling multiple neutronics simulations.
    """

    def __init__(self, path: str) -> None:
        super().__init__(path)
        self.simulations: List[PRKESimulationReader] = []

    def read_simulation(self, path: str) -> PRKESimulationReader:
        return PRKESimulationReader(path)

    @property
    def times(self) -> List[float]:
        return self.simulations[0].times

    @property
    def n_precursors(self) -> int:
        return self.simulations[0].n_precursors

