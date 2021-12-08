import os
import numpy as np

from typing import List

from pyPDEs.utilities import Vector

from readers.base.dataset_reader import DatasetReader
from readers.neutronics.simulation_reader import NeutronicsSimulationReader


class NeutronicsDatasetReader(DatasetReader):
    """
    A class for reading and handling multiple neutronics simulations.
    """

    def __init__(self, path: str) -> None:
        super().__init__(path)
        self.simulations: List[NeutronicsSimulationReader] = []

    def read_simulation(self, path: str) -> NeutronicsSimulationReader:
        return NeutronicsSimulationReader(path)

    @property
    def dim(self) -> int:
        return self.simulations[0].dim

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
