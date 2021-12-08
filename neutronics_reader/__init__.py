import os
import struct

import numpy as np

from numpy import ndarray
from typing import List, Tuple

from pyPDEs.utilities import Vector
from simulation_reader import SimulationReader


class NeutronicsReader(SimulationReader):
    """
    A class for reading and handling transient neutronics data.
    """

    from ._reader import read_simulation_data

    from ._mappings import map_phi_dof, map_precursor_dof

    from ._getters import (get_flux_moment,
                           get_group_flux_moment,
                           get_precursor_species,
                           get_power_densities,
                           get_temperatures,
                           get_variable_by_key,
                           _interpolate,
                           _validate_times)

    from ._formatters import (create_simulation_matrix,
                              create_simulation_vector)

    from ._plotting import (plot_power,
                            plot_temperatures,
                            plot_power_densities,
                            plot_temperature_profiles,
                            _format_subplots)

    from ._plot_flux_moments import (plot_flux_moments,
                                     _plot_1d_flux_moments,
                                     _plot_2d_flux_moments)

    from ._plot_precursors import plot_precursors


    def __init__(self, path: str) -> None:
        super().__init__(path)

        self.dim: int = 0
        self.n_snapshots: int = 0
        self.n_cells: int = 0
        self.n_nodes: int = 0
        self.nodes_per_cell: int = 0
        self.n_moments: int = 0
        self.n_groups: int = 0
        self.n_precursors: int = 0
        self.max_precursors: int = 0
        self.n_materials: int = 0

        self.times: ndarray = []
        self.nodes: List[Vector] = []
        self.centroids: List[Vector] = []
        self.material_ids: List[int] = []

        self.powers: ndarray = []
        self.peak_power_densities: ndarray = []
        self.average_power_densities: ndarray = []
        self.peak_temperatures: ndarray = []
        self.average_temperatures: ndarray = []

        self.flux_moments: ndarray = []
        self.precursors: ndarray = []
        self.temperatures: ndarray = []
        self.power_densities: ndarray = []

    def initialize_storage(self) -> None:
        """
        Size all data vectors based on the macro-quantities.
        """
        T, N, C = self.n_snapshots, self.n_nodes, self.n_cells
        M, G, P = self.n_moments, self.n_groups, self.max_precursors

        self.times = np.empty(T, dtype=float)

        self.powers = np.empty(T, dtype=float)
        self.peak_power_densities = np.empty(T, dtype=float)
        self.average_power_densities = np.empty(T, dtype=float)
        self.peak_temperatures = np.empty(T, dtype=float)
        self.average_temperatures = np.empty(T, dtype=float)

        self.flux_moments = np.empty((T, N * M * G), dtype=float)
        self.precursors = np.empty((T, C * P), dtype=float)
        self.temperatures = np.empty((T, C), dtype=float)
        self.power_densities = np.empty((T, C), dtype=float)

    def _determine_dimension(self) -> None:
        """
        Determine the spatial dimension.
        """
        grid = np.array([[p.x, p.y, p.z] for p in self.nodes])
        if np.sum(grid[:, :2]) == 0.0:
            self.dim = 1
        elif np.sum(grid[:, 2]) == 0.0:
            self.dim = 2
        else:
            self.dim = 3
