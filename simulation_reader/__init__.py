import os
import struct

import numpy as np

from numpy import ndarray
from typing import List, Tuple

from pyPDEs.utilities import Vector


class SimulationReader:
    """A class for reading and handling simulation data
    from the Chi-Tech module LBTransientSolver.
    """

    from ._read import (read_simulation_data,
                        read_uint64_t,
                        read_unsigned_int,
                        read_double)

    from ._mappings import map_phi_dof, map_precursor_dof

    from ._getters import (get_flux_moment,
                           get_group_flux_moment,
                           get_precursor_species,
                           get_nodes,
                           get_cell_centers,
                           _interpolate)

    from ._plot_flux_moments import (plot_flux_moments,
                                     _plot_1d_flux_moments,
                                     _plot_2d_flux_moments)

    from ._plot_precursors import plot_precursors

    from ._plotting import (plot_power,
                            plot_power_densities,
                            plot_temperatures)

    def __init__(self, path: str) -> None:
        if not os.path.isdir(path):
            raise NotADirectoryError(
                "The provided path is not a valid directory.")

        self.path: str = os.path.abspath(path)

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
        self.powers: ndarray = []

        self.material_ids: List[int] = []
        self.centroids: List[Vector] = []
        self.nodes: List[Vector] = []

        self.flux_moments: ndarray = []
        self.precursors: ndarray = []
        self.temperature: ndarray = []
        self.power_density: ndarray = []

    def initialize_storage(self) -> None:
        """Size all data vectors based on the macro-quantities.
        """
        T, N, C = self.n_snapshots, self.n_nodes, self.n_cells
        M, G, P = self.n_moments, self.n_groups, self.max_precursors
        self.times = np.empty(T, dtype=float)
        self.powers = np.empty(T, dtype=float)
        self.flux_moments = np.empty((T, N * M * G), dtype=float)
        self.precursors = np.empty((T, C * P), dtype=float)
        self.temperature = np.empty((T, C), dtype=float)
        self.power_density = np.empty((T, C), dtype=float)

    def clear(self) -> None:
        self.material_ids.clear()
        self.centroids.clear()
        self.nodes.clear()

        self.times = []
        self.powers = []
        self.flux_moments = []
        self.precursors = []
        self.temperature = []
        self.power_density = []

    def _determine_dimension(self) -> None:
        """Determine the spatial dimension.
        """
        grid = np.array([[p.x, p.y, p.z] for p in self.nodes])
        if np.sum(grid[:, :2]) == 0.0:
            self.dim = 1
        elif np.sum(grid[:, 2]) == 0.0:
            self.dim = 2
        else:
            self.dim = 3

    def _validate_times(self, times: List[float]) -> List[float]:
        if times is None:
            times = [self.times[0], self.times[-1]]
        if isinstance(times, float):
            times = [times]
        return times

    @staticmethod
    def _format_subplots(n_plots: int) -> Tuple[int, int]:
        """Determine the number of rows and columns for subplots.

        Parameters
        ----------
        n_plots : int
            The number of subplots that will be used.

        """
        n_rows, n_cols = 1, 1
        if n_plots < 4:
            n_rows, n_cols = 1, 3
        elif 4 <= n_plots < 9:
            ref = int(np.ceil(np.sqrt((n_plots))))
            n_rows = n_cols = ref
            for n in range(1, n_cols + 1):
                if n * n_cols >= n_plots:
                    n_rows = n
                    break
        else:
            raise AssertionError("Maximum number of plots is 9.")
        return n_rows, n_cols
