import os
import struct

import numpy as np

from numpy import ndarray
from typing import List

from pyPDEs.utilities import Vector


class SimulationReader:
    """A class for reading and handling simulation data
    from the Chi-Tech module LBTransientSolver.
    """

    from ._read import read_simulation_data

    from ._getters import (get_phi_m, get_phi_mg,
                           get_precursor_j)

    from ._plotting import (plot_power,
                            plot_flux_moment,
                            plot_power_density,
                            plot_temperature,
                            plot_precursors)

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

    def map_phi_dof(self, cell_id: int, node: int,
                    moment: int, group: int) -> int:
        """Get a flux moment DoF.

        This routine maps a cell, node, moment, and group
        to a DoF in the flux moment vector. This assumes
        a nodal ordering.

        Parameters
        ----------
        cell_id : int
            The unique ID for the cell.
        node : int
            The local node ID for the cell.
        moment : int
            The flux moment desired.
        group : int
            The energy group desired.

        Returns
        -------
        int
        """
        N = self.nodes_per_cell
        M, G = self.n_moments, self.n_groups
        return cell_id*N*M*G + node*M*G + moment*G + group

    def map_precursor_dof(self, cell_id: int, precursor: int) -> int:
        """Get a delayed neutron precursor DoF.

        This routine maps a cell and precursor to a DoF in the
        precursor vector. Precursors are defined at cell centers,
        so no nodal information is needed.

        Parameters
        ----------
        cell_id : int
            The unique ID for the cell.
        precursor : int
            The local precursor number. This must be less than
            `max_precursors`.

        Returns
        -------
        int

        Notes
        -----
        Not all delayed neutron precursors live on every cell.
        The number stored is equivalent to the maximum number of
        precursors that exist on a material. For this reason, local
        precursor IDs do not map uniquely to a single species.
        Local precursor IDs and material IDs used together can map
        uniquely to a specific species.
        """
        return cell_id*self.max_precursors + precursor

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

    def _interpolate(self, t: float, data: ndarray) -> ndarray:
        """Interpolate at a specified time.

        Parameters
        ----------
        t : float
            The desired time to obtain data for.
        data : ndarray (n_steps, n_nodes)
            The data to interpolate.

        Returns
        -------
        ndarray
            The interpolated data.
        """
        if not self.times[0] <= t <= self.times[-1]:
            raise ValueError(
                "Provided time is outside of simulation bounds.")

        dt = np.diff(self.times)[0]
        i = [int(np.floor(t/dt)), int(np.ceil(t/dt))]
        w = [i[1] - t/dt, t/dt - i[0]]
        if i[0] == i[1]:
            w = [1.0, 0.0]
        return w[0]*data[i[0]] + w[1]*data[i[1]]

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

    @staticmethod
    def read_double(file) -> float:
        return struct.unpack("d", file.read(8))[0]

    @staticmethod
    def read_uint64_t(file) -> int:
        return struct.unpack("Q", file.read(8))[0]

    @staticmethod
    def read_unsigned_int(file) -> int:
        return struct.unpack("I", file.read(4))[0]
