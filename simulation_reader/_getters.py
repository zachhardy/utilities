import numpy as np
from numpy import ndarray

from typing import TYPE_CHECKING, Tuple, List, Union
if TYPE_CHECKING:
    from . import SimulationReader


Grid = Union[ndarray, Tuple[ndarray, ndarray]]


def get_flux_moment(self: "SimulationReader",
                    moment: int, times: List[float]) -> ndarray:
    """Get flux moment `m` at time `t`.

    Parameters
    ----------
    moment : int
        The requested flux moment index.
    times : List[float]
        The times to get the flux moment at.

    Returns
    -------
    ndarray (n_nodes * n_groups)
    """
    assert moment < self.n_moments
    assert group < self.n_groups

    npc = self.nodes_per_cell
    N, G = self.n_nodes, self.n_groups
    times = times if isinstance(times, list) else [times]

    vals = np.zeros(len(times), N*G)
    tmp = self._interpolate(time, self.flux_moments)
    for c in range(self.n_cells):
        for n in range(npc):
            start = c * npc * G + n * G
            dof = self.map_phi_dof(c, n, moment, 0)
            for t in range(len(times)):
                vals[t, start:start+G] = tmp[t, dof:dof+G]
    return vals


def get_group_flux_moment(self: "SimulationReader", moment: int,
                          group: int, times: List[float]) -> ndarray:
    """Get group `g` flux moment `m` and time `t`.

    Parameters
    ----------
    moment : int
        The flux moment index.
    group : int
        The group index.
    time : float
        The time to get the flux moment at.

    Returns
    -------
    ndarray (n_nodes,)
    """
    assert moment < self.n_moments
    assert group < self.n_groups

    npc = self.nodes_per_cell
    times = times if isinstance(times, list) else [times]

    vals = np.zeros((len(times), self.n_nodes))
    tmp = self._interpolate(times, self.flux_moments)
    for c in range(self.n_cells):
        for n in range(npc):
            i = c*npc + n
            dof = self.map_phi_dof(c, n, moment, group)
            for t in range(len(times)):
                vals[t, i] = tmp[t, dof]
    return vals


def get_precursor_species(self: "SimulationReader",
                          species: Tuple[int, int],
                          times: List[float]) -> ndarray:
    """Get the delayed neutron precursor `j` on `material_id`.

    Parameters
    ----------
    specied : Tuple[int, int]
        The material ID, local precursor ID pair.
    time : float
        The time to get the precursor at.
    """
    assert species[0] < self.n_materials
    assert species[1] < self.max_precursors

    times = times if isinstance(times, list) else [times]

    vals = np.zeros((len(times), self.n_cells))
    tmp = self._interpolate(times, self.precursors)
    for c in range(self.n_cells):
        if self.material_ids[c] == species[0]:
            dof = self.map_precursor_dof(c, species[1])
            for t in range(len(times)):
                vals[t, c] = tmp[t, dof]
    return vals


def get_nodes(self: "SimulationReader") -> Grid:
    if self.dim == 1:
        return [p.z for p in self.nodes]
    elif self.dim == 2:
        x = [p.x for p in self.nodes]
        y = [p.y for p in self.nodes]
        return np.meshgrid(np.unique(x), np.unique(y))


def get_cell_centers(self: "SimulationReader") -> Grid:
    if self.dim == 1:
        return [p.z for p in self.centroids]
    elif self.dim == 2:
        x = [p.x for p in self.centroids]
        y = [p.y for p in self.centroids]
        return np.meshgrid(np.unique(x), np.unique(y))
