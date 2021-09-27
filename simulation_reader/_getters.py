import numpy as np
from numpy import ndarray

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import SimulationReader


def get_phi_m(self: "SimulationReader",
              moment: int, time: float) -> ndarray:
    """Get flux moment `m` at time `t`.

    Parameters
    ----------
    moment : int
        The requested flux moment index.
    time : float
        The time to get the flux moment at.

    Returns
    -------
    ndarray (n_nodes * n_groups)
    """
    npc = self.nodes_per_cell
    N, G = self.n_nodes, self.n_groups

    vals = np.zeros(N * G)
    tmp = self._interpolate(time, self.flux_moments)
    for c in range(self.n_cells):
        for n in range(npc):
            start = c * npc * G + n * G
            dof = self.map_phi_dof(c, n, moment, 0)
            vals[start:start+G] = tmp[dof:dof+G]
    return vals


def get_phi_mg(self: "SimulationReader", moment: int,
               group: int, time: float) -> ndarray:
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
    npc = self.nodes_per_cell
    vals = np.zeros(self.n_nodes)
    tmp = self._interpolate(time, self.flux_moments)
    for c in range(self.n_cells):
        for n in range(npc):
            i = c*npc + n
            dof = self.map_phi_dof(c, n, moment, group)
            vals[i] = tmp[dof]
    return vals


def get_precursor_j(self: "SimulationReader", material_id: int,
                    precursor_num: int, time: float) -> ndarray:
    """Get the delayed neutron precursor `j` on `material_id`.

    Parameters
    ----------
    material_id : int
        The material ID.
    precursor_num : int
        The precursor index.
    time : float
        The time to get the precursor at.
    """
    vals = np.zeros(self.n_cells)
    tmp = self._interpolate(time, self.precursors)
    for c in range(self.n_cells):
        if self.material_ids[c] == material_id:
            dof = self.map_precursor_dof(c, precursor_num)
            vals[c] = tmp[dof]
    return vals
