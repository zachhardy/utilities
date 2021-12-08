import numpy as np
from numpy import ndarray

from typing import TYPE_CHECKING, Tuple, List, Union
if TYPE_CHECKING:
    from . import NeutronicsSimulationReader


def get_flux_moment(self: 'NeutronicsSimulationReader',
                    moment: int, times: List[float]) -> ndarray:
    """
    Get flux moment `m` at time `t`.

    Parameters
    ----------
    moment : int
        The requested flux moment index.
    times : List[float]
        The times to get the flux moment at.

    Returns
    -------
    ndarray (n_times, n_nodes * n_groups)
    """
    assert moment < self.n_moments
    npc = self.nodes_per_cell
    N, G = self.n_nodes, self.n_groups
    times = self._validate_times(times)

    vals = np.zeros((len(times), N * G))
    tmp = self._interpolate(times, self.flux_moments)
    for c in range(self.n_cells):
        for n in range(npc):
            start = c * npc * G + n * G
            dof = self.map_phi_dof(c, n, moment, 0)
            for t in range(len(times)):
                vals[t, start:start+G] = tmp[t, dof:dof+G]
    return vals


def get_group_flux_moment(self: 'NeutronicsSimulationReader', moment: int,
                          group: int, times: List[float]) -> ndarray:
    """
    Get group `g` flux moment `m` and time `t`.

    Parameters
    ----------
    moment : int
        The flux moment index.
    groups : List[int]
        The group indices to plot.
    times : List[float]
        The times to get the group flux moment at.

    Returns
    -------
    ndarray (n_times, n_nodes)
    """
    assert moment < self.n_moments
    assert group < self.n_groups
    npc = self.nodes_per_cell
    times = self._validate_times(times)

    vals = np.zeros((len(times), self.n_nodes))
    tmp = self._interpolate(times, self.flux_moments)
    for c in range(self.n_cells):
        for n in range(npc):
            i = c*npc + n
            dof = self.map_phi_dof(c, n, moment, group)
            for t in range(len(times)):
                vals[t, i] = tmp[t, dof]
    return vals


def get_precursor_species(self: 'NeutronicsSimulationReader',
                          species: Tuple[int, int],
                          times: List[float]) -> ndarray:
    """
    Get the delayed neutron precursor `j` on `material_id`.

    Parameters
    ----------
    specied : Tuple[int, int]
        The material ID, local precursor ID pair.
    times : List[float]
        The times to get the precursor species at.

    Returns
    -------
    ndarray (n_times, n_cells)
    """
    assert species[0] < self.n_materials
    assert species[1] < self.max_precursors
    times = self._validate_times(times)

    vals = np.zeros((len(times), self.n_cells))
    tmp = self._interpolate(times, self.precursors)
    for c in range(self.n_cells):
        if self.material_ids[c] == species[0]:
            dof = self.map_precursor_dof(c, species[1])
            for t in range(len(times)):
                vals[t, c] = tmp[t, dof]
    return vals


def get_power_densities(self: 'NeutronicsSimulationReader',
                        times: List[float]) -> ndarray:
    """
    Get the power densities at the providied times.

    Parameters
    ----------
    times : List[float]
        The times to get the power densities at.

    Returns
    -------
    ndarray (n_times, n_cells)
    """
    return self._interpolate(times, self.power_densities)


def get_temperatures(self: 'NeutronicsSimulationReader',
                     times: List[float]) -> ndarray:
    """
    Get the temperatures at the providied times.

    Parameters
    ----------
    times : List[float]
        The times to get the temperatures at.

    Returns
    -------
    ndarray (n_times, n_cells)
    """
    return self._interpolate(times, self.temperatures)


def _interpolate(self: 'NeutronicsSimulationReader',
                 times: List[float], data: ndarray) -> ndarray:
    """
    Interpolate at a specified time.

    Parameters
    ----------
    times : List[float]
        The desired times to obtain data for.
    data : ndarray (n_steps, n_nodes)
        The data to interpolate.

    Returns
    -------
    ndarray
        The interpolated data.
    """
    times = self._validate_times(times)
    vals = np.zeros((len(times), data.shape[1]))
    for t, time in enumerate(times):
        dt = np.diff(self.times)[0]
        i = [int(np.floor(time/dt)), int(np.ceil(time/dt))]
        w = [i[1] - time/dt, time/dt - i[0]]
        if i[0] == i[1]:
            w = [1.0, 0.0]
        vals[t] = w[0]*data[i[0]] + w[1]*data[i[1]]
    return vals


def get_variable_by_key(
        self: 'NeutronicsSimulationReader', key: str) -> ndarray:
    """
    Get a variable by its name.

    Parameters
    ----------
    key : str
        The variable name.

    Returns
    -------
    ndarray (shape varies)
    """
    if 'flux' in key:
        if 'm' not in key and 'g' not in key:
            return self.flux_moments
        elif 'm' in key and 'g' not in key:
            m = int(key[key.find('m') + 1])
            return self.get_flux_moment(m, self.times)
        elif 'm' in key and 'g' in key:
            m = int(key[key.find('m') + 1])
            g = int(key[key.find('g') + 1])
            return self.get_group_flux_moment(m, g, self.times)
    elif key == 'temperature':
        return self.get_temperatures(self.times)
    elif key == 'power_density':
        return self.get_power_densities(self.times)


def _validate_times(self: 'NeutronicsSimulationReader',
                    times: List[float]) -> List[float]:
    """
    Ensure the plotting times are valid.

    Parameters
    ----------
    times : List[float]
        The times to validate
    """
    if times is None:
        times = [self.times[0], self.times[-1]]
    if isinstance(times, float):
        times = [times]
    if isinstance(times, ndarray):
        times = list(times)
    for time in times:
        if not self.times[0] <= time <= self.times[-1]:
            raise ValueError(
                'A specified time falls outside of simulation bounds.')
    return times
