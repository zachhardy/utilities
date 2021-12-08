from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import NeutronicsSimulationReader


def map_phi_dof(self: 'NeutronicsSimulationReader',
                cell_id: int, node: int,
                moment: int, group: int) -> int:
    """
    Get a flux moment DoF.

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
    return cell_id * N * M *G + node * M *G + moment *G + group

def map_precursor_dof(self: 'NeutronicsSimulationReader',
                      cell_id: int, precursor: int) -> int:
    """
    Get a delayed neutron precursor DoF.

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
    return cell_id *self.max_precursors + precursor
