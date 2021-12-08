import numpy as np
import matplotlib.pyplot as plt

from matplotlib.pyplot import Figure, Axes

from typing import List, Tuple, TYPE_CHECKING
if TYPE_CHECKING:
    from . import NeutronicsReader

phi_label = r"$\phi_{m,g}(r)$ [$\frac{n}{cm^{2}~s}$]"


def plot_flux_moments(self: "NeutronicsReader",
                      moment: int, groups: List[int] = None,
                      times: List[float] = None) -> None:
    """Plot groupwise flux moments at various times.

    Parameters
    ----------
    moment : int
        The moment index to plot.
    groups : List[int]
        The group indices to plot.
    times : List[float]
        The times to plot the flux moment at.
    """

    # Get the groups to plot
    if groups is None:
        groups = [0]
    if groups == -1:
        groups = [g for g in range(self.n_groups)]
    if isinstance(groups, int):
        groups = [groups]

    # Get the times to plot
    time = self._validate_times(times)

    if self.dim == 1:
        self._plot_1d_flux_moments(moment, groups, times)
    elif self.dim == 2:
        self._plot_2d_flux_moments(moment, groups, times)

def _plot_1d_flux_moments(self: "NeutronicsReader",
                          moment: int, groups: List[int],
                          times: List[float]) -> None:
    """Plot 1D groupwise flux moments at various times.

    Parameters
    ----------
    moment : int
        The moment index to plot.
    groups : List[int]
        The group indices to plot.
    times : List[float
        The times to plot the flux moment at.
    """
    shape = (len(times), self.n_cells)

    # Get the grid
    z = np.array([p.z for p in self.nodes])

    # Loop over groups
    for group in groups:
        fig: Figure = plt.figure()
        ax: Axes = fig.add_subplot(1, 1, 1)
        ax.set_title(f"Moment {moment} Group {group} "
                     f"Flux Moments")
        ax.set_xlabel("z [cm]")
        ax.set_ylabel(f"{phi_label}")

        # Get the flux moments at specified times
        phi = self.get_group_flux_moment(moment, group, times)
        phi /= np.max(phi)

        # Plot the flux moments
        for t, time in enumerate(times):
            label = f"Time = {time:.3f} sec"
            ax.plot(z, phi[t], label=label)
        ax.legend()
        ax.grid(True)
        fig.tight_layout()


def _plot_2d_flux_moments(self: "NeutronicsReader",
                          moment: int, groups: List[int],
                          times: List[float]) -> None:
    """Plot 2D groupwise flux moments at various times.

    Parameters
    ----------
    moment : int
        The moment index to plot.
    groups : List[int]
        The group indices to plot.
    times : List[float
        The times to plot the flux moment at.
    """
    x = np.array([p.x for p in self.nodes])
    y = np.array([p.y for p in self.nodes])
    X, Y = np.meshgrid(np.unique(x), np.unique(y))

    # Subplot dimentsions
    n_rows, n_cols = self._format_subplots(len(times))

    # Loop over groups
    for group in groups:
        figsize = (4*n_cols, 4*n_rows)
        fig: Figure = plt.figure(figsize=figsize)
        fig.suptitle(f"Moment {moment} Group {group} "
                     f"Flux Moment")

        # Get the flux moments at specified times
        shape = (len(times), self.n_nodes)
        phi = np.zeros(shape, dtype=float)
        for t, time in enumerate(times):
            phi[t] = self.get_group_flux_moment(moment, group, time)

        # Plot the flux moments
        for t, time in enumerate(times):
            phi_fmtd = phi[t].reshape(X.shape)

            ax: Axes = fig.add_subplot(n_rows, n_cols, t + 1)
            ax.set_xlabel("X [cm]")
            ax.set_ylabel("Y [cm]")
            ax.set_title(f"Time = {time:.3f} sec")
            im = ax.pcolor(X, Y, phi_fmtd, cmap="jet", shading="auto",
                           vmin=0.0, vmax=phi_fmtd.max())
            fig.colorbar(im)
        fig.tight_layout()