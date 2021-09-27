import numpy as np
import matplotlib.pyplot as plt

from matplotlib.pyplot import Figure, Axes

from typing import List, Tuple, TYPE_CHECKING
if TYPE_CHECKING:
    from . import SimulationReader


phi_label = r"$\phi_{m,g}(r)$ [$\frac{n}{cm^{2}~s}$]"


def plot_flux_moments(self: "SimulationReader",
                      moment: int, groups: List[int] = None,
                      times: List[float] = None) -> None:
    """Plot a particular energy group's flux moment at a time `t`.

    Parameters
    ----------
    moment : int
        The moment index to plot.
    groups : List[int]
        The group indices to plot.
    times : List[float
        The times to plot the flux moment at.
    """

    # Get the groups to plot
    if groups is None:
        groups = [0]
    if groups == "all":
        groups = [g for g in range(self.n_groups)]
    if isinstance(groups, int):
        groups = [groups]

    # Get the times to plot
    if times is None:
        times = [self.times[0], self.times[-1]]
    if isinstance(times, float):
        times = [times]

    if self.dim == 1:
        self._plot_1d_flux_moments(moment, groups, times)
    elif self.dim == 2:
        self._plot_2d_flux_moments(moment, groups, times)

def _plot_1d_flux_moments(self: "SimulationReader",
                         moment: int, groups: List[int],
                         times: List[float]) -> None:
    """Plot 1D flux moments.

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
        shape = (len(times), self.n_nodes)
        phi = np.zeros(shape, dtype=float)
        for t, time in enumerate(times):
            phi[t] = self.get_group_flux_moment(moment, group, time)
        phi /= np.max(phi)

        # Plot the flux moments
        for t, time in enumerate(times):
            label = f"Time = {time:.3f} sec"
            ax.plot(z, phi[t], label=label)
        ax.legend()
        ax.grid()
        fig.tight_layout()


def _plot_2d_flux_moments(self: "SimulationReader",
                         moment: int, groups: List[int],
                         times: List[float]) -> None:
    """Plot 2D flux moments.

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
    n_rows, n_cols = _format_subplots(len(times))

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
        phi /= np.max(phi)

        # Plot the flux moments
        for t, time in enumerate(times):
            phi_fmtd = phi[t].reshape(X.shape)

            ax: Axes = fig.add_subplot(n_rows, n_cols, t + 1)
            ax.set_xlabel("X [cm]")
            ax.set_ylabel("Y [cm]")
            ax.set_title(f"Time = {t:.3f} sec")
            im = ax.pcolor(X, Y, phi_fmtd, cmap="jet", shading="auto",
                           vmin=0.0, vmax=phi.max())
            fig.colorbar(im)
        fig.tight_layout()


def plot_power_density(self: "SimulationReader", time: float) -> None:
    """Plot the power density at time `t`

    Parameters
    ----------
    time : float
        The time to plot the power density at.
    """
    power_density = self._interpolate(time, self.power_density)
    grid = np.array([[p.x, p.y, p.z] for p in self.centroids])

    plt.figure()
    plt.title(f"Power Density\nTime = {time} sec")
    if np.sum(grid[:, :2]) == 0.0:
        plt.xlabel("z [cm]")
        plt.ylabel(r"P(z) $[ \frac{W}{cm^{3}} ]$")
        plt.plot(grid[:, 2], power_density, "*-b")
    plt.grid(True)
    plt.tight_layout()


def plot_temperature(self: "SimulationReader", time: float) -> None:
    """Plot the temperature at time `t`

    Parameters
    ----------
    time : float
        The time to plot the temperature at.
    """
    temperature = self._interpolate(time, self.temperature)
    grid = np.array([[p.x, p.y, p.z] for p in self.centroids])

    plt.figure()
    plt.title(f"Temperature\nTime = {time} sec")
    if np.sum(grid[:, :2]) == 0.0:
        plt.xlabel("z (cm)")
        plt.ylabel("T(z) [K]")
        plt.plot(grid[:, 2], temperature, "-*b")

    plt.grid(True)
    plt.tight_layout()


def plot_precursors(self: "SimulationReader",
                    material_id: int, species: int,
                    time: float) -> None:
    precursors = self.get_precursor_j(material_id, species, time)
    grid = np.array([[p.x, p.y, p.z] for p in self.centroids])

    plt.figure()
    plt.title(f"Precursor Species {species}, Material {material_id}\n"
              f"Time = {time} sec")
    if np.sum(grid[:, :2]) == 0.0:
        plt.xlabel("z (cm)")
        plt.ylabel(r"$C_{j}(z)$ $[ \frac{n}{cm^{3}} ]$")
        plt.plot(grid[:, 2], precursors, "-*b")
    plt.grid(True)
    plt.tight_layout()


def plot_power(self: "SimulationReader") -> None:
    """Plot the system power as a function of time.

    In special cases, this routine plots reference lines to
    show agreement with expected results.
    """
    fig: Figure = plt.figure()
    axs: List[Axes] = []
    if "shutdown" in self.path.lower():
        decay0 = self.powers[1] * np.exp(-0.1*self.times)
        decay1 = self.powers[1] * np.exp(-0.5*self.times)
        decay2 = self.powers[1] * np.exp(-1.0*self.times)

        ax: Axes = fig.add_subplot(1, 2, 1)
        ax.semilogy(self.times, self.powers, "-*b", label="Power")
        ax.semilogy(self.times, decay0, "r", label="Decay Rate 0")
        ax.semilogy(self.times, decay1, "g", label="Decay Rate 1")
        ax.semilogy(self.times, decay2, "k", label="Decay Rate 2")
        ax.set_ylim(bottom=1.0e-4)
        axs += [ax]

        ax: Axes = fig.add_subplot(1, 2, 2)
        ax.plot(self.times[1:], self.powers[1:], "-*b", label="Power")
        ax.plot(self.times[1:], decay0[1:], "r", label="Decay Rate 0")
        ax.plot(self.times[1:], decay1[1:], "g", label="Decay Rate 1")
        ax.plot(self.times[1:], decay2[1:], "k", label="Decay Rate 2")
        axs += [ax]

    else:
        ax: Axes = fig.add_subplot(1, 1, 1)
        ax.plot(self.times, self.powers, "-*b", label="Power")
        axs += [ax]

    for ax in axs:
        ax.set_xlabel("Time (sec)")
        ax.set_ylabel("Power (arb. units)")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()


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
