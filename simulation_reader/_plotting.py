import numpy as np
import matplotlib.pyplot as plt

from numpy import ndarray
from matplotlib.pyplot import Figure, Axes

from typing import List, Tuple, TYPE_CHECKING
if TYPE_CHECKING:
    from . import SimulationReader


def plot_power_densities(self: "SimulationReader",
                         times: List[float] = None) -> None:
    """Plot power densities at the various times.

    Parameters
    ----------
    times : List[float
        The times to plot the flux moment at.
    """
    times = self._validate_times(times)

    # Get power densities
    P = self._interpolate(times, self.power_density)

    # Define labels for plots
    labels = [f"Time {t:.3f} sec" for t in times]

    # Plot 1D profiles
    if self.dim == 1:
        title = "Power Density"
        ylabel = r"$P(z)$ [$\frac{W}{cm^{3}}$]"
        self._plot_1d_cell_centered(P, labels, title, ylabel)

    # Plot 2D profiles
    elif self.dim == 2:
        x = np.array([p.x for p in self.nodes])
        y = np.array([p.y for p in self.nodes])
        X, Y = np.meshgrid(np.unique(x), np.unique(y))

        # Subplot dimentsions
        n_rows, n_cols = self._format_subplots(len(times))

        # Initialize figure
        figsize = (4*n_cols, 4*n_rows)
        fig: Figure = plt.figure(figsize=figsize)
        fig.suptitle(r"P(x, y) [$\frac{W}{cm^{3}}$]")

        for t, time in enumerate(times):
            P_fmtd = P[t].reshape(X.shape)

            ax: Axes = fig.add_subplot(n_rows, n_cols, t + 1)
            ax.set_xlabel("X [cm]")
            ax.set_ylabel("Y [cm]")
            ax.set_title(f"Time = {t:.3f} sec")
            im = ax.pcolor(X, Y, P_fmtd, cmap="jet", shading="auto",
                           vmin=0.0, vmax=P.max())
            fig.colorbar(im)
        fig.tight_layout()


def plot_temperatures(self: "SimulationReader",
                      times: List[float]) -> None:
    """Plot temperatures at various times.

    Parameters
    ----------
    times : List[float
        The times to plot the temperatures at.
    """
    times = self._validate_times(times)

    # Get power densities
    T = self._interpolate(times, self.temperature)

    # Define labels for plots
    labels = [f"Time {t:.3f} sec" for t in times]

    # Plot 1D profiles
    if self.dim == 1:
        title = "Temperature"
        ylabel = "T(z) [K]"
        self._plot_1d(T, labels, title, ylabel)

    # Plot 2D profiles
    elif self.dim == 2:
        x = np.array([p.x for p in self.nodes])
        y = np.array([p.y for p in self.nodes])
        X, Y = np.meshgrid(np.unique(x), np.unique(y))

        # Subplot dimentsions
        n_rows, n_cols = self._format_subplots(len(times))

        # Initialize figure
        figsize = (4 * n_cols, 4 * n_rows)
        fig: Figure = plt.figure(figsize=figsize)
        fig.suptitle("T(x, y) [K]")

        for t, time in enumerate(times):
            T_fmtd = T[t].reshape(X.shape)

            ax: Axes = fig.add_subplot(n_rows, n_cols, t + 1)
            ax.set_xlabel("X [cm]")
            ax.set_ylabel("Y [cm]")
            ax.set_title(f"Time = {t:.3f} sec")
            im = ax.pcolor(X, Y, T_fmtd, cmap="jet", shading="auto",
                           vmin=0.0, vmax=T.max())
            fig.colorbar(im)
        fig.tight_layout()


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


def _plot_cell_centered_data(self: "SimulationReader",
                             data: ndarray, labels: List[str] = None,
                             title: str = None, ylabel: str = None) -> None:


    plt.figure()
    plt.title("" if title is None else title)
    plt.xlabel("z [cm]")
    plt.ylabel("" if ylabel is None else ylabel)

    for t, time in enumerate(times):
        plt.plot(z, data[t], label=labels[t])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()


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
