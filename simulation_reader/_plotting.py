import numpy as np
import matplotlib.pyplot as plt

from numpy import ndarray
from matplotlib.pyplot import Figure, Axes

from typing import List, Tuple, TYPE_CHECKING
if TYPE_CHECKING:
    from . import SimulationReader


def plot_power_densities(self: "SimulationReader",
                         times: List[float] = None,
                         singular_normalization: bool = True) -> None:
    """Plot power densities at the various times.

    Parameters
    ----------
    times : List[float
        The times to plot the flux moment at.
    """
    times = self._validate_times(times)

    # Get power densities
    P = self._interpolate(times, self.power_densities)

    # Plot 1D profiles
    if self.dim == 1:
        # Initialize figure
        fig: Figure = plt.figure()
        ax: Axes = fig.add_subplot(1, 1, 1)
        ax.set_title("Power Densities")
        ax.set_xlabel("z [cm]")
        ax.set_ylabel(r"P(z) [$\frac{W}{cm^{3}}$]")

        # Generate grid
        z = np.array([p.z for p in self.centroids])

        # Plot at specified times
        for t, time in enumerate(times):
            ax.plot(z, P[t], label=f"Time = {time:.3f} sec")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()

    # Plot 2D profiles
    elif self.dim == 2:
        # Subplot dimentsions
        n_rows, n_cols = self._format_subplots(len(times))

        # Initialize figure
        figsize = (4*n_cols, 4*n_rows)
        fig: Figure = plt.figure(figsize=figsize)
        fig.suptitle(r"P(x, y) [$\frac{W}{cm^{3}}$]")

        # Generate grid
        x = np.array([p.x for p in self.centroids])
        y = np.array([p.y for p in self.centroids])
        X, Y = np.meshgrid(np.unique(x), np.unique(y))

        # Plot at specified times
        for t, time in enumerate(times):
            P_fmtd = P[t].reshape(X.shape)

            ax: Axes = fig.add_subplot(n_rows, n_cols, t + 1)
            ax.set_xlabel("X [cm]")
            ax.set_ylabel("Y [cm]")
            ax.set_title(f"Time = {time:.3f} sec")
            im = ax.pcolor(X, Y, P_fmtd, cmap="jet", shading="auto",
                           vmin=0.0, vmax=P_fmtd.max())
            fig.colorbar(im)
        fig.tight_layout()


def plot_temperature_profiles(self: "SimulationReader",
                              times: List[float]) -> None:
    """Plot temperatures at various times.

    Parameters
    ----------
    times : List[float
        The times to plot the temperatures at.
    """
    times = self._validate_times(times)

    # Get power densities
    T = self._interpolate(times, self.temperatures)

    # Plot 1D profiles
    if self.dim == 1:
        # Initialize figure
        fig: Figure = plt.figure()
        ax: Axes = fig.add_subplot(1, 1, 1)
        ax.set_title("Temperatures")
        ax.set_xlabel("z [cm]")
        ax.set_ylabel(r"T(z) [K]")

        # Generate grid
        z = np.array([p.z for p in self.centroids])

        # Plot at specified times
        for t, time in enumerate(times):
            ax.plot(z, T[t], label=f"Time = {time:.3f} sec")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()

    # Plot 2D profiles
    elif self.dim == 2:
        # Subplot dimentsions
        n_rows, n_cols = self._format_subplots(len(times))

        # Initialize figure
        figsize = (4*n_cols, 4*n_rows)
        fig: Figure = plt.figure(figsize=figsize)
        fig.suptitle("T(x, y) [K]")

        # Generate grid
        x = np.array([p.x for p in self.nodes])
        y = np.array([p.y for p in self.nodes])
        X, Y = np.meshgrid(np.unique(x), np.unique(y))

        # Plot at specified times
        for t, time in enumerate(times):
            T_fmtd = T[t].reshape(X.shape)

            ax: Axes = fig.add_subplot(n_rows, n_cols, t + 1)
            ax.set_xlabel("X [cm]")
            ax.set_ylabel("Y [cm]")
            ax.set_title(f"Time = {t:.3f} sec")
            im = ax.pcolor(X, Y, T_fmtd, cmap="jet", shading="auto",
                           vmin=0.0, vmax=T_fmtd.max())
            fig.colorbar(im)
        fig.tight_layout()


def plot_power(self: "SimulationReader", mode: int = 0,
               log_scale: bool = False) -> None:
    """Plot the power as a function of time.

    In special cases, this routine plots reference lines to
    show agreement with expected results.

    Parameters
    ----------
    mode : int, default 0
        Flag for plotting system power or average power density.
        If 0, system power is plotted.
        If 1, the average power density is plotted.
        If 2, the peak power density is plotted.
    log_scale : bool, default False
        Flag for plotting linear or log scale on the y-axis.
    """
    fig: Figure = plt.figure()
    axs: List[Axes] = []
    if "shutdown" in self.path.lower():
        times = np.array(self.times)
        decay0 = self.powers[1] * np.exp(-0.1*times)
        decay1 = self.powers[1] * np.exp(-0.5*times)
        decay2 = self.powers[1] * np.exp(-1.0*times)

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
        p = self.powers
        if mode == 1:
            p = self.average_power_densities
        elif mode == 2:
            p = self.peak_power_densities

        ax: Axes = fig.add_subplot(1, 1, 1)
        plotter = ax.plot if not log_scale else ax.semilogy
        plotter(self.times, p, "-*b", label="Power")
        axs += [ax]

    for ax in axs:
        ax.set_xlabel("Time [sec]")
        ax.set_ylabel("Power [arb. units]")
        ax.legend()
        ax.grid(True)
    fig.tight_layout()


def plot_temperatures(self: "SimulationReader",
                      mode: int = 0,
                      log_scale: bool = False) -> None:
    """Plot the temperature as a function of time.

    Parameters
    ----------
    mode : int, default 0
        Flag for plotting average or peak temperatures.
        If 0, average temperatures are plotted.
        If 1, peak temperatures are plotted.
    log_scale : bool, default False
        log_scale : bool, default False
        Flag for plotting linear or log scale on the y-axis.
    """
    T = self.average_temperatures
    if mode == 1:
        T = self.peak_temperatures

    fig: Figure = plt.figure()
    ax: Axes = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Time [sec]")
    ax.set_ylabel("Temperature [K]")
    plotter = ax.semilogy if log_scale else ax.plot
    plotter(self.times, T, "-*b")
    ax.grid(True)
    fig.tight_layout()


@staticmethod
def _format_subplots(n_plots: int) -> Tuple[int, int]:
    """Determine the number of rows and columns for subplots.

    Parameters
    ----------
    n_plots : int
        The number of subplots that will be used.

    """
    if n_plots < 4:
        n_rows, n_cols = 1, n_plots
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
