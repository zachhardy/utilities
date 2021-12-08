import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from typing import List, TYPE_CHECKING
if TYPE_CHECKING:
    from . import PRKESimulationReader


def plot_results(self: 'PRKESimulationReader',
                 log_scale: bool = True) -> None:
    """
    Plot all results.

    Parameters
    ----------
    log_scale : bool
    """
    fig: Figure = plt.figure()
    fig.suptitle('PRKE Results', fontsize=12)

    # Plot power
    ax: Axes = fig.add_subplot(2, 2, 1)
    ax.set_ylabel('Power (W)', fontsize=12)
    plotter = ax.semilogy if log_scale else ax.plot
    plotter(self.times, self.powers, '-b*')
    ax.grid(True)

    # Plot precursors
    ax: Axes = fig.add_subplot(2, 2, 2)
    ax.set_ylabel('$C_{j}$ ($m^{-3})$', fontsize=12)
    for j in range(self.n_precursors):
        cj = self.precursors[:, j]
        ax.plot(self.times, cj, '-*', label=f'Precursor {j}')
    ax.legend()
    ax.grid(True)

    # Plot fuel temperature
    ax: Axes = fig.add_subplot(2, 2, 3)
    ax.set_xlabel('Time (sec)', fontsize=12)
    ax.set_ylabel('$T_{fuel}$ (K)', fontsize=12)
    ax.plot(self.times, self.fuel_temperatures, '-b*')
    ax.grid(True)

    # Plot coolant temperature
    ax: Axes = fig.add_subplot(2, 2, 4)
    ax.set_xlabel('Time (sec)', fontsize=12)
    ax.set_ylabel('$T_{coolant}$ (K)', fontsize=12)
    ax.plot(self.times, self.coolant_temperatures, '-b*')
    ax.grid(True)

    plt.tight_layout()


def plot_powers(self: 'PRKESimulationReader',
                log_scale: bool = True) -> None:
    """
    Plot the system power as a function of time.

    Parameters
    ----------
    log_scale : bool, default False
        Flag for logscale y-axis
    """
    fig: Figure = plt.figure()
    ax: Axes = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Time (sec)', fontsize=12)
    ax.set_ylabel('Power (W)', fontsize=12)
    plotter = ax.semilogy if log_scale else ax.plot
    plotter(self.times, self.powers, '-b*')
    ax.grid(True)
    plt.tight_layout()


def plot_fuel_temperatures(self: 'PRKESimulationReader') -> None:
    """
    Plot the fuel temperature as a function of time.
    """
    fig: Figure = plt.figure()
    ax: Axes = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Time (sec)', fontsize=12)
    ax.set_ylabel('Fuel Temperature (K)', fontsize=12)
    ax.plot(self.times, self.fuel_temperatures, '-b*')
    ax.grid(True)
    plt.tight_layout()


def plot_coolant_temperatures(self: 'PRKESimulationReader') -> None:
    """
    Plot the fuel temperature as a function of time.
    """
    fig: Figure = plt.figure()
    ax: Axes = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Time (sec)', fontsize=12)
    ax.set_ylabel('Coolant Temperature (K)', fontsize=12)
    ax.plot(self.times, self.coolant_temperatures, '-b*')
    ax.grid(True)
    plt.tight_layout()


def plot_precursors(self: 'PRKESimulationReader',
                    precursor_nums: List[int] = None) -> None:
    """
    Plot the precursors as a function of time.

    Parameters
    ----------
    precursor_nums : List[int], default None
        The precursor indices to plot
    """
    if precursor_nums is None:
        precursor_nums = [list(range(self.n_precursors))]
    elif isinstance(precursor_nums, int):
        precursor_nums = [precursor_nums]
    if not isinstance(precursor_nums, list):
        raise ValueError('precursors must be a list of indices.')

    fig: Figure = plt.figure()
    ax: Axes = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Time (sec)', fontsize=12)
    ax.set_ylabel('Precursor Concentration ($m^{-3}$)', fontsize=12)
    for j in precursor_nums:
        cj = self.precursors[:, j]
        ax.plot(self.times, cj, label=f'Precursor {j}')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
