import numpy as np
import matplotlib.pyplot as plt

from matplotlib.pyplot import Figure, Axes

from typing import List, TYPE_CHECKING
if TYPE_CHECKING:
    from . import SimulationReader


def plot_flux_moment(self: "SimulationReader",
                     moment: int, group: int,
                     time: float) -> None:
    """Plot a particular energy group's flux moment at a time `t`.

    Parameters
    ----------
    moment : int
        The moment index to plot.
    group : int
        The group index to plot.
    time : float
        The time to plot the flux moment at.
    """
    phi = self.get_phi_mg(moment, group, time)
    grid = np.array([[p.x, p.y, p.z] for p in self.nodes])

    plt.figure()
    plt.title(f"Flux Moment {moment}, Groups {group}\nTime = {time} sec")
    if np.sum(grid[:, :2]) == 0.0:
        plt.xlabel("z [cm]")
        plt.ylabel(r"$\phi_{m,g}(z)$ $[ \frac{n}{cm^{3} s} ]$")
        plt.plot(grid[:, 2], phi / np.linalg.norm(phi), "-*b")

    plt.grid(True)
    plt.tight_layout()


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
