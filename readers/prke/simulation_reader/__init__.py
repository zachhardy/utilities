import os

import numpy as np
from numpy import ndarray
from typing import List

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from readers.base.simulation_reader import SimulationReader


class PRKEReader(SimulationReader):
    def __init__(self, path: str) -> None:
        super().__init__(path)

        self.n_snapshots: int = 0
        self.n_precursors: int = 0

        self.times: ndarray = []

        self.powers: ndarray = []
        self.fuel_temperatures: ndarray = []
        self.coolant_temperatures: ndarray = []
        self.precursors: ndarray = []

    def initialize_storage(self) -> None:
        """
        Initialize storage based on macro-quantities.
        """
        T, J = self.n_snapshots, self.n_precursors

        self.times = np.empty(T, dtype=float)
        self.powers = np.empty(T, dtype=float)
        self.fuel_temperatures = np.empty(T, dtype=float)
        self.coolant_temperatures = np.empty(T, dtype=float)
        self.precursors = np.empty((T, J), dtype=float)

    def read_simulation_data(self) -> None:
        """
        Parse the bindary files of a simulation.
        """
        self.clear()

        # Get sorted file list
        files = sorted(os.listdir(self.path))
        self.n_snapshots = len(files)

        # Loop over files
        for snapshot_num, snapshot in enumerate(files):
            path = os.path.join(self.path, snapshot)

            # Open and read snapshot file
            with open(path, mode='rb') as f:
                f.read(499) # skip header

                step = self.read_uint64_t(f)
                n_precursors = self.read_uint64_t(f)

                # Set macro-data
                if snapshot_num == 0:
                    self.n_precursors = n_precursors
                    self.initialize_storage()

                # Check for compatibility
                else:
                    assert n_precursors == self.n_precursors

                # Set snapshot data
                self.times[step] = self.read_double(f)
                self.powers[step] = self.read_double(f)
                self.fuel_temperatures[step] = self.read_double(f)
                self.coolant_temperatures[step] = self.read_double(f)
                for j in range(n_precursors):
                    precursor = self.read_unsigned_int(f)
                    self.precursors[step, precursor] = self.read_double(f)

    def plot_results(self, log_scale: bool = True) -> None:
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

    def plot_powers(self, log_scale: bool = True) -> None:
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

    def plot_fuel_temperatures(self) -> None:
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

    def plot_coolant_temperatures(self) -> None:
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

    def plot_precursors(self, precursor_nums: List[int] = None) -> None:
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
