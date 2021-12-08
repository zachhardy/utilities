import numpy as np
import matplotlib.pyplot as plt

from matplotlib.pyplot import Figure, Axes

from typing import List, Tuple, TYPE_CHECKING
if TYPE_CHECKING:
    from . import NeutronicsReader


def plot_precursors(self: 'NeutronicsReader',
                    species: List[Tuple[int, int]],
                    times: List[float] = None) -> None:
    """
    Plot precursor species at various times.

    Parameters
    ----------
    species : List[Tuple[int, int]]
        The precursor species to plot. The first entry
        in the descriptor tuple is the material ID and the
        second is the local precursor ID.
    times : List[float]
        The times to plot the precursors at.
    """
    # Get the species to plot
    if species is None:
        species = [(0, 0)]
    elif isinstance(species, tuple):
        species = [species]

    # Get the times to plot
    times = self._validate_times(times)

    if self.dim == 1:
        z = np.array([p.z for p in self.centroids])

        # Loop over species
        for species_ in species:
            fig: Figure = plt.figure()
            ax: Axes = fig.add_subplot(1, 1, 1)
            ax.set_title(f'Material {species_[0]} '
                         f'Precursor Species {species_[1]}')
            ax.set_xlabel('z [cm]')
            ax.set_ylabel(r'$C_{i,j}(z)$ $[$\frac{#}{cm^3}$]')

            # Get the precursors at specified times
            C = self.get_precursor_species(species_, times)
            C /= np.max(C)

            # Plot the precursors
            for t, time in enumerate(times):
                label = f'Time = {time:.3f} sec'
                ax.plot(z, C[t], label=label)
            ax.legend()
            ax.grid(True)
            fig.tight_layout()

    elif self.dim == 2:
        x = np.array([p.x for p in self.nodes])
        y = np.array([p.y for p in self.nodes])
        X, Y = np.meshgrid(np.unique(x), np.unique(y))

        # Subplot dimentsions
        n_rows, n_cols =self._format_subplots(len(times))

        # Loop over groups
        for species_ in species:
            figsize = (4 * n_cols, 4 * n_rows)
            fig: Figure = plt.figure(figsize=figsize)
            fig.suptitle(f'Material {species_[0]} '
                         f'Precursor Species {species_[1]}')

            # Get the flux moments at specified times
            shape = (len(times), self.n_cells)
            C = np.zeros(shape, dtype=float)
            for t, time in enumerate(times):
                C[t] = self.get_precursor_species(species_, time)
            C /= np.max(C)

            # Plot the flux moments
            for t, time in enumerate(times):
                C_fmtd = C[t].reshape(X.shape)

                ax: Axes = fig.add_subplot(n_rows, n_cols, t + 1)
                ax.set_xlabel('X [cm]')
                ax.set_ylabel('Y [cm]')
                ax.set_title(f'Time = {t:.3f} sec')
                im = ax.pcolor(X, Y, C_fmtd, cmap='jet', shading='auto',
                               vmin=0.0, vmax=C.max())
                fig.colorbar(im)
            fig.tight_layout()
