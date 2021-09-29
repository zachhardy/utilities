import os
import sys
import struct
import numpy as np

from pyPDEs.utilities import Vector

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import SimulationReader


def read_simulation_data(self: "SimulationReader") -> None:
    """Parse the binary files of a simulation.
    """
    self.clear()

    # Get sorted file list
    files = sorted(os.listdir(self.path))
    self.n_snapshots = len(files)

    # Loop over files
    for snapshot_num, snapshot in enumerate(files):
        path = os.path.join(self.path, snapshot)

        # Open snapshot file
        with open(path, mode="rb") as f:
            f.read(1649)  # skip header

            step = self.read_uint64_t(f)
            time = self.read_double(f)
            power = self.read_double(f)
            peak_power = self.read_double(f)
            avg_power = self.read_double(f)
            peak_temp = self.read_double(f)
            avg_temp = self.read_double(f)

            n_cells = self.read_uint64_t(f)
            n_nodes = self.read_uint64_t(f)
            n_moments = self.read_uint64_t(f)
            n_groups = self.read_uint64_t(f)
            n_precursors = self.read_uint64_t(f)
            max_precursors = self.read_uint64_t(f)

            # Set macro-data if first snapshot
            if snapshot_num == 0:
                self.n_cells = n_cells
                self.n_nodes = n_nodes
                self.n_moments = n_moments
                self.n_groups = n_groups
                self.n_precursors = n_precursors
                self.max_precursors = max_precursors

                self.initialize_storage()

            # Else check for compatibility
            else:
                assert n_cells == self.n_cells
                assert n_nodes == self.n_nodes
                assert n_moments == self.n_moments
                assert n_groups == self.n_groups
                assert n_precursors == self.n_precursors
                assert max_precursors == self.max_precursors

            # Set time and power
            self.times[step] = time
            self.powers[step] = power
            self.peak_powers[step] = peak_power
            self.average_powers[step] = avg_power
            self.peak_temperatures[step] = peak_temp
            self.average_temperatures[step] = avg_temp

            n_blocks = self.read_unsigned_int(f)

            # Get the cell-wise data
            for c in range(n_cells):
                cell_id = self.read_uint64_t(f)
                material_id = self.read_uint64_t(f)
                nodes_per_cell = self.read_uint64_t(f)

                # Save material IDs on first snapshot
                if snapshot_num == 0:
                    self.material_ids.append(material_id)
                    if c == 0:
                        self.nodes_per_cell = nodes_per_cell
                else:
                    assert nodes_per_cell == self.nodes_per_cell

                # Get the centroid coordinates
                p = []
                for d in range(3):
                    p.append(self.read_double(f))

                # Save centroid on first snapshot
                if snapshot_num == 0:
                    self.centroids.append(Vector(p[0], p[1], p[2]))

                # Loop over nodes and get their coordinates
                cell_nodes = []
                for n in range(nodes_per_cell):
                    p = []
                    for d in range(3):
                        p.append(self.read_double(f))

                    # Save nodes on first snapshot
                    if snapshot_num == 0:
                        self.nodes.append(Vector(p[0], p[1], p[2]))

            # Define n_materials by count of unique material IDs
            self.n_materials = len(np.unique(self.material_ids))

            # Parse the flux moments
            self.read_unsigned_int(f)  # skip record type
            n_records = self.read_uint64_t(f)
            for dof in range(n_records):
                cell_id = self.read_uint64_t(f)
                node = self.read_unsigned_int(f)
                moment = self.read_unsigned_int(f)
                group = self.read_unsigned_int(f)

                dof_map = self.map_phi_dof(cell_id, node, moment, group)
                self.flux_moments[step, dof_map] = self.read_double(f)

            # Parse precursors
            self.read_unsigned_int(f)  # skip record type
            n_records = self.read_uint64_t(f)
            for dof in range(n_records):
                cell_id = self.read_uint64_t(f)
                self.read_uint64_t(f)
                precursor = self.read_unsigned_int(f)

                dof_map = self.map_precursor_dof(cell_id, precursor)
                self.precursors[step, dof_map] = self.read_double(f)

            # Parse temperatures
            self.read_unsigned_int(f)  # skip record type
            n_records = self.read_uint64_t(f)
            for dof in range(n_records):
                cell_id = self.read_uint64_t(f)
                self.temperatures[step, cell_id] = self.read_double(f)

            # Parse power density
            self.read_unsigned_int(f)  # skip record type
            n_records = self.read_uint64_t(f)
            for dof in range(n_records):
                cell_id = self.read_uint64_t(f)
                self.power_densities[step, cell_id] = self.read_double(f)

    self._determine_dimension()


@staticmethod
def read_double(file) -> float:
    return struct.unpack("d", file.read(8))[0]


@staticmethod
def read_uint64_t(file) -> int:
    return struct.unpack("Q", file.read(8))[0]


@staticmethod
def read_unsigned_int(file) -> int:
    return struct.unpack("I", file.read(4))[0]
