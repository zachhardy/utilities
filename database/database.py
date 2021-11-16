import os
import numpy as np
from numpy import ndarray
from typing import List, Union, Dict, Tuple


Var = Union[List[str], str]


class Database:
    """Database handler.
    """

    def __init__(self) -> None:
        self._times: ndarray = []
        self._grid: ndarray = []
        self._parameters: ndarray = []

        self._data: Dict[str, ndarray] = {}

        self._n_nodal_variables: int = 0
        self._n_global_variables: int = 0

    @property
    def n_simulations(self) -> int:
        """
        Get the number of simulations.

        Returns
        -------
        int
        """
        return len(self._data[self.variable_names[0]])

    @property
    def n_timesteps(self) -> int:
        """
        Get the number of timesteps per simulation.

        Returns
        -------
        int
        """
        return len(self._times)

    @property
    def n_grid_points(self) -> int:
        """
        Get the number of grid points.

        Returns
        -------
        int
        """
        return len(self._grid)

    @property
    def n_parameters(self) -> int:
        """
        Get the number of parameters.

        Returns
        -------
        int
        """
        return self._parameters.shape[1]

    @property
    def n_variables(self) -> int:
        """
        Get the number of variables.

        Returns
        -------
        int
        """
        return len(self.variable_names)

    @property
    def n_global_variables(self) -> int:
        """
        Get the total number of globally defined variables.

        Returns
        -------
        int
        """
        return self._n_global_variables

    @property
    def n_nodal_variables(self) -> int:
        """
        Get the total number of nodally defined variables.

        Returns
        -------
        int
        """
        return self._n_nodal_variables

    @property
    def n_dofs(self) -> int:
        """
        Get the total number of DoFs.

        This returns the total across all variables.

        Returns
        -------
        int
        """
        n_nodal = self.n_nodal_variables * self.n_grid_points
        return n_nodal + self.n_global_variables

    @property
    def data(self) -> Dict[str, ndarray]:
        """
        Get the data stored within the database.

        Returns
        -------
        Dict[str, ndarray]
            For each variable key, store a list of simulation
            results containing a list of timesteps containing
            a list of the DoFs for that variable.
        """
        return self._data

    @property
    def variable_names(self) -> List[str]:
        """
        Get the variable_names corresponding.

        Returns
        -------
        List[str]
        """
        return list(self._data.keys())

    @property
    def grid(self) -> ndarray:
        """
        Get the grid nodal variables are defined on.

        Returns
        -------
        ndarray
        """
        return self._grid

    @property
    def times(self) -> ndarray:
        """
        Get the snapshot times.

        Returns
        -------
        ndarray
        """
        return self._times

    @property
    def parameters(self) -> List[List[float]]:
        """
        Get the parameters of each simulation.

        Returns
        -------
        List[List[float]]
            Each entry in the outer list holds the list
            of parameters that define the simulation.
        """
        return self._parameters

    def read_database(
            self, datapath: str, verbose: bool = False) -> None:
        """
        Read a database.

        Parameters
        ----------
        datapath : str
            The path to the directory of results.
        verbose : bool, default False
        """
        if not os.path.isdir(datapath):
            raise NotADirectoryError(f"{datapath} is not valid.")

        print(f"\n***** Parsing database at {datapath} *****")

        # ======================================== Get macro data
        if len(self._times) == 0:
            self.parse_times(datapath)
        if len(self._grid) == 0:
            self.parse_grid(datapath)
        if len(self._parameters) == 0:
            self.parse_parameters(datapath)

        # number of digits in simulation directory
        n_digits = len(os.listdir(datapath)[0])

        # number of simulations
        n_simulations = len(os.listdir(datapath))
        if "params.txt" in os.listdir(datapath):
            n_simulations -= 1
        if "reference" in os.listdir(datapath):
            n_simulations -= 1

        # ======================================== Loop over simulations
        for s in range(n_simulations):
            simulation = str(s).zfill(n_digits)
            simulation_path = os.path.join(datapath, simulation)
            self.read_database_entry(simulation_path, verbose)

        # ======================================== Convert to ndarrays
        for name, data in self._data.items():
            self._data[name] = np.array(data)

        # ======================================== Print summary
        size = self.n_simulations * self.n_timesteps * self.n_dofs
        msg = "===== Database Information ====="
        head = "=" * len(msg)
        print("\n".join(["", head, msg, head]))
        print(f"Database Size:\t{size}")
        print(f"Simulations:\t{self.n_simulations}")
        print(f"Time Steps:\t{self.n_timesteps}")
        print(f"Grid Points:\t{self.n_grid_points}")
        print(f"Nodal Vars:\t{self.n_nodal_variables}")
        print(f"Global Vars:\t{self.n_global_variables}")
        print(f"Total DoFs:\t{self.n_dofs}")

    def read_database_entry(
            self, datapath: str, verbose: bool = False) -> None:
        """
        Read a database entry or a standalone simulation result.

        Parameters
        ----------
        datapath : str
            The path to the directory of results.
        verbose : bool, default False
        """
        if not os.path.isdir(datapath):
            raise NotADirectoryError(f"{datapath} is not valid.")

        if verbose:
            print(f"***** Parsing data at {datapath} *****")

        # ======================================== Get macro data
        if len(self._times) == 0:
            self.parse_times(datapath)
        if len(self._grid) == 0:
            self.parse_grid(datapath)

        # ======================================== Go through directory
        skip = ["times", "grid", ".pdf", "k_eff"]
        for var in sorted(os.listdir(datapath)):
            if any([name in var for name in skip]):
                continue

            var_path = os.path.join(datapath, var)

            # ============================== If file, parse
            if os.path.isfile(var_path):
                self.parse_variable(var_path, False)

            # ============================== If directory, traverse it
            elif os.path.isdir(var_path):

                # ========================= Go through directory
                for varc in sorted(os.listdir(var_path)):
                    varc_filepath = os.path.join(var_path, varc)

                    # ==================== If file, parse
                    if os.path.isfile(varc_filepath):
                        self.parse_variable(varc_filepath, True)

                    # ==================== Otherwise, throw error
                    else:
                        raise NotImplementedError(
                            "Base level of variable definition reached. "
                            "Variable definitions passed component-wise "
                            "is not supported.")

    def combine_variables(self, nodal: bool = True,
                          variables: List[str] = None) -> ndarray:
        """
        Combine the variables into a dataset.

        The ordering of the DoFs in the vector is based on the value
        of `nodal`. If true, all DoFs that live on a node are
        contiguious, otherwise, all DoFs for a given unknown are
        contiguous.

        Parameters
        ----------
        nodal : bool, default True
            Boolean for nodal/block stacking
        variables : List[str]
            The variable_names of variables to be included.

        Returns
        -------
        ndarray (n_simulations, n_steps, n_dofs)
        """
        if not variables:
            variables = self.variable_names
        if isinstance(variables, str):
            variables = [variables]

        # ==================== Shorthand
        n_sims = self.n_simulations
        n_steps = self.n_timesteps
        n_pts = self.n_grid_points

        # ==================== Get n_dofs
        n_dofs = 0
        n_nv = 0  # number of nodal variables
        for var in variables:
            n_dofs += self.data[var].shape[2]
            if self.data[var].shape[2] > 1:
                n_nv += 1

        nn0, nnf = 0, n_pts  # start, end for nodal vars
        nnt = n_nv * n_pts  # end of nodal vars
        ng = nnt  # start of global var

        # =================================== Loop over variables
        data = np.zeros((n_sims, n_steps, n_dofs))
        for v, x in enumerate(self.data.values()):
            if self.variable_names[v] not in variables:
                continue

            # ============================== Loop over simulations
            for s in range(n_sims):

                # ========================= Loop over timesteps
                for t in range(n_steps):
                    x_st = x[s][t]

                    # ========== If this variable is nodal
                    if len(x_st) == n_pts:
                        if nodal:  # nodal stacking
                            data[s][t][v:nnt:n_nv] = x_st
                        else:  # block stacking
                            data[s][t][nn0:nnf] = x_st
                            nn0 += n_pts
                            nnf += n_pts

                    # ========== If this variable is global
                    elif len(x_st) == 1:
                        data[s][t][ng] = x[s][t][0]
                        ng += 1
        return data

    def stack_simulations(self, nodal: bool = True,
                          variables: List[str] = None) -> ndarray:
        """
        Stack each simulation data set into individual vectors.

        The ordering of the DoFs in the vector is based on the value
        of `nodal`. If true, all DoFs that live on a node are
        contiguious, otherwise, all DoFs for a given unknown are
        contiguous.

        Parameters
        ----------
        nodal : bool, default True
            Boolean for nodal/block stacking
        variables : List[str]
            The variable_names of variables to be included.

        Returns
        -------
        ndarray (n_simulations, n_steps * n_dofs)
        """
        if not variables:
            variables = self.variable_names
        if isinstance(variables, str):
            variables = [variables]

        # ======================================== Get combined data
        x = self.combine_variables(nodal, variables)
        n_dofs = x.shape[2]

        n_rows = self.n_simulations
        n_cols = self.n_timesteps * n_dofs
        data = np.zeros((n_rows, n_cols))

        # ======================================== Loop over simulations
        for s in range(self.n_simulations):
            n0, nf = 0, n_dofs

            # =================================== Loop over time steps
            for t in range(self.n_timesteps):
                data[s][n0:nf] = x[s][t]
                n0 += n_dofs
                nf += n_dofs
        return data

    def unstack_simulations(self, x: ndarray) -> ndarray:
        """
        Unstack a simulation vector or matrix.

        This converts simulation vectors into independent time
        series of snapshots.

        Parameters
        ----------
        x : ndarray (-1, n_steps * n_dofs)
            A matrix whose rows contains stacked simulation vectors.

        Returns
        -------
        ndarray (n_simulations, n_steps, n_dofs)
        """
        if x.ndim == 1:
            x = np.atleast_2d(x)
        return x.reshape(x.shape[0], self.n_timesteps, -1)

    def parse_times(self, filepath: str) -> None:
        """
        Parse the times.

        Parameters
        ----------
        filepath : str
            The path to the database directory.
        """
        times_filepath = os.path.join(filepath, "times.txt")
        if os.path.isfile(times_filepath):
            self._times = np.loadtxt(times_filepath)

    def parse_grid(self, filepath: str) -> None:
        """
        Parse the grid.

        Parameters
        ----------
        filepath : str
            The path to the database directory.
        """
        grid_filepath = os.path.join(filepath, "grid.txt")
        if os.path.isfile(grid_filepath):
            self._grid = np.loadtxt(grid_filepath)

    def parse_parameters(self, filepath: str) -> None:
        """
        Parse the parameters that define simulations.

        Parameters
        ----------
        filepath : str
            The path to the database directory.
        """
        params_filepath = os.path.join(filepath, "params.txt")
        if os.path.isfile(params_filepath):
            self._parameters = np.loadtxt(params_filepath)
            if self._parameters.ndim == 1:
                self._parameters = np.atleast_2d(self._parameters).T
            self._parameters = self._parameters

    def parse_variable(
            self, filepath: str, has_components: bool = False) -> None:
        """
        Parse a file containing simulation output data.

        Parameters
        ----------
        filepath : str
            The file path to the file to parse.
        has_components : bool, default False
            Flag for whether or not the file being parsed is
            a part of a set of components comprising a variable.
        """
        # ======================================== Get the data
        data = np.loadtxt(filepath)
        if data.ndim == 1:
            data = np.atleast_2d(data).T
        elif len(self._grid) == 0:
            raise ValueError(
                "A grid must be parsed before processing "
                "nodally defined variables.")
        elif data.shape[1] != self.n_grid_points:
            raise ValueError(
                "All nodally defined variables must have the "
                "same number of entries as the number of "
                "grid points.")

        # ======================================== Get the variable name
        name = self.get_key_from_file(filepath, has_components)

        # ======================================== Initialize the variable
        if name not in self._data:
            self._data[name]: List = []

            # ==================== Increment variable type counters
            if data.shape[1] == self.n_grid_points:
                self._n_nodal_variables += 1
            elif data.shape[1] == 1:
                self._n_global_variables += 1

        # ======================================== Store the data
        self._data[name].append([])  # creat simulation storage
        for t in range(data.shape[0]):
            self._data[name][-1].append(list(data[t]))

    def parse_keff(self, datapath: str) -> ndarray:
        """
        Parse k-eigenvalues from neutronics simulation databases.

        Parameters
        ----------
        datapath : str

        Returns
        -------
        ndarray (n_simulations, 2)
            The k-eigenvalues and rectivity of each simulation.
        """
        # number of digits in simulation directory
        n_digits = len(os.listdir(datapath)[0])

        # number of simulations
        n_simulations = len(os.listdir(datapath))
        if "params.txt" in os.listdir(datapath):
            n_simulations -= 1

        # ======================================== Loop over simulations
        data = np.zeros((n_simulations, 2))
        for s in range(n_simulations):
            simulation = str(s).zfill(n_digits)
            simulation_path = os.path.join(datapath, simulation)
            keff_path = os.path.join(simulation_path, "k_eff.txt")

            # =================================== Get k-eigenvalue
            if os.path.isfile(keff_path):
                with open(keff_path, "r") as f:
                    k = float(f.readlines()[0])
                    data[s][0] = k
                    data[s][1] = (k - 1.0) / k * 1.0e5
        return data

    @staticmethod
    def get_key_from_file(
            filepath: str, has_components: bool = False) -> str:
        """
        Get the variable name (key) from the filepath.

        Parameters
        ----------
        filepath : str
            The file path to the file to parse.
        has_components : bool, default False
            Flag for whether or not the file being parsed is
            a part of a set of components comprising a variable.

        Returns
        -------
        str
            The name of the variable.
        """
        # ======================================== Define the key
        name = filepath.split(".")[0]  # strip the file extension

        # If single variable, strip the directories
        if not has_components:
            return name.split("/")[-1]

        # If has components, join the innermost directory
        # name (variable name) with the filename by an
        # underscore
        else:
            return "_".join(name.split("/")[-2:])

    def reset(self) -> None:
        self._data.clear()
        self._times.clear()
        self._grid.clear()
        self._parameters.clear()
