import os
import struct

import numpy as np

from numpy import ndarray
from typing import List, Tuple

from pyPDEs.utilities import Vector


class SimulationReader:
    """
    Base class for reading simulation data
    """
    def __init__(self, path: str) -> None:
        if not os.path.isdir(path):
            raise NotADirectoryError(
                'The provided path is not a valid directory.')

        self.path: str = os.path.abspath(path)
        self.n_snapshots: int = 0

    def initialize_storage(self) -> None:
        cls_name = self.__class__.__name__
        raise NotImplementedError(
            f'{cls_name}.initialize_storage must be implemented.')

    def read_simulation_data(self) -> None:
        cls_name = self.__class__.__name__
        raise NotImplementedError(
            f'{cls_name}.read_simulation_data must be implemented.')

    def create_simulation_matrix(
            self, variables: List[str] = None) -> ndarray:
        cls_name = self.__class__.__name__
        raise NotImplementedError(
            f'{cls_name}.create_simulation_matrix must be implemented.')

    def create_simulation_vector(
            self, variables: List[str] = None) -> ndarray:
        """
        Create a simulation matrix.

        Parameters
        ----------
        variables : List[str], default None
            The variables to stack.

        Returns
        -------
        ndarray (n_snapshots * varies,)
        """
        data = self.create_simulation_matrix(variables)
        return data.reshape(data.size, 1)

    def get_variable_by_key(self, key: str) -> ndarray:
        cls_name = self.__class__.__name__
        raise NotImplementedError(
            f'{cls_name}.get_variable_by_key must be implemented.')

    def clear(self) -> None:
        self.__init__(self.path)

    @staticmethod
    def read_double(file) -> float:
        return struct.unpack('d', file.read(8))[0]

    @staticmethod
    def read_uint64_t(file) -> int:
        return struct.unpack('Q', file.read(8))[0]

    @staticmethod
    def read_unsigned_int(file) -> int:
        return struct.unpack('I', file.read(4))[0]



