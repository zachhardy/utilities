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

    def initialize_storage(self) -> None:
        cls_name = self.__class__.__name__
        raise NotImplementedError(
            f'{cls_name}.initialize_storage must be implemented.')

    def read_simulation_data(self) -> None:
        cls_name = self.__class__.__name__
        raise NotImplementedError(
            f'{cls_name}.read_simulation_data must be implemented.')

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



