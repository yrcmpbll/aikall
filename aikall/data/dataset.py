from abc import ABC, abstractmethod


class Dataset(ABC):

    def __init__(self) -> None:
        super().__init__()