from enum import Enum


class NomadJobDatacenters(str, Enum):
    BERKELEY_775 = "berkeley-775"
    FREMONT_FAB = "fremont-fab"
    OXFORD_INSTRUMENTS = "oxford-instruments"

    def __str__(self) -> str:
        return str(self.value)
