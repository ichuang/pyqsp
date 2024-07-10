from enum import Enum


class Family(str, Enum):
    NONE = "None"
    FULL = "Full"
    ASPEN = "Aspen"
    ANKAA = "Ankaa"

    def __str__(self) -> str:
        return str(self.value)
