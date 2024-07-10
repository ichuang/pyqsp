from enum import Enum


class ChecksumDescriptionType(str, Enum):
    MD5 = "md5"

    def __str__(self) -> str:
        return str(self.value)
