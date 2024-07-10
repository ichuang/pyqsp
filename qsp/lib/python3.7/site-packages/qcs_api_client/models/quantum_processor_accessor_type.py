from enum import Enum


class QuantumProcessorAccessorType(str, Enum):
    GATEWAY_V1 = "gateway.v1"

    def __str__(self) -> str:
        return str(self.value)
