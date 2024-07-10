from enum import Enum


class BillingPriceScheme(str, Enum):
    PER_UNIT = "per_unit"
    TIERED = "tiered"

    def __str__(self) -> str:
        return str(self.value)
