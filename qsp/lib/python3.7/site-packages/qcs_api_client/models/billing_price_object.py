from enum import Enum


class BillingPriceObject(str, Enum):
    PRICE = "price"

    def __str__(self) -> str:
        return str(self.value)
