from enum import Enum


class BillingPriceRecurrenceAggregateUsage(str, Enum):
    LAST_DURING_PERIOD = "last_during_period"
    LAST_EVER = "last_ever"
    MAX = "max"
    SUM = "sum"

    def __str__(self) -> str:
        return str(self.value)
