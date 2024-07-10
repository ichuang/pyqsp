from enum import Enum


class BillingInvoiceLineLineItemType(str, Enum):
    INVOICEITEM = "invoiceitem"
    SUBSCRIPTION = "subscription"

    def __str__(self) -> str:
        return str(self.value)
