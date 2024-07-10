import datetime
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar

import attr
from dateutil.parser import isoparse
from rfc3339 import rfc3339

from ..models.billing_invoice_status import BillingInvoiceStatus
from ..types import UNSET
from ..util.serialization import is_not_none

T = TypeVar("T", bound="BillingUpcomingInvoice")


@attr.s(auto_attribs=True)
class BillingUpcomingInvoice:
    """An unfinalized billing invoice.

    Attributes:
        period_end (datetime.datetime):
        period_start (datetime.datetime):
        starting_balance (int):
        status (BillingInvoiceStatus):
        subtotal (int):
        tax (int):
        total (int):
    """

    period_end: datetime.datetime
    period_start: datetime.datetime
    starting_balance: int
    status: BillingInvoiceStatus
    subtotal: int
    tax: int
    total: int
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self, pick_by_predicate: Optional[Callable[[Any], bool]] = is_not_none) -> Dict[str, Any]:
        assert self.period_end.tzinfo is not None, "Datetime must have timezone information"
        period_end = rfc3339(self.period_end)

        assert self.period_start.tzinfo is not None, "Datetime must have timezone information"
        period_start = rfc3339(self.period_start)

        starting_balance = self.starting_balance
        status = self.status.value

        subtotal = self.subtotal
        tax = self.tax
        total = self.total

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "periodEnd": period_end,
                "periodStart": period_start,
                "startingBalance": starting_balance,
                "status": status,
                "subtotal": subtotal,
                "tax": tax,
                "total": total,
            }
        )

        field_dict = {k: v for k, v in field_dict.items() if v != UNSET}
        if pick_by_predicate is not None:
            field_dict = {k: v for k, v in field_dict.items() if pick_by_predicate(v)}

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        period_end = isoparse(d.pop("periodEnd"))

        period_start = isoparse(d.pop("periodStart"))

        starting_balance = d.pop("startingBalance")

        status = BillingInvoiceStatus(d.pop("status"))

        subtotal = d.pop("subtotal")

        tax = d.pop("tax")

        total = d.pop("total")

        billing_upcoming_invoice = cls(
            period_end=period_end,
            period_start=period_start,
            starting_balance=starting_balance,
            status=status,
            subtotal=subtotal,
            tax=tax,
            total=total,
        )

        billing_upcoming_invoice.additional_properties = d
        return billing_upcoming_invoice

    @property
    def additional_keys(self) -> List[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
