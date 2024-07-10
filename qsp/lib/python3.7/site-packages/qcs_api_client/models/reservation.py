import datetime
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

import attr
from dateutil.parser import isoparse
from rfc3339 import rfc3339

from ..models.account_type import AccountType
from ..types import UNSET, Unset
from ..util.serialization import is_not_none

T = TypeVar("T", bound="Reservation")


@attr.s(auto_attribs=True)
class Reservation:
    """
    Attributes:
        account_id (str): userId for `accountType` "user", group name for `accountType` "group".
        account_type (AccountType): There are two types of accounts within QCS: user (representing a single user in
            Okta) and group
            (representing one or more users in Okta).
        created_time (datetime.datetime):
        end_time (datetime.datetime):
        id (int):
        price (int):
        quantum_processor_id (str):
        start_time (datetime.datetime):
        user_id (str): Deprecated in favor of `accountId`.
        cancellation_billing_invoice_item_id (Union[Unset, str]):
        cancelled (Union[Unset, bool]):
        creation_billing_invoice_item_id (Union[Unset, str]):
        notes (Union[Unset, str]):
        updated_time (Union[Unset, datetime.datetime]):
    """

    account_id: str
    account_type: AccountType
    created_time: datetime.datetime
    end_time: datetime.datetime
    id: int
    price: int
    quantum_processor_id: str
    start_time: datetime.datetime
    user_id: str
    cancellation_billing_invoice_item_id: Union[Unset, str] = UNSET
    cancelled: Union[Unset, bool] = UNSET
    creation_billing_invoice_item_id: Union[Unset, str] = UNSET
    notes: Union[Unset, str] = UNSET
    updated_time: Union[Unset, datetime.datetime] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self, pick_by_predicate: Optional[Callable[[Any], bool]] = is_not_none) -> Dict[str, Any]:
        account_id = self.account_id
        account_type = self.account_type.value

        assert self.created_time.tzinfo is not None, "Datetime must have timezone information"
        created_time = rfc3339(self.created_time)

        assert self.end_time.tzinfo is not None, "Datetime must have timezone information"
        end_time = rfc3339(self.end_time)

        id = self.id
        price = self.price
        quantum_processor_id = self.quantum_processor_id
        assert self.start_time.tzinfo is not None, "Datetime must have timezone information"
        start_time = rfc3339(self.start_time)

        user_id = self.user_id
        cancellation_billing_invoice_item_id = self.cancellation_billing_invoice_item_id
        cancelled = self.cancelled
        creation_billing_invoice_item_id = self.creation_billing_invoice_item_id
        notes = self.notes
        updated_time: Union[Unset, str] = UNSET
        if not isinstance(self.updated_time, Unset):
            updated_time = rfc3339(self.updated_time)

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "accountId": account_id,
                "accountType": account_type,
                "createdTime": created_time,
                "endTime": end_time,
                "id": id,
                "price": price,
                "quantumProcessorId": quantum_processor_id,
                "startTime": start_time,
                "userId": user_id,
            }
        )
        if cancellation_billing_invoice_item_id is not UNSET:
            field_dict["cancellationBillingInvoiceItemId"] = cancellation_billing_invoice_item_id
        if cancelled is not UNSET:
            field_dict["cancelled"] = cancelled
        if creation_billing_invoice_item_id is not UNSET:
            field_dict["creationBillingInvoiceItemId"] = creation_billing_invoice_item_id
        if notes is not UNSET:
            field_dict["notes"] = notes
        if updated_time is not UNSET:
            field_dict["updatedTime"] = updated_time

        field_dict = {k: v for k, v in field_dict.items() if v != UNSET}
        if pick_by_predicate is not None:
            field_dict = {k: v for k, v in field_dict.items() if pick_by_predicate(v)}

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        account_id = d.pop("accountId")

        account_type = AccountType(d.pop("accountType"))

        created_time = isoparse(d.pop("createdTime"))

        end_time = isoparse(d.pop("endTime"))

        id = d.pop("id")

        price = d.pop("price")

        quantum_processor_id = d.pop("quantumProcessorId")

        start_time = isoparse(d.pop("startTime"))

        user_id = d.pop("userId")

        cancellation_billing_invoice_item_id = d.pop("cancellationBillingInvoiceItemId", UNSET)

        cancelled = d.pop("cancelled", UNSET)

        creation_billing_invoice_item_id = d.pop("creationBillingInvoiceItemId", UNSET)

        notes = d.pop("notes", UNSET)

        _updated_time = d.pop("updatedTime", UNSET)
        updated_time: Union[Unset, datetime.datetime]
        if isinstance(_updated_time, Unset):
            updated_time = UNSET
        else:
            updated_time = isoparse(_updated_time)

        reservation = cls(
            account_id=account_id,
            account_type=account_type,
            created_time=created_time,
            end_time=end_time,
            id=id,
            price=price,
            quantum_processor_id=quantum_processor_id,
            start_time=start_time,
            user_id=user_id,
            cancellation_billing_invoice_item_id=cancellation_billing_invoice_item_id,
            cancelled=cancelled,
            creation_billing_invoice_item_id=creation_billing_invoice_item_id,
            notes=notes,
            updated_time=updated_time,
        )

        reservation.additional_properties = d
        return reservation

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
