import datetime
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

import attr
from dateutil.parser import isoparse
from rfc3339 import rfc3339

from ..models.account_type import AccountType
from ..types import UNSET, Unset
from ..util.serialization import is_not_none

T = TypeVar("T", bound="CreateReservationRequest")


@attr.s(auto_attribs=True)
class CreateReservationRequest:
    """
    Attributes:
        end_time (datetime.datetime):
        quantum_processor_id (str):
        start_time (datetime.datetime):
        account_id (Union[Unset, str]): userId for `accountType` "user", group name for `accountType` "group".
        account_type (Union[Unset, AccountType]): There are two types of accounts within QCS: user (representing a
            single user in Okta) and group
            (representing one or more users in Okta).
        notes (Union[Unset, str]):
    """

    end_time: datetime.datetime
    quantum_processor_id: str
    start_time: datetime.datetime
    account_id: Union[Unset, str] = UNSET
    account_type: Union[Unset, AccountType] = UNSET
    notes: Union[Unset, str] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self, pick_by_predicate: Optional[Callable[[Any], bool]] = is_not_none) -> Dict[str, Any]:
        assert self.end_time.tzinfo is not None, "Datetime must have timezone information"
        end_time = rfc3339(self.end_time)

        quantum_processor_id = self.quantum_processor_id
        assert self.start_time.tzinfo is not None, "Datetime must have timezone information"
        start_time = rfc3339(self.start_time)

        account_id = self.account_id
        account_type: Union[Unset, str] = UNSET
        if not isinstance(self.account_type, Unset):
            account_type = self.account_type.value

        notes = self.notes

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "endTime": end_time,
                "quantumProcessorId": quantum_processor_id,
                "startTime": start_time,
            }
        )
        if account_id is not UNSET:
            field_dict["accountId"] = account_id
        if account_type is not UNSET:
            field_dict["accountType"] = account_type
        if notes is not UNSET:
            field_dict["notes"] = notes

        field_dict = {k: v for k, v in field_dict.items() if v != UNSET}
        if pick_by_predicate is not None:
            field_dict = {k: v for k, v in field_dict.items() if pick_by_predicate(v)}

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        end_time = isoparse(d.pop("endTime"))

        quantum_processor_id = d.pop("quantumProcessorId")

        start_time = isoparse(d.pop("startTime"))

        account_id = d.pop("accountId", UNSET)

        _account_type = d.pop("accountType", UNSET)
        account_type: Union[Unset, AccountType]
        if isinstance(_account_type, Unset):
            account_type = UNSET
        else:
            account_type = AccountType(_account_type)

        notes = d.pop("notes", UNSET)

        create_reservation_request = cls(
            end_time=end_time,
            quantum_processor_id=quantum_processor_id,
            start_time=start_time,
            account_id=account_id,
            account_type=account_type,
            notes=notes,
        )

        create_reservation_request.additional_properties = d
        return create_reservation_request

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
