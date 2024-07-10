from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union, cast

import attr

from ..models.account_type import AccountType
from ..types import UNSET, Unset
from ..util.serialization import is_not_none

T = TypeVar("T", bound="CreateEngagementRequest")


@attr.s(auto_attribs=True)
class CreateEngagementRequest:
    """
    Attributes:
        account_id (Union[Unset, str]): Either the client's user ID or the name of a group on behalf of which the client
            wishes to engage. This value will override any values set in the headers.
        account_type (Union[Unset, AccountType]): There are two types of accounts within QCS: user (representing a
            single user in Okta) and group
            (representing one or more users in Okta).
        endpoint_id (Union[Unset, str]): Unique, opaque identifier for the endpoint
        quantum_processor_id (Union[Unset, str]): Public identifier for a quantum processor [example: Aspen-1]
        tags (Union[Unset, List[str]]): Tags recorded on QPU requests, which reporting services may later use for
            querying usage records.
    """

    account_id: Union[Unset, str] = UNSET
    account_type: Union[Unset, AccountType] = UNSET
    endpoint_id: Union[Unset, str] = UNSET
    quantum_processor_id: Union[Unset, str] = UNSET
    tags: Union[Unset, List[str]] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self, pick_by_predicate: Optional[Callable[[Any], bool]] = is_not_none) -> Dict[str, Any]:
        account_id = self.account_id
        account_type: Union[Unset, str] = UNSET
        if not isinstance(self.account_type, Unset):
            account_type = self.account_type.value

        endpoint_id = self.endpoint_id
        quantum_processor_id = self.quantum_processor_id
        tags: Union[Unset, List[str]] = UNSET
        if not isinstance(self.tags, Unset):
            tags = self.tags

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if account_id is not UNSET:
            field_dict["accountId"] = account_id
        if account_type is not UNSET:
            field_dict["accountType"] = account_type
        if endpoint_id is not UNSET:
            field_dict["endpointId"] = endpoint_id
        if quantum_processor_id is not UNSET:
            field_dict["quantumProcessorId"] = quantum_processor_id
        if tags is not UNSET:
            field_dict["tags"] = tags

        field_dict = {k: v for k, v in field_dict.items() if v != UNSET}
        if pick_by_predicate is not None:
            field_dict = {k: v for k, v in field_dict.items() if pick_by_predicate(v)}

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        account_id = d.pop("accountId", UNSET)

        _account_type = d.pop("accountType", UNSET)
        account_type: Union[Unset, AccountType]
        if isinstance(_account_type, Unset):
            account_type = UNSET
        else:
            account_type = AccountType(_account_type)

        endpoint_id = d.pop("endpointId", UNSET)

        quantum_processor_id = d.pop("quantumProcessorId", UNSET)

        tags = cast(List[str], d.pop("tags", UNSET))

        create_engagement_request = cls(
            account_id=account_id,
            account_type=account_type,
            endpoint_id=endpoint_id,
            quantum_processor_id=quantum_processor_id,
            tags=tags,
        )

        create_engagement_request.additional_properties = d
        return create_engagement_request

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
