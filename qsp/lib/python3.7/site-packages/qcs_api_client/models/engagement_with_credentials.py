from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union, cast

import attr

from ..models.account_type import AccountType
from ..models.engagement_credentials import EngagementCredentials
from ..types import UNSET, Unset
from ..util.serialization import is_not_none

T = TypeVar("T", bound="EngagementWithCredentials")


@attr.s(auto_attribs=True)
class EngagementWithCredentials:
    """An engagement is the authorization of a user to execute work on a Quantum Processor Endpoint.

    Attributes:
        address (str): The network address of the endpoint to which this engagement grants access
        credentials (EngagementCredentials): Credentials are the ZeroMQ CURVE Keys used to encrypt the connection with
            the Quantum Processor
            Endpoint.
        endpoint_id (str): The ID of the endpoint to which this engagement grants access
        expires_at (str): Time after which the engagement is no longer valid. Given in RFC3339 format.
        user_id (str):
        account_id (Union[Unset, str]): User ID or group name on behalf of which the engagement is made.
        account_type (Union[Unset, AccountType]): There are two types of accounts within QCS: user (representing a
            single user in Okta) and group
            (representing one or more users in Okta).
        minimum_priority (Union[Unset, int]): The minimum priority value allowed for execution
        quantum_processor_ids (Union[Unset, List[str]]): The quantum processors for which this engagement enables access
            and execution
        tags (Union[Unset, List[str]]): Tags recorded on QPU requests and recorded on usage records.
    """

    address: str
    credentials: EngagementCredentials
    endpoint_id: str
    expires_at: str
    user_id: str
    account_id: Union[Unset, str] = UNSET
    account_type: Union[Unset, AccountType] = UNSET
    minimum_priority: Union[Unset, int] = UNSET
    quantum_processor_ids: Union[Unset, List[str]] = UNSET
    tags: Union[Unset, List[str]] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self, pick_by_predicate: Optional[Callable[[Any], bool]] = is_not_none) -> Dict[str, Any]:
        address = self.address
        credentials = self.credentials.to_dict()

        endpoint_id = self.endpoint_id
        expires_at = self.expires_at
        user_id = self.user_id
        account_id = self.account_id
        account_type: Union[Unset, str] = UNSET
        if not isinstance(self.account_type, Unset):
            account_type = self.account_type.value

        minimum_priority = self.minimum_priority
        quantum_processor_ids: Union[Unset, List[str]] = UNSET
        if not isinstance(self.quantum_processor_ids, Unset):
            quantum_processor_ids = self.quantum_processor_ids

        tags: Union[Unset, List[str]] = UNSET
        if not isinstance(self.tags, Unset):
            tags = self.tags

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "address": address,
                "credentials": credentials,
                "endpointId": endpoint_id,
                "expiresAt": expires_at,
                "userId": user_id,
            }
        )
        if account_id is not UNSET:
            field_dict["accountId"] = account_id
        if account_type is not UNSET:
            field_dict["accountType"] = account_type
        if minimum_priority is not UNSET:
            field_dict["minimumPriority"] = minimum_priority
        if quantum_processor_ids is not UNSET:
            field_dict["quantumProcessorIds"] = quantum_processor_ids
        if tags is not UNSET:
            field_dict["tags"] = tags

        field_dict = {k: v for k, v in field_dict.items() if v != UNSET}
        if pick_by_predicate is not None:
            field_dict = {k: v for k, v in field_dict.items() if pick_by_predicate(v)}

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        address = d.pop("address")

        credentials = EngagementCredentials.from_dict(d.pop("credentials"))

        endpoint_id = d.pop("endpointId")

        expires_at = d.pop("expiresAt")

        user_id = d.pop("userId")

        account_id = d.pop("accountId", UNSET)

        _account_type = d.pop("accountType", UNSET)
        account_type: Union[Unset, AccountType]
        if isinstance(_account_type, Unset):
            account_type = UNSET
        else:
            account_type = AccountType(_account_type)

        minimum_priority = d.pop("minimumPriority", UNSET)

        quantum_processor_ids = cast(List[str], d.pop("quantumProcessorIds", UNSET))

        tags = cast(List[str], d.pop("tags", UNSET))

        engagement_with_credentials = cls(
            address=address,
            credentials=credentials,
            endpoint_id=endpoint_id,
            expires_at=expires_at,
            user_id=user_id,
            account_id=account_id,
            account_type=account_type,
            minimum_priority=minimum_priority,
            quantum_processor_ids=quantum_processor_ids,
            tags=tags,
        )

        engagement_with_credentials.additional_properties = d
        return engagement_with_credentials

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
