from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

import attr

from ..types import UNSET, Unset
from ..util.serialization import is_not_none

T = TypeVar("T", bound="EndpointAddresses")


@attr.s(auto_attribs=True)
class EndpointAddresses:
    """Addresses at which an endpoint is reachable over the network.

    Attributes:
        grpc (Union[Unset, str]):
        rpcq (Union[Unset, str]):
    """

    grpc: Union[Unset, str] = UNSET
    rpcq: Union[Unset, str] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self, pick_by_predicate: Optional[Callable[[Any], bool]] = is_not_none) -> Dict[str, Any]:
        grpc = self.grpc
        rpcq = self.rpcq

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if grpc is not UNSET:
            field_dict["grpc"] = grpc
        if rpcq is not UNSET:
            field_dict["rpcq"] = rpcq

        field_dict = {k: v for k, v in field_dict.items() if v != UNSET}
        if pick_by_predicate is not None:
            field_dict = {k: v for k, v in field_dict.items() if pick_by_predicate(v)}

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        grpc = d.pop("grpc", UNSET)

        rpcq = d.pop("rpcq", UNSET)

        endpoint_addresses = cls(
            grpc=grpc,
            rpcq=rpcq,
        )

        endpoint_addresses.additional_properties = d
        return endpoint_addresses

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
