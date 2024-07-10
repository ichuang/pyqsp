from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union, cast

import attr

from ..models.endpoint_addresses import EndpointAddresses
from ..types import UNSET, Unset
from ..util.serialization import is_not_none

T = TypeVar("T", bound="Endpoint")


@attr.s(auto_attribs=True)
class Endpoint:
    """An Endpoint is the entry point for remote access to a QuantumProcessor.

    Attributes:
        addresses (EndpointAddresses): Addresses at which an endpoint is reachable over the network.
        healthy (bool): Whether the endpoint is operating as intended
        id (str): Unique, opaque identifier for the endpoint
        mock (bool): Whether the endpoint serves simulated or substituted data for testing purposes
        address (Union[Unset, None, str]): Network address at which the endpoint is locally reachable
        datacenter (Union[Unset, str]): Datacenter within which the endpoint is deployed
        quantum_processor_ids (Union[Unset, List[str]]): Public identifiers for quantum processors served by this
            endpoint.
    """

    addresses: EndpointAddresses
    healthy: bool
    id: str
    mock: bool
    address: Union[Unset, None, str] = UNSET
    datacenter: Union[Unset, str] = UNSET
    quantum_processor_ids: Union[Unset, List[str]] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self, pick_by_predicate: Optional[Callable[[Any], bool]] = is_not_none) -> Dict[str, Any]:
        addresses = self.addresses.to_dict()

        healthy = self.healthy
        id = self.id
        mock = self.mock
        address = self.address
        datacenter = self.datacenter
        quantum_processor_ids: Union[Unset, List[str]] = UNSET
        if not isinstance(self.quantum_processor_ids, Unset):
            quantum_processor_ids = self.quantum_processor_ids

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "addresses": addresses,
                "healthy": healthy,
                "id": id,
                "mock": mock,
            }
        )
        if address is not UNSET:
            field_dict["address"] = address
        if datacenter is not UNSET:
            field_dict["datacenter"] = datacenter
        if quantum_processor_ids is not UNSET:
            field_dict["quantumProcessorIds"] = quantum_processor_ids

        field_dict = {k: v for k, v in field_dict.items() if v != UNSET}
        if pick_by_predicate is not None:
            field_dict = {k: v for k, v in field_dict.items() if pick_by_predicate(v)}

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        addresses = EndpointAddresses.from_dict(d.pop("addresses"))

        healthy = d.pop("healthy")

        id = d.pop("id")

        mock = d.pop("mock")

        address = d.pop("address", UNSET)

        datacenter = d.pop("datacenter", UNSET)

        quantum_processor_ids = cast(List[str], d.pop("quantumProcessorIds", UNSET))

        endpoint = cls(
            addresses=addresses,
            healthy=healthy,
            id=id,
            mock=mock,
            address=address,
            datacenter=datacenter,
            quantum_processor_ids=quantum_processor_ids,
        )

        endpoint.additional_properties = d
        return endpoint

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
