from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

import attr

from ..models.quantum_processor_accessor_type import QuantumProcessorAccessorType
from ..types import UNSET, Unset
from ..util.serialization import is_not_none

T = TypeVar("T", bound="QuantumProcessorAccessor")


@attr.s(auto_attribs=True)
class QuantumProcessorAccessor:
    """Method of accessing an available QPU.

    Attributes:
        access_type (QuantumProcessorAccessorType): Types of access mechanisms for a QPU. Each accessor type has its own
            access characteristics, benefits,
            and drawbacks.
        live (bool): Whether an accessor represents access to a physical, live quantum processor. When false, this
            accessor provides access instead to a simulated or test QPU.
        url (str): Address used to connect to the accessor.
        id (Union[Unset, str]): Unique identifier for the accessor.
        rank (Union[Unset, int]): Rank of this accessor against others for the same QPU. If two accessors both serve a
            client's purposes, that with the lower rank value should be used for access.
    """

    access_type: QuantumProcessorAccessorType
    live: bool
    url: str
    id: Union[Unset, str] = UNSET
    rank: Union[Unset, int] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self, pick_by_predicate: Optional[Callable[[Any], bool]] = is_not_none) -> Dict[str, Any]:
        access_type = self.access_type.value

        live = self.live
        url = self.url
        id = self.id
        rank = self.rank

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "accessType": access_type,
                "live": live,
                "url": url,
            }
        )
        if id is not UNSET:
            field_dict["id"] = id
        if rank is not UNSET:
            field_dict["rank"] = rank

        field_dict = {k: v for k, v in field_dict.items() if v != UNSET}
        if pick_by_predicate is not None:
            field_dict = {k: v for k, v in field_dict.items() if pick_by_predicate(v)}

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        access_type = QuantumProcessorAccessorType(d.pop("accessType"))

        live = d.pop("live")

        url = d.pop("url")

        id = d.pop("id", UNSET)

        rank = d.pop("rank", UNSET)

        quantum_processor_accessor = cls(
            access_type=access_type,
            live=live,
            url=url,
            id=id,
            rank=rank,
        )

        quantum_processor_accessor.additional_properties = d
        return quantum_processor_accessor

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
