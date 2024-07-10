from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

import attr

from ..types import UNSET, Unset
from ..util.serialization import is_not_none

T = TypeVar("T", bound="RestartEndpointRequest")


@attr.s(auto_attribs=True)
class RestartEndpointRequest:
    """
    Attributes:
        component_name (Union[Unset, str]): Individual component to restart
    """

    component_name: Union[Unset, str] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self, pick_by_predicate: Optional[Callable[[Any], bool]] = is_not_none) -> Dict[str, Any]:
        component_name = self.component_name

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if component_name is not UNSET:
            field_dict["componentName"] = component_name

        field_dict = {k: v for k, v in field_dict.items() if v != UNSET}
        if pick_by_predicate is not None:
            field_dict = {k: v for k, v in field_dict.items() if pick_by_predicate(v)}

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        component_name = d.pop("componentName", UNSET)

        restart_endpoint_request = cls(
            component_name=component_name,
        )

        restart_endpoint_request.additional_properties = d
        return restart_endpoint_request

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
