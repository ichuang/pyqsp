from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

import attr

from ..types import UNSET, Unset
from ..util.serialization import is_not_none

T = TypeVar("T", bound="GetQuiltCalibrationsResponse")


@attr.s(auto_attribs=True)
class GetQuiltCalibrationsResponse:
    """
    Attributes:
        quilt (str): Calibrations suitable for use in a Quil-T program
        settings_timestamp (Union[Unset, str]): ISO8601 timestamp of the settings used to generate these calibrations
    """

    quilt: str
    settings_timestamp: Union[Unset, str] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self, pick_by_predicate: Optional[Callable[[Any], bool]] = is_not_none) -> Dict[str, Any]:
        quilt = self.quilt
        settings_timestamp = self.settings_timestamp

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "quilt": quilt,
            }
        )
        if settings_timestamp is not UNSET:
            field_dict["settingsTimestamp"] = settings_timestamp

        field_dict = {k: v for k, v in field_dict.items() if v != UNSET}
        if pick_by_predicate is not None:
            field_dict = {k: v for k, v in field_dict.items() if pick_by_predicate(v)}

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        quilt = d.pop("quilt")

        settings_timestamp = d.pop("settingsTimestamp", UNSET)

        get_quilt_calibrations_response = cls(
            quilt=quilt,
            settings_timestamp=settings_timestamp,
        )

        get_quilt_calibrations_response.additional_properties = d
        return get_quilt_calibrations_response

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
