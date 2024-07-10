from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

import attr

from ..types import UNSET, Unset
from ..util.serialization import is_not_none

T = TypeVar("T", bound="Tier")


@attr.s(auto_attribs=True)
class Tier:
    """
    Attributes:
        up_to (int): A value of `-1` will imply that this tier is up to Infinity.
        flat_amount (Union[Unset, int]):
        flat_amount_decimal (Union[Unset, float]):
        unit_amount (Union[Unset, int]):
        unit_amount_decimal (Union[Unset, float]):
    """

    up_to: int
    flat_amount: Union[Unset, int] = UNSET
    flat_amount_decimal: Union[Unset, float] = UNSET
    unit_amount: Union[Unset, int] = UNSET
    unit_amount_decimal: Union[Unset, float] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self, pick_by_predicate: Optional[Callable[[Any], bool]] = is_not_none) -> Dict[str, Any]:
        up_to = self.up_to
        flat_amount = self.flat_amount
        flat_amount_decimal = self.flat_amount_decimal
        unit_amount = self.unit_amount
        unit_amount_decimal = self.unit_amount_decimal

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "upTo": up_to,
            }
        )
        if flat_amount is not UNSET:
            field_dict["flatAmount"] = flat_amount
        if flat_amount_decimal is not UNSET:
            field_dict["flatAmountDecimal"] = flat_amount_decimal
        if unit_amount is not UNSET:
            field_dict["unitAmount"] = unit_amount
        if unit_amount_decimal is not UNSET:
            field_dict["unitAmountDecimal"] = unit_amount_decimal

        field_dict = {k: v for k, v in field_dict.items() if v != UNSET}
        if pick_by_predicate is not None:
            field_dict = {k: v for k, v in field_dict.items() if pick_by_predicate(v)}

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        up_to = d.pop("upTo")

        flat_amount = d.pop("flatAmount", UNSET)

        flat_amount_decimal = d.pop("flatAmountDecimal", UNSET)

        unit_amount = d.pop("unitAmount", UNSET)

        unit_amount_decimal = d.pop("unitAmountDecimal", UNSET)

        tier = cls(
            up_to=up_to,
            flat_amount=flat_amount,
            flat_amount_decimal=flat_amount_decimal,
            unit_amount=unit_amount,
            unit_amount_decimal=unit_amount_decimal,
        )

        tier.additional_properties = d
        return tier

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
