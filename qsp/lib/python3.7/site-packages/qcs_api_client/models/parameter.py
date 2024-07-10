from typing import Any, Callable, Dict, Optional, Type, TypeVar

import attr

from ..types import UNSET
from ..util.serialization import is_not_none

T = TypeVar("T", bound="Parameter")


@attr.s(auto_attribs=True)
class Parameter:
    """A parameter to an operation.

    Attributes:
        name (str): The name of the parameter, such as the name of a mathematical symbol.
    """

    name: str

    def to_dict(self, pick_by_predicate: Optional[Callable[[Any], bool]] = is_not_none) -> Dict[str, Any]:
        name = self.name

        field_dict: Dict[str, Any] = {}
        field_dict.update(
            {
                "name": name,
            }
        )

        field_dict = {k: v for k, v in field_dict.items() if v != UNSET}
        if pick_by_predicate is not None:
            field_dict = {k: v for k, v in field_dict.items() if pick_by_predicate(v)}

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        name = d.pop("name")

        parameter = cls(
            name=name,
        )

        return parameter
