from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, cast

import attr

from ..models.characteristic import Characteristic
from ..types import UNSET
from ..util.serialization import is_not_none

T = TypeVar("T", bound="OperationSite")


@attr.s(auto_attribs=True)
class OperationSite:
    """A site for an operation, with its site-dependent characteristics.

    Attributes:
        characteristics (List[Characteristic]): The list of site-dependent characteristics of this operation.
        node_ids (List[int]): The list of architecture node ids for the site. The order of these node ids obey the
            definition of node symmetry from the enclosing operation.
    """

    characteristics: List[Characteristic]
    node_ids: List[int]

    def to_dict(self, pick_by_predicate: Optional[Callable[[Any], bool]] = is_not_none) -> Dict[str, Any]:
        characteristics = []
        for characteristics_item_data in self.characteristics:
            characteristics_item = characteristics_item_data.to_dict()

            characteristics.append(characteristics_item)

        node_ids = self.node_ids

        field_dict: Dict[str, Any] = {}
        field_dict.update(
            {
                "characteristics": characteristics,
                "node_ids": node_ids,
            }
        )

        field_dict = {k: v for k, v in field_dict.items() if v != UNSET}
        if pick_by_predicate is not None:
            field_dict = {k: v for k, v in field_dict.items() if pick_by_predicate(v)}

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        characteristics = []
        _characteristics = d.pop("characteristics")
        for characteristics_item_data in _characteristics:
            characteristics_item = Characteristic.from_dict(characteristics_item_data)

            characteristics.append(characteristics_item)

        node_ids = cast(List[int], d.pop("node_ids"))

        operation_site = cls(
            characteristics=characteristics,
            node_ids=node_ids,
        )

        return operation_site
