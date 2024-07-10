from typing import Any, Callable, Dict, Optional, Type, TypeVar

import attr

from ..types import UNSET
from ..util.serialization import is_not_none

T = TypeVar("T", bound="Node")


@attr.s(auto_attribs=True)
class Node:
    """A logical node in the quantum processor's architecture.

    The existence of a node in the ISA `Architecture` does not necessarily mean that a given 1Q
    operation will be available on the node. This information is conveyed by the presence of the
    specific `node_id` in instances of `Instruction`.

        Attributes:
            node_id (int): An integer id assigned to the computational node. The ids may not be contiguous and will be
                assigned based on the architecture family.
    """

    node_id: int

    def to_dict(self, pick_by_predicate: Optional[Callable[[Any], bool]] = is_not_none) -> Dict[str, Any]:
        node_id = self.node_id

        field_dict: Dict[str, Any] = {}
        field_dict.update(
            {
                "node_id": node_id,
            }
        )

        field_dict = {k: v for k, v in field_dict.items() if v != UNSET}
        if pick_by_predicate is not None:
            field_dict = {k: v for k, v in field_dict.items() if pick_by_predicate(v)}

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        node_id = d.pop("node_id")

        node = cls(
            node_id=node_id,
        )

        return node
