from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, cast

import attr

from ..types import UNSET
from ..util.serialization import is_not_none

T = TypeVar("T", bound="Edge")


@attr.s(auto_attribs=True)
class Edge:
    """A degree-two logical connection in the quantum processor's architecture.

    The existence of an edge in the ISA `Architecture` does not necessarily mean that a given 2Q
    operation will be available on the edge. This information is conveyed by the presence of the
    two `node_id` values in instances of `Instruction`.

    Note that edges are undirected in this model. Thus edge :math:`(a, b)` is equivalent to edge
    :math:`(b, a)`.

        Attributes:
            node_ids (List[int]): The integer ids of the computational nodes at the two ends of the edge. Order is not
                important; an architecture edge is treated as undirected.
    """

    node_ids: List[int]

    def to_dict(self, pick_by_predicate: Optional[Callable[[Any], bool]] = is_not_none) -> Dict[str, Any]:
        node_ids = self.node_ids

        field_dict: Dict[str, Any] = {}
        field_dict.update(
            {
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
        node_ids = cast(List[int], d.pop("node_ids"))

        edge = cls(
            node_ids=node_ids,
        )

        return edge
