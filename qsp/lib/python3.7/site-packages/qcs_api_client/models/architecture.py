from typing import Any, Callable, Dict, List, Optional, Type, TypeVar

import attr

from ..models.edge import Edge
from ..models.family import Family
from ..models.node import Node
from ..types import UNSET
from ..util.serialization import is_not_none

T = TypeVar("T", bound="Architecture")


@attr.s(auto_attribs=True)
class Architecture:
    """Represents the logical underlying architecture of a quantum processor.

    The architecture is defined in detail by the nodes and edges that constitute the quantum
    processor. This defines the set of all nodes that could be operated upon, and indicates to
    some approximation their physical layout. The main purpose of this is to support geometry
    calculations that are independent of the available operations, and rendering ISA-based
    information. Architecture layouts are defined by the `family`, as follows.

    The "Aspen" family of quantum processor indicates a 2D planar grid layout of octagon unit
    cells. The `node_id` in this architecture is computed as :math:`100 p_y + 10 p_x + p_u` where
    :math:`p_y` is the zero-based Y position in the unit cell grid, :math:`p_x` is the zero-based
    X position in the unit cell grid, and :math:`p_u` is the zero-based position in the octagon
    unit cell and always ranges from 0 to 7. This scheme has a natural size limit of a 10x10
    unit cell grid, which permits the architecture to scale up to 800 nodes.

    Note that the operations that are actually available are defined entirely by `Operation`
    instances. The presence of a node or edge in the `Architecture` model provides no guarantee
    that any 1Q or 2Q operation will be available to users writing QUIL programs.

        Attributes:
            edges (List[Edge]): A list of all computational edges in the instruction set architecture.
            family (Family): Family identifier.

                Value 'None' implies the architecture has no specific layout topology.
                Value 'Full' implies that each node is connected to every other (a fully-connected architecture)

                For other values based on deployed architecture layouts (e.g. `Aspen` and `Ankaa`), refer to
                the architecture classes themselves for more details.
            nodes (List[Node]): A list of all computational nodes in the instruction set architecture.
    """

    edges: List[Edge]
    family: Family
    nodes: List[Node]

    def to_dict(self, pick_by_predicate: Optional[Callable[[Any], bool]] = is_not_none) -> Dict[str, Any]:
        edges = []
        for edges_item_data in self.edges:
            edges_item = edges_item_data.to_dict()

            edges.append(edges_item)

        family = self.family.value

        nodes = []
        for nodes_item_data in self.nodes:
            nodes_item = nodes_item_data.to_dict()

            nodes.append(nodes_item)

        field_dict: Dict[str, Any] = {}
        field_dict.update(
            {
                "edges": edges,
                "family": family,
                "nodes": nodes,
            }
        )

        field_dict = {k: v for k, v in field_dict.items() if v != UNSET}
        if pick_by_predicate is not None:
            field_dict = {k: v for k, v in field_dict.items() if pick_by_predicate(v)}

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        edges = []
        _edges = d.pop("edges")
        for edges_item_data in _edges:
            edges_item = Edge.from_dict(edges_item_data)

            edges.append(edges_item)

        family = Family(d.pop("family"))

        nodes = []
        _nodes = d.pop("nodes")
        for nodes_item_data in _nodes:
            nodes_item = Node.from_dict(nodes_item_data)

            nodes.append(nodes_item)

        architecture = cls(
            edges=edges,
            family=family,
            nodes=nodes,
        )

        return architecture
