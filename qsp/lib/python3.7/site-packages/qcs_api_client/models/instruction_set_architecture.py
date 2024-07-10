from typing import Any, Callable, Dict, List, Optional, Type, TypeVar

import attr

from ..models.architecture import Architecture
from ..models.operation import Operation
from ..types import UNSET
from ..util.serialization import is_not_none

T = TypeVar("T", bound="InstructionSetArchitecture")


@attr.s(auto_attribs=True)
class InstructionSetArchitecture:
    """The native instruction set architecture of a quantum processor, annotated with characteristics.

    The operations described by the `instructions` field are named by their QUIL instruction name,
    while the operation described by the `benchmarks` field are named by their benchmark routine
    and are a future extension point that will be populated in future iterations.

    The characteristics that annotate both instructions and benchmarks assist the user to generate
    the best native QUIL program for a desired task, and so are provided as part of the native ISA.

        Attributes:
            architecture (Architecture): Represents the logical underlying architecture of a quantum processor.

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
            benchmarks (List[Operation]): The list of benchmarks that have characterized the quantum processor.
            instructions (List[Operation]): The list of native QUIL instructions supported by the quantum processor.
            name (str): The name of the quantum processor.
    """

    architecture: Architecture
    benchmarks: List[Operation]
    instructions: List[Operation]
    name: str

    def to_dict(self, pick_by_predicate: Optional[Callable[[Any], bool]] = is_not_none) -> Dict[str, Any]:
        architecture = self.architecture.to_dict()

        benchmarks = []
        for benchmarks_item_data in self.benchmarks:
            benchmarks_item = benchmarks_item_data.to_dict()

            benchmarks.append(benchmarks_item)

        instructions = []
        for instructions_item_data in self.instructions:
            instructions_item = instructions_item_data.to_dict()

            instructions.append(instructions_item)

        name = self.name

        field_dict: Dict[str, Any] = {}
        field_dict.update(
            {
                "architecture": architecture,
                "benchmarks": benchmarks,
                "instructions": instructions,
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
        architecture = Architecture.from_dict(d.pop("architecture"))

        benchmarks = []
        _benchmarks = d.pop("benchmarks")
        for benchmarks_item_data in _benchmarks:
            benchmarks_item = Operation.from_dict(benchmarks_item_data)

            benchmarks.append(benchmarks_item)

        instructions = []
        _instructions = d.pop("instructions")
        for instructions_item_data in _instructions:
            instructions_item = Operation.from_dict(instructions_item_data)

            instructions.append(instructions_item)

        name = d.pop("name")

        instruction_set_architecture = cls(
            architecture=architecture,
            benchmarks=benchmarks,
            instructions=instructions,
            name=name,
        )

        return instruction_set_architecture
