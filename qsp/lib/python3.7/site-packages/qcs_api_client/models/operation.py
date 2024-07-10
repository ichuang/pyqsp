from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

import attr

from ..models.characteristic import Characteristic
from ..models.operation_site import OperationSite
from ..models.parameter import Parameter
from ..types import UNSET, Unset
from ..util.serialization import is_not_none

T = TypeVar("T", bound="Operation")


@attr.s(auto_attribs=True)
class Operation:
    """An operation, with its sites and site-independent characteristics.

    Attributes:
        characteristics (List[Characteristic]): The list of site-independent characteristics of this operation.
        name (str): The name of the operation.
        parameters (List[Parameter]): The list of parameters. Each parameter must be uniquely named. May be empty.
        sites (List[OperationSite]): The list of sites at which this operation can be applied, together with its site-
            dependent characteristics.
        node_count (Union[Unset, int]): The number of nodes that this operation applies to. None if unspecified.
    """

    characteristics: List[Characteristic]
    name: str
    parameters: List[Parameter]
    sites: List[OperationSite]
    node_count: Union[Unset, int] = UNSET

    def to_dict(self, pick_by_predicate: Optional[Callable[[Any], bool]] = is_not_none) -> Dict[str, Any]:
        characteristics = []
        for characteristics_item_data in self.characteristics:
            characteristics_item = characteristics_item_data.to_dict()

            characteristics.append(characteristics_item)

        name = self.name
        parameters = []
        for parameters_item_data in self.parameters:
            parameters_item = parameters_item_data.to_dict()

            parameters.append(parameters_item)

        sites = []
        for sites_item_data in self.sites:
            sites_item = sites_item_data.to_dict()

            sites.append(sites_item)

        node_count = self.node_count

        field_dict: Dict[str, Any] = {}
        field_dict.update(
            {
                "characteristics": characteristics,
                "name": name,
                "parameters": parameters,
                "sites": sites,
            }
        )
        if node_count is not UNSET:
            field_dict["node_count"] = node_count

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

        name = d.pop("name")

        parameters = []
        _parameters = d.pop("parameters")
        for parameters_item_data in _parameters:
            parameters_item = Parameter.from_dict(parameters_item_data)

            parameters.append(parameters_item)

        sites = []
        _sites = d.pop("sites")
        for sites_item_data in _sites:
            sites_item = OperationSite.from_dict(sites_item_data)

            sites.append(sites_item)

        node_count = d.pop("node_count", UNSET)

        operation = cls(
            characteristics=characteristics,
            name=name,
            parameters=parameters,
            sites=sites,
            node_count=node_count,
        )

        return operation
