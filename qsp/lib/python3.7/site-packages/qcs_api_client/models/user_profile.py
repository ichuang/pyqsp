from typing import Any, Callable, Dict, List, Optional, Type, TypeVar

import attr

from ..types import UNSET
from ..util.serialization import is_not_none

T = TypeVar("T", bound="UserProfile")


@attr.s(auto_attribs=True)
class UserProfile:
    """
    Attributes:
        email (str):
        first_name (str):
        last_name (str):
        organization (str):
    """

    email: str
    first_name: str
    last_name: str
    organization: str
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self, pick_by_predicate: Optional[Callable[[Any], bool]] = is_not_none) -> Dict[str, Any]:
        email = self.email
        first_name = self.first_name
        last_name = self.last_name
        organization = self.organization

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "email": email,
                "firstName": first_name,
                "lastName": last_name,
                "organization": organization,
            }
        )

        field_dict = {k: v for k, v in field_dict.items() if v != UNSET}
        if pick_by_predicate is not None:
            field_dict = {k: v for k, v in field_dict.items() if pick_by_predicate(v)}

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        email = d.pop("email")

        first_name = d.pop("firstName")

        last_name = d.pop("lastName")

        organization = d.pop("organization")

        user_profile = cls(
            email=email,
            first_name=first_name,
            last_name=last_name,
            organization=organization,
        )

        user_profile.additional_properties = d
        return user_profile

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
