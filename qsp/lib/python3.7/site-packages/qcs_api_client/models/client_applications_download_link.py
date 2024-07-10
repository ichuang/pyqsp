from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

import attr

from ..models.checksum_description import ChecksumDescription
from ..types import UNSET, Unset
from ..util.serialization import is_not_none

T = TypeVar("T", bound="ClientApplicationsDownloadLink")


@attr.s(auto_attribs=True)
class ClientApplicationsDownloadLink:
    """
    Attributes:
        url (str):
        checksum_description (Union[Unset, ChecksumDescription]):
        platform (Union[Unset, str]):
    """

    url: str
    checksum_description: Union[Unset, ChecksumDescription] = UNSET
    platform: Union[Unset, str] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self, pick_by_predicate: Optional[Callable[[Any], bool]] = is_not_none) -> Dict[str, Any]:
        url = self.url
        checksum_description: Union[Unset, Dict[str, Any]] = UNSET
        if not isinstance(self.checksum_description, Unset):
            checksum_description = self.checksum_description.to_dict()

        platform = self.platform

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "url": url,
            }
        )
        if checksum_description is not UNSET:
            field_dict["checksumDescription"] = checksum_description
        if platform is not UNSET:
            field_dict["platform"] = platform

        field_dict = {k: v for k, v in field_dict.items() if v != UNSET}
        if pick_by_predicate is not None:
            field_dict = {k: v for k, v in field_dict.items() if pick_by_predicate(v)}

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        url = d.pop("url")

        _checksum_description = d.pop("checksumDescription", UNSET)
        checksum_description: Union[Unset, ChecksumDescription]
        if isinstance(_checksum_description, Unset):
            checksum_description = UNSET
        else:
            checksum_description = ChecksumDescription.from_dict(_checksum_description)

        platform = d.pop("platform", UNSET)

        client_applications_download_link = cls(
            url=url,
            checksum_description=checksum_description,
            platform=platform,
        )

        client_applications_download_link.additional_properties = d
        return client_applications_download_link

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
