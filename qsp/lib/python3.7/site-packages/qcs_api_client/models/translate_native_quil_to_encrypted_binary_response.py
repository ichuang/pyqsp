from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union, cast

import attr

from ..models.translate_native_quil_to_encrypted_binary_response_memory_descriptors import (
    TranslateNativeQuilToEncryptedBinaryResponseMemoryDescriptors,
)
from ..types import UNSET, Unset
from ..util.serialization import is_not_none

T = TypeVar("T", bound="TranslateNativeQuilToEncryptedBinaryResponse")


@attr.s(auto_attribs=True)
class TranslateNativeQuilToEncryptedBinaryResponse:
    """
    Attributes:
        program (str): Encrypted binary built for execution on the target quantum processor
        memory_descriptors (Union[Unset, TranslateNativeQuilToEncryptedBinaryResponseMemoryDescriptors]):
        ro_sources (Union[Unset, List[List[str]]]):
        settings_timestamp (Union[Unset, str]): ISO8601 timestamp of the settings used to translate the program.
            Translation is deterministic; a program translated twice with the same settings by the same version of the
            service will have identical output.
    """

    program: str
    memory_descriptors: Union[Unset, TranslateNativeQuilToEncryptedBinaryResponseMemoryDescriptors] = UNSET
    ro_sources: Union[Unset, List[List[str]]] = UNSET
    settings_timestamp: Union[Unset, str] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self, pick_by_predicate: Optional[Callable[[Any], bool]] = is_not_none) -> Dict[str, Any]:
        program = self.program
        memory_descriptors: Union[Unset, Dict[str, Any]] = UNSET
        if not isinstance(self.memory_descriptors, Unset):
            memory_descriptors = self.memory_descriptors.to_dict()

        ro_sources: Union[Unset, List[List[str]]] = UNSET
        if not isinstance(self.ro_sources, Unset):
            ro_sources = []
            for ro_sources_item_data in self.ro_sources:
                ro_sources_item = ro_sources_item_data

                ro_sources.append(ro_sources_item)

        settings_timestamp = self.settings_timestamp

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "program": program,
            }
        )
        if memory_descriptors is not UNSET:
            field_dict["memoryDescriptors"] = memory_descriptors
        if ro_sources is not UNSET:
            field_dict["roSources"] = ro_sources
        if settings_timestamp is not UNSET:
            field_dict["settingsTimestamp"] = settings_timestamp

        field_dict = {k: v for k, v in field_dict.items() if v != UNSET}
        if pick_by_predicate is not None:
            field_dict = {k: v for k, v in field_dict.items() if pick_by_predicate(v)}

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        program = d.pop("program")

        _memory_descriptors = d.pop("memoryDescriptors", UNSET)
        memory_descriptors: Union[Unset, TranslateNativeQuilToEncryptedBinaryResponseMemoryDescriptors]
        if isinstance(_memory_descriptors, Unset):
            memory_descriptors = UNSET
        else:
            memory_descriptors = TranslateNativeQuilToEncryptedBinaryResponseMemoryDescriptors.from_dict(
                _memory_descriptors
            )

        ro_sources = []
        _ro_sources = d.pop("roSources", UNSET)
        for ro_sources_item_data in _ro_sources or []:
            ro_sources_item = cast(List[str], ro_sources_item_data)

            ro_sources.append(ro_sources_item)

        settings_timestamp = d.pop("settingsTimestamp", UNSET)

        translate_native_quil_to_encrypted_binary_response = cls(
            program=program,
            memory_descriptors=memory_descriptors,
            ro_sources=ro_sources,
            settings_timestamp=settings_timestamp,
        )

        translate_native_quil_to_encrypted_binary_response.additional_properties = d
        return translate_native_quil_to_encrypted_binary_response

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
