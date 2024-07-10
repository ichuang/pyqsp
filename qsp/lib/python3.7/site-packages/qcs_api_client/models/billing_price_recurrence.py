from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

import attr

from ..models.billing_price_recurrence_aggregate_usage import BillingPriceRecurrenceAggregateUsage
from ..models.billing_price_recurrence_interval import BillingPriceRecurrenceInterval
from ..models.billing_price_recurrence_usage_type import BillingPriceRecurrenceUsageType
from ..types import UNSET, Unset
from ..util.serialization import is_not_none

T = TypeVar("T", bound="BillingPriceRecurrence")


@attr.s(auto_attribs=True)
class BillingPriceRecurrence:
    """The recurring components of a price such as `interval` and `usageType`.

    Attributes:
        interval (BillingPriceRecurrenceInterval):
        aggregate_usage (Union[Unset, BillingPriceRecurrenceAggregateUsage]):
        interval_count (Union[Unset, int]):
        usage_type (Union[Unset, BillingPriceRecurrenceUsageType]):
    """

    interval: BillingPriceRecurrenceInterval
    aggregate_usage: Union[Unset, BillingPriceRecurrenceAggregateUsage] = UNSET
    interval_count: Union[Unset, int] = UNSET
    usage_type: Union[Unset, BillingPriceRecurrenceUsageType] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self, pick_by_predicate: Optional[Callable[[Any], bool]] = is_not_none) -> Dict[str, Any]:
        interval = self.interval.value

        aggregate_usage: Union[Unset, str] = UNSET
        if not isinstance(self.aggregate_usage, Unset):
            aggregate_usage = self.aggregate_usage.value

        interval_count = self.interval_count
        usage_type: Union[Unset, str] = UNSET
        if not isinstance(self.usage_type, Unset):
            usage_type = self.usage_type.value

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "interval": interval,
            }
        )
        if aggregate_usage is not UNSET:
            field_dict["aggregateUsage"] = aggregate_usage
        if interval_count is not UNSET:
            field_dict["intervalCount"] = interval_count
        if usage_type is not UNSET:
            field_dict["usageType"] = usage_type

        field_dict = {k: v for k, v in field_dict.items() if v != UNSET}
        if pick_by_predicate is not None:
            field_dict = {k: v for k, v in field_dict.items() if pick_by_predicate(v)}

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        interval = BillingPriceRecurrenceInterval(d.pop("interval"))

        _aggregate_usage = d.pop("aggregateUsage", UNSET)
        aggregate_usage: Union[Unset, BillingPriceRecurrenceAggregateUsage]
        if isinstance(_aggregate_usage, Unset):
            aggregate_usage = UNSET
        else:
            aggregate_usage = BillingPriceRecurrenceAggregateUsage(_aggregate_usage)

        interval_count = d.pop("intervalCount", UNSET)

        _usage_type = d.pop("usageType", UNSET)
        usage_type: Union[Unset, BillingPriceRecurrenceUsageType]
        if isinstance(_usage_type, Unset):
            usage_type = UNSET
        else:
            usage_type = BillingPriceRecurrenceUsageType(_usage_type)

        billing_price_recurrence = cls(
            interval=interval,
            aggregate_usage=aggregate_usage,
            interval_count=interval_count,
            usage_type=usage_type,
        )

        billing_price_recurrence.additional_properties = d
        return billing_price_recurrence

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
