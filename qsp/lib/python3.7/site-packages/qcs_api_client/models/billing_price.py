from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

import attr

from ..models.billing_price_object import BillingPriceObject
from ..models.billing_price_price_type import BillingPricePriceType
from ..models.billing_price_recurrence import BillingPriceRecurrence
from ..models.billing_price_scheme import BillingPriceScheme
from ..models.billing_price_tiers_mode import BillingPriceTiersMode
from ..models.billing_product import BillingProduct
from ..models.tier import Tier
from ..types import UNSET, Unset
from ..util.serialization import is_not_none

T = TypeVar("T", bound="BillingPrice")


@attr.s(auto_attribs=True)
class BillingPrice:
    """The price schedule for a particular service applied to an invoice line item.

    Attributes:
        id (str): Unique identifier for the object.
        active (Union[Unset, bool]): Whether the price can be used for new purchases.
        billing_scheme (Union[Unset, BillingPriceScheme]): Describes how to compute the price per period. Either
            `per_unit` or `tiered`. `per_unit` indicates that the fixed amount (specified in `unitAmount` or
            `unitAmountDecimal`) will be charged per unit in `quantity` (for prices with `usageType=licensed`), or per unit
            of total usage (for prices with `usageType=metered`). `tiered` indicates that the unit pricing will be computed
            using a tiering strategy as defined using the `tiers` and `tiersMode` attributes.
        object_ (Union[Unset, BillingPriceObject]): String representing the object's type. Objects of the same type
            share the same value.
        price_type (Union[Unset, BillingPricePriceType]): One of `one_time` or `recurring` depending on whether the
            price is for a one-time purchase or a recurring (subscription) purchase.
        product (Union[Unset, BillingProduct]): A QCS service product. This may represent one time (such as
            reservations) or metered services.
        recurring (Union[Unset, BillingPriceRecurrence]): The recurring components of a price such as `interval` and
            `usageType`.
        tiers (Union[Unset, List[Tier]]): Each element represents a pricing tier. This parameter requires
            `billingScheme` to be set to `tiered`. See also the documentation for `billingScheme`.
        tiers_mode (Union[Unset, BillingPriceTiersMode]): Defines if the tiering price should be `graduated` or `volume`
            based. In `volume`-based tiering, the maximum quantity within a period determines the per unit price, in
            `graduated` tiering pricing can successively change as the quantity grows.
        unit_amount_decimal (Union[Unset, float]): The unit amount in `currency` to be charged. Only set if
            `billingScheme=per_unit`.
    """

    id: str
    active: Union[Unset, bool] = UNSET
    billing_scheme: Union[Unset, BillingPriceScheme] = UNSET
    object_: Union[Unset, BillingPriceObject] = UNSET
    price_type: Union[Unset, BillingPricePriceType] = UNSET
    product: Union[Unset, BillingProduct] = UNSET
    recurring: Union[Unset, BillingPriceRecurrence] = UNSET
    tiers: Union[Unset, List[Tier]] = UNSET
    tiers_mode: Union[Unset, BillingPriceTiersMode] = UNSET
    unit_amount_decimal: Union[Unset, float] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self, pick_by_predicate: Optional[Callable[[Any], bool]] = is_not_none) -> Dict[str, Any]:
        id = self.id
        active = self.active
        billing_scheme: Union[Unset, str] = UNSET
        if not isinstance(self.billing_scheme, Unset):
            billing_scheme = self.billing_scheme.value

        object_: Union[Unset, str] = UNSET
        if not isinstance(self.object_, Unset):
            object_ = self.object_.value

        price_type: Union[Unset, str] = UNSET
        if not isinstance(self.price_type, Unset):
            price_type = self.price_type.value

        product: Union[Unset, Dict[str, Any]] = UNSET
        if not isinstance(self.product, Unset):
            product = self.product.to_dict()

        recurring: Union[Unset, Dict[str, Any]] = UNSET
        if not isinstance(self.recurring, Unset):
            recurring = self.recurring.to_dict()

        tiers: Union[Unset, List[Dict[str, Any]]] = UNSET
        if not isinstance(self.tiers, Unset):
            tiers = []
            for tiers_item_data in self.tiers:
                tiers_item = tiers_item_data.to_dict()

                tiers.append(tiers_item)

        tiers_mode: Union[Unset, str] = UNSET
        if not isinstance(self.tiers_mode, Unset):
            tiers_mode = self.tiers_mode.value

        unit_amount_decimal = self.unit_amount_decimal

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
            }
        )
        if active is not UNSET:
            field_dict["active"] = active
        if billing_scheme is not UNSET:
            field_dict["billingScheme"] = billing_scheme
        if object_ is not UNSET:
            field_dict["object"] = object_
        if price_type is not UNSET:
            field_dict["priceType"] = price_type
        if product is not UNSET:
            field_dict["product"] = product
        if recurring is not UNSET:
            field_dict["recurring"] = recurring
        if tiers is not UNSET:
            field_dict["tiers"] = tiers
        if tiers_mode is not UNSET:
            field_dict["tiersMode"] = tiers_mode
        if unit_amount_decimal is not UNSET:
            field_dict["unitAmountDecimal"] = unit_amount_decimal

        field_dict = {k: v for k, v in field_dict.items() if v != UNSET}
        if pick_by_predicate is not None:
            field_dict = {k: v for k, v in field_dict.items() if pick_by_predicate(v)}

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        id = d.pop("id")

        active = d.pop("active", UNSET)

        _billing_scheme = d.pop("billingScheme", UNSET)
        billing_scheme: Union[Unset, BillingPriceScheme]
        if isinstance(_billing_scheme, Unset):
            billing_scheme = UNSET
        else:
            billing_scheme = BillingPriceScheme(_billing_scheme)

        _object_ = d.pop("object", UNSET)
        object_: Union[Unset, BillingPriceObject]
        if isinstance(_object_, Unset):
            object_ = UNSET
        else:
            object_ = BillingPriceObject(_object_)

        _price_type = d.pop("priceType", UNSET)
        price_type: Union[Unset, BillingPricePriceType]
        if isinstance(_price_type, Unset):
            price_type = UNSET
        else:
            price_type = BillingPricePriceType(_price_type)

        _product = d.pop("product", UNSET)
        product: Union[Unset, BillingProduct]
        if isinstance(_product, Unset):
            product = UNSET
        else:
            product = BillingProduct.from_dict(_product)

        _recurring = d.pop("recurring", UNSET)
        recurring: Union[Unset, BillingPriceRecurrence]
        if isinstance(_recurring, Unset):
            recurring = UNSET
        else:
            recurring = BillingPriceRecurrence.from_dict(_recurring)

        tiers = []
        _tiers = d.pop("tiers", UNSET)
        for tiers_item_data in _tiers or []:
            tiers_item = Tier.from_dict(tiers_item_data)

            tiers.append(tiers_item)

        _tiers_mode = d.pop("tiersMode", UNSET)
        tiers_mode: Union[Unset, BillingPriceTiersMode]
        if isinstance(_tiers_mode, Unset):
            tiers_mode = UNSET
        else:
            tiers_mode = BillingPriceTiersMode(_tiers_mode)

        unit_amount_decimal = d.pop("unitAmountDecimal", UNSET)

        billing_price = cls(
            id=id,
            active=active,
            billing_scheme=billing_scheme,
            object_=object_,
            price_type=price_type,
            product=product,
            recurring=recurring,
            tiers=tiers,
            tiers_mode=tiers_mode,
            unit_amount_decimal=unit_amount_decimal,
        )

        billing_price.additional_properties = d
        return billing_price

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
