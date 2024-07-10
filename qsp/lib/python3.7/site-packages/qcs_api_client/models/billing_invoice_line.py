from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

import attr

from ..models.billing_invoice_line_line_item_type import BillingInvoiceLineLineItemType
from ..models.billing_invoice_line_metadata import BillingInvoiceLineMetadata
from ..models.billing_price import BillingPrice
from ..types import UNSET, Unset
from ..util.serialization import is_not_none

T = TypeVar("T", bound="BillingInvoiceLine")


@attr.s(auto_attribs=True)
class BillingInvoiceLine:
    """A line item on an invoice representing a particular purchase (such as a reservation) or aggregate usage for the
    invoice period.

        Attributes:
            amount (int):
            description (str):
            id (str):
            line_item_type (BillingInvoiceLineLineItemType):
            metadata (BillingInvoiceLineMetadata):
            quantity (int):
            invoice_item (Union[Unset, str]):
            price (Union[Unset, BillingPrice]): The price schedule for a particular service applied to an invoice line item.
            subscription (Union[Unset, str]):
            subscription_item (Union[Unset, str]):
    """

    amount: int
    description: str
    id: str
    line_item_type: BillingInvoiceLineLineItemType
    metadata: BillingInvoiceLineMetadata
    quantity: int
    invoice_item: Union[Unset, str] = UNSET
    price: Union[Unset, BillingPrice] = UNSET
    subscription: Union[Unset, str] = UNSET
    subscription_item: Union[Unset, str] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self, pick_by_predicate: Optional[Callable[[Any], bool]] = is_not_none) -> Dict[str, Any]:
        amount = self.amount
        description = self.description
        id = self.id
        line_item_type = self.line_item_type.value

        metadata = self.metadata.to_dict()

        quantity = self.quantity
        invoice_item = self.invoice_item
        price: Union[Unset, Dict[str, Any]] = UNSET
        if not isinstance(self.price, Unset):
            price = self.price.to_dict()

        subscription = self.subscription
        subscription_item = self.subscription_item

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "amount": amount,
                "description": description,
                "id": id,
                "lineItemType": line_item_type,
                "metadata": metadata,
                "quantity": quantity,
            }
        )
        if invoice_item is not UNSET:
            field_dict["invoiceItem"] = invoice_item
        if price is not UNSET:
            field_dict["price"] = price
        if subscription is not UNSET:
            field_dict["subscription"] = subscription
        if subscription_item is not UNSET:
            field_dict["subscriptionItem"] = subscription_item

        field_dict = {k: v for k, v in field_dict.items() if v != UNSET}
        if pick_by_predicate is not None:
            field_dict = {k: v for k, v in field_dict.items() if pick_by_predicate(v)}

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        amount = d.pop("amount")

        description = d.pop("description")

        id = d.pop("id")

        line_item_type = BillingInvoiceLineLineItemType(d.pop("lineItemType"))

        metadata = BillingInvoiceLineMetadata.from_dict(d.pop("metadata"))

        quantity = d.pop("quantity")

        invoice_item = d.pop("invoiceItem", UNSET)

        _price = d.pop("price", UNSET)
        price: Union[Unset, BillingPrice]
        if isinstance(_price, Unset):
            price = UNSET
        else:
            price = BillingPrice.from_dict(_price)

        subscription = d.pop("subscription", UNSET)

        subscription_item = d.pop("subscriptionItem", UNSET)

        billing_invoice_line = cls(
            amount=amount,
            description=description,
            id=id,
            line_item_type=line_item_type,
            metadata=metadata,
            quantity=quantity,
            invoice_item=invoice_item,
            price=price,
            subscription=subscription,
            subscription_item=subscription_item,
        )

        billing_invoice_line.additional_properties = d
        return billing_invoice_line

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
