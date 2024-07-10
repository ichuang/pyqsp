from typing import Any, Dict

import httpx
from retrying import retry

from ...models.billing_upcoming_invoice import BillingUpcomingInvoice
from ...types import Response
from ...util.errors import QCSHTTPStatusError, raise_for_status
from ...util.retry import DEFAULT_RETRY_ARGUMENTS


def _get_kwargs(
    user_id: str,
    *,
    client: httpx.Client,
) -> Dict[str, Any]:
    url = "{}/v1/users/{userId}/billingInvoices:getUpcoming".format(client.base_url, userId=user_id)

    headers = {k: v for (k, v) in client.headers.items()}
    cookies = {k: v for (k, v) in client.cookies}

    return {
        "method": "get",
        "url": url,
        "headers": headers,
        "cookies": cookies,
        "timeout": client.timeout,
    }


def _parse_response(*, response: httpx.Response) -> BillingUpcomingInvoice:
    raise_for_status(response)
    if response.status_code == 200:
        response_200 = BillingUpcomingInvoice.from_dict(response.json())

        return response_200
    else:
        raise QCSHTTPStatusError(
            f"Unexpected response: status code {response.status_code}", response=response, error=None
        )


def _build_response(*, response: httpx.Response) -> Response[BillingUpcomingInvoice]:
    """
    Construct the Response class from the raw ``httpx.Response``.
    """
    return Response.build_from_httpx_response(response=response, parse_function=_parse_response)


@retry(**DEFAULT_RETRY_ARGUMENTS)
def sync(
    user_id: str,
    *,
    client: httpx.Client,
    httpx_request_kwargs: Dict[str, Any] = {},
) -> Response[BillingUpcomingInvoice]:
    """Retrieve upcoming invoice for QCS user billing customer.

    Args:
        user_id (str):

    Returns:
        Response[BillingUpcomingInvoice]
    """

    kwargs = _get_kwargs(
        user_id=user_id,
        client=client,
    )
    kwargs.update(httpx_request_kwargs)
    response = client.request(
        **kwargs,
    )

    return _build_response(response=response)


@retry(**DEFAULT_RETRY_ARGUMENTS)
def sync_from_dict(
    user_id: str,
    *,
    client: httpx.Client,
    httpx_request_kwargs: Dict[str, Any] = {},
) -> Response[BillingUpcomingInvoice]:

    kwargs = _get_kwargs(
        user_id=user_id,
        client=client,
    )
    kwargs.update(httpx_request_kwargs)
    response = client.request(
        **kwargs,
    )

    return _build_response(response=response)


@retry(**DEFAULT_RETRY_ARGUMENTS)
async def asyncio(
    user_id: str,
    *,
    client: httpx.AsyncClient,
    httpx_request_kwargs: Dict[str, Any] = {},
) -> Response[BillingUpcomingInvoice]:
    """Retrieve upcoming invoice for QCS user billing customer.

    Args:
        user_id (str):

    Returns:
        Response[BillingUpcomingInvoice]
    """

    kwargs = _get_kwargs(
        user_id=user_id,
        client=client,
    )
    kwargs.update(httpx_request_kwargs)
    response = await client.request(
        **kwargs,
    )

    return _build_response(response=response)


@retry(**DEFAULT_RETRY_ARGUMENTS)
async def asyncio_from_dict(
    user_id: str,
    *,
    client: httpx.AsyncClient,
    httpx_request_kwargs: Dict[str, Any] = {},
) -> Response[BillingUpcomingInvoice]:

    kwargs = _get_kwargs(
        user_id=user_id,
        client=client,
    )
    kwargs.update(httpx_request_kwargs)
    response = client.request(
        **kwargs,
    )

    return _build_response(response=response)
