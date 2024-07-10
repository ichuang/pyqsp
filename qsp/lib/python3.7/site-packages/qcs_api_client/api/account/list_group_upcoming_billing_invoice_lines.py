from typing import Any, Dict

import httpx
from retrying import retry

from ...models.list_account_billing_invoice_lines_response import ListAccountBillingInvoiceLinesResponse
from ...types import Response
from ...util.errors import QCSHTTPStatusError, raise_for_status
from ...util.retry import DEFAULT_RETRY_ARGUMENTS


def _get_kwargs(
    group_name: str,
    *,
    client: httpx.Client,
) -> Dict[str, Any]:
    url = "{}/v1/groups/{groupName}/billingInvoices:listUpcomingLines".format(client.base_url, groupName=group_name)

    headers = {k: v for (k, v) in client.headers.items()}
    cookies = {k: v for (k, v) in client.cookies}

    return {
        "method": "get",
        "url": url,
        "headers": headers,
        "cookies": cookies,
        "timeout": client.timeout,
    }


def _parse_response(*, response: httpx.Response) -> ListAccountBillingInvoiceLinesResponse:
    raise_for_status(response)
    if response.status_code == 200:
        response_200 = ListAccountBillingInvoiceLinesResponse.from_dict(response.json())

        return response_200
    else:
        raise QCSHTTPStatusError(
            f"Unexpected response: status code {response.status_code}", response=response, error=None
        )


def _build_response(*, response: httpx.Response) -> Response[ListAccountBillingInvoiceLinesResponse]:
    """
    Construct the Response class from the raw ``httpx.Response``.
    """
    return Response.build_from_httpx_response(response=response, parse_function=_parse_response)


@retry(**DEFAULT_RETRY_ARGUMENTS)
def sync(
    group_name: str,
    *,
    client: httpx.Client,
    httpx_request_kwargs: Dict[str, Any] = {},
) -> Response[ListAccountBillingInvoiceLinesResponse]:
    """List invoice lines for QCS group billing customer upcoming invoice.

    Args:
        group_name (str):

    Returns:
        Response[ListAccountBillingInvoiceLinesResponse]
    """

    kwargs = _get_kwargs(
        group_name=group_name,
        client=client,
    )
    kwargs.update(httpx_request_kwargs)
    response = client.request(
        **kwargs,
    )

    return _build_response(response=response)


@retry(**DEFAULT_RETRY_ARGUMENTS)
def sync_from_dict(
    group_name: str,
    *,
    client: httpx.Client,
    httpx_request_kwargs: Dict[str, Any] = {},
) -> Response[ListAccountBillingInvoiceLinesResponse]:

    kwargs = _get_kwargs(
        group_name=group_name,
        client=client,
    )
    kwargs.update(httpx_request_kwargs)
    response = client.request(
        **kwargs,
    )

    return _build_response(response=response)


@retry(**DEFAULT_RETRY_ARGUMENTS)
async def asyncio(
    group_name: str,
    *,
    client: httpx.AsyncClient,
    httpx_request_kwargs: Dict[str, Any] = {},
) -> Response[ListAccountBillingInvoiceLinesResponse]:
    """List invoice lines for QCS group billing customer upcoming invoice.

    Args:
        group_name (str):

    Returns:
        Response[ListAccountBillingInvoiceLinesResponse]
    """

    kwargs = _get_kwargs(
        group_name=group_name,
        client=client,
    )
    kwargs.update(httpx_request_kwargs)
    response = await client.request(
        **kwargs,
    )

    return _build_response(response=response)


@retry(**DEFAULT_RETRY_ARGUMENTS)
async def asyncio_from_dict(
    group_name: str,
    *,
    client: httpx.AsyncClient,
    httpx_request_kwargs: Dict[str, Any] = {},
) -> Response[ListAccountBillingInvoiceLinesResponse]:

    kwargs = _get_kwargs(
        group_name=group_name,
        client=client,
    )
    kwargs.update(httpx_request_kwargs)
    response = client.request(
        **kwargs,
    )

    return _build_response(response=response)
