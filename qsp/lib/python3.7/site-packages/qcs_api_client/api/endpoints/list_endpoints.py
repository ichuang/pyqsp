from typing import Any, Dict, Union

import httpx
from retrying import retry

from ...models.list_endpoints_response import ListEndpointsResponse
from ...types import UNSET, Response, Unset
from ...util.errors import QCSHTTPStatusError, raise_for_status
from ...util.retry import DEFAULT_RETRY_ARGUMENTS


def _get_kwargs(
    *,
    client: httpx.Client,
    filter_: Union[Unset, None, str] = UNSET,
    page_size: Union[Unset, None, int] = 10,
    page_token: Union[Unset, None, str] = UNSET,
) -> Dict[str, Any]:
    url = "{}/v1/endpoints".format(client.base_url)

    headers = {k: v for (k, v) in client.headers.items()}
    cookies = {k: v for (k, v) in client.cookies}

    params: Dict[str, Any] = {}
    params["filter"] = filter_

    params["pageSize"] = page_size

    params["pageToken"] = page_token

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    return {
        "method": "get",
        "url": url,
        "headers": headers,
        "cookies": cookies,
        "timeout": client.timeout,
        "params": params,
    }


def _parse_response(*, response: httpx.Response) -> ListEndpointsResponse:
    raise_for_status(response)
    if response.status_code == 200:
        response_200 = ListEndpointsResponse.from_dict(response.json())

        return response_200
    else:
        raise QCSHTTPStatusError(
            f"Unexpected response: status code {response.status_code}", response=response, error=None
        )


def _build_response(*, response: httpx.Response) -> Response[ListEndpointsResponse]:
    """
    Construct the Response class from the raw ``httpx.Response``.
    """
    return Response.build_from_httpx_response(response=response, parse_function=_parse_response)


@retry(**DEFAULT_RETRY_ARGUMENTS)
def sync(
    *,
    client: httpx.Client,
    filter_: Union[Unset, None, str] = UNSET,
    page_size: Union[Unset, None, int] = 10,
    page_token: Union[Unset, None, str] = UNSET,
    httpx_request_kwargs: Dict[str, Any] = {},
) -> Response[ListEndpointsResponse]:
    """List Endpoints

     List all endpoints, optionally filtering by attribute.

    Args:
        filter_ (Union[Unset, None, str]): Filtering logic specified using [rule-
            engine](https://zerosteiner.github.io/rule-engine/syntax.html) grammar
        page_size (Union[Unset, None, int]):  Default: 10.
        page_token (Union[Unset, None, str]):

    Returns:
        Response[ListEndpointsResponse]
    """

    kwargs = _get_kwargs(
        client=client,
        filter_=filter_,
        page_size=page_size,
        page_token=page_token,
    )
    kwargs.update(httpx_request_kwargs)
    response = client.request(
        **kwargs,
    )

    return _build_response(response=response)


@retry(**DEFAULT_RETRY_ARGUMENTS)
def sync_from_dict(
    *,
    client: httpx.Client,
    filter_: Union[Unset, None, str] = UNSET,
    page_size: Union[Unset, None, int] = 10,
    page_token: Union[Unset, None, str] = UNSET,
    httpx_request_kwargs: Dict[str, Any] = {},
) -> Response[ListEndpointsResponse]:

    kwargs = _get_kwargs(
        client=client,
        filter_=filter_,
        page_size=page_size,
        page_token=page_token,
    )
    kwargs.update(httpx_request_kwargs)
    response = client.request(
        **kwargs,
    )

    return _build_response(response=response)


@retry(**DEFAULT_RETRY_ARGUMENTS)
async def asyncio(
    *,
    client: httpx.AsyncClient,
    filter_: Union[Unset, None, str] = UNSET,
    page_size: Union[Unset, None, int] = 10,
    page_token: Union[Unset, None, str] = UNSET,
    httpx_request_kwargs: Dict[str, Any] = {},
) -> Response[ListEndpointsResponse]:
    """List Endpoints

     List all endpoints, optionally filtering by attribute.

    Args:
        filter_ (Union[Unset, None, str]): Filtering logic specified using [rule-
            engine](https://zerosteiner.github.io/rule-engine/syntax.html) grammar
        page_size (Union[Unset, None, int]):  Default: 10.
        page_token (Union[Unset, None, str]):

    Returns:
        Response[ListEndpointsResponse]
    """

    kwargs = _get_kwargs(
        client=client,
        filter_=filter_,
        page_size=page_size,
        page_token=page_token,
    )
    kwargs.update(httpx_request_kwargs)
    response = await client.request(
        **kwargs,
    )

    return _build_response(response=response)


@retry(**DEFAULT_RETRY_ARGUMENTS)
async def asyncio_from_dict(
    *,
    client: httpx.AsyncClient,
    filter_: Union[Unset, None, str] = UNSET,
    page_size: Union[Unset, None, int] = 10,
    page_token: Union[Unset, None, str] = UNSET,
    httpx_request_kwargs: Dict[str, Any] = {},
) -> Response[ListEndpointsResponse]:

    kwargs = _get_kwargs(
        client=client,
        filter_=filter_,
        page_size=page_size,
        page_token=page_token,
    )
    kwargs.update(httpx_request_kwargs)
    response = client.request(
        **kwargs,
    )

    return _build_response(response=response)
