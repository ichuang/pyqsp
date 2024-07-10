from typing import Any, Dict, Union

import httpx
from retrying import retry

from ...models.list_quantum_processors_response import ListQuantumProcessorsResponse
from ...types import UNSET, Response, Unset
from ...util.errors import QCSHTTPStatusError, raise_for_status
from ...util.retry import DEFAULT_RETRY_ARGUMENTS


def _get_kwargs(
    *,
    client: httpx.Client,
    page_size: Union[Unset, None, int] = 10,
    page_token: Union[Unset, None, str] = UNSET,
) -> Dict[str, Any]:
    url = "{}/v1/quantumProcessors".format(client.base_url)

    headers = {k: v for (k, v) in client.headers.items()}
    cookies = {k: v for (k, v) in client.cookies}

    params: Dict[str, Any] = {}
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


def _parse_response(*, response: httpx.Response) -> ListQuantumProcessorsResponse:
    raise_for_status(response)
    if response.status_code == 200:
        response_200 = ListQuantumProcessorsResponse.from_dict(response.json())

        return response_200
    else:
        raise QCSHTTPStatusError(
            f"Unexpected response: status code {response.status_code}", response=response, error=None
        )


def _build_response(*, response: httpx.Response) -> Response[ListQuantumProcessorsResponse]:
    """
    Construct the Response class from the raw ``httpx.Response``.
    """
    return Response.build_from_httpx_response(response=response, parse_function=_parse_response)


@retry(**DEFAULT_RETRY_ARGUMENTS)
def sync(
    *,
    client: httpx.Client,
    page_size: Union[Unset, None, int] = 10,
    page_token: Union[Unset, None, str] = UNSET,
    httpx_request_kwargs: Dict[str, Any] = {},
) -> Response[ListQuantumProcessorsResponse]:
    """List Quantum Processors

     List all QuantumProcessors available to the user.

    Args:
        page_size (Union[Unset, None, int]):  Default: 10.
        page_token (Union[Unset, None, str]):

    Returns:
        Response[ListQuantumProcessorsResponse]
    """

    kwargs = _get_kwargs(
        client=client,
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
    page_size: Union[Unset, None, int] = 10,
    page_token: Union[Unset, None, str] = UNSET,
    httpx_request_kwargs: Dict[str, Any] = {},
) -> Response[ListQuantumProcessorsResponse]:

    kwargs = _get_kwargs(
        client=client,
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
    page_size: Union[Unset, None, int] = 10,
    page_token: Union[Unset, None, str] = UNSET,
    httpx_request_kwargs: Dict[str, Any] = {},
) -> Response[ListQuantumProcessorsResponse]:
    """List Quantum Processors

     List all QuantumProcessors available to the user.

    Args:
        page_size (Union[Unset, None, int]):  Default: 10.
        page_token (Union[Unset, None, str]):

    Returns:
        Response[ListQuantumProcessorsResponse]
    """

    kwargs = _get_kwargs(
        client=client,
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
    page_size: Union[Unset, None, int] = 10,
    page_token: Union[Unset, None, str] = UNSET,
    httpx_request_kwargs: Dict[str, Any] = {},
) -> Response[ListQuantumProcessorsResponse]:

    kwargs = _get_kwargs(
        client=client,
        page_size=page_size,
        page_token=page_token,
    )
    kwargs.update(httpx_request_kwargs)
    response = client.request(
        **kwargs,
    )

    return _build_response(response=response)
