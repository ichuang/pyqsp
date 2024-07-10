from typing import Any, Dict, Union

import httpx
from retrying import retry

from ...models.list_quantum_processor_accessors_response import ListQuantumProcessorAccessorsResponse
from ...types import UNSET, Response, Unset
from ...util.errors import QCSHTTPStatusError, raise_for_status
from ...util.retry import DEFAULT_RETRY_ARGUMENTS


def _get_kwargs(
    quantum_processor_id: str,
    *,
    client: httpx.Client,
    page_size: Union[Unset, None, int] = 10,
    page_token: Union[Unset, None, str] = UNSET,
) -> Dict[str, Any]:
    url = "{}/v1/quantumProcessors/{quantumProcessorId}/accessors".format(
        client.base_url, quantumProcessorId=quantum_processor_id
    )

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


def _parse_response(*, response: httpx.Response) -> ListQuantumProcessorAccessorsResponse:
    raise_for_status(response)
    if response.status_code == 200:
        response_200 = ListQuantumProcessorAccessorsResponse.from_dict(response.json())

        return response_200
    else:
        raise QCSHTTPStatusError(
            f"Unexpected response: status code {response.status_code}", response=response, error=None
        )


def _build_response(*, response: httpx.Response) -> Response[ListQuantumProcessorAccessorsResponse]:
    """
    Construct the Response class from the raw ``httpx.Response``.
    """
    return Response.build_from_httpx_response(response=response, parse_function=_parse_response)


@retry(**DEFAULT_RETRY_ARGUMENTS)
def sync(
    quantum_processor_id: str,
    *,
    client: httpx.Client,
    page_size: Union[Unset, None, int] = 10,
    page_token: Union[Unset, None, str] = UNSET,
    httpx_request_kwargs: Dict[str, Any] = {},
) -> Response[ListQuantumProcessorAccessorsResponse]:
    """List Quantum Processor Accessors

     List all means of accessing a QuantumProcessor available to the user.

    Args:
        quantum_processor_id (str): Public identifier for a quantum processor [example: Aspen-1]
        page_size (Union[Unset, None, int]):  Default: 10.
        page_token (Union[Unset, None, str]):

    Returns:
        Response[ListQuantumProcessorAccessorsResponse]
    """

    kwargs = _get_kwargs(
        quantum_processor_id=quantum_processor_id,
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
    quantum_processor_id: str,
    *,
    client: httpx.Client,
    page_size: Union[Unset, None, int] = 10,
    page_token: Union[Unset, None, str] = UNSET,
    httpx_request_kwargs: Dict[str, Any] = {},
) -> Response[ListQuantumProcessorAccessorsResponse]:

    kwargs = _get_kwargs(
        quantum_processor_id=quantum_processor_id,
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
    quantum_processor_id: str,
    *,
    client: httpx.AsyncClient,
    page_size: Union[Unset, None, int] = 10,
    page_token: Union[Unset, None, str] = UNSET,
    httpx_request_kwargs: Dict[str, Any] = {},
) -> Response[ListQuantumProcessorAccessorsResponse]:
    """List Quantum Processor Accessors

     List all means of accessing a QuantumProcessor available to the user.

    Args:
        quantum_processor_id (str): Public identifier for a quantum processor [example: Aspen-1]
        page_size (Union[Unset, None, int]):  Default: 10.
        page_token (Union[Unset, None, str]):

    Returns:
        Response[ListQuantumProcessorAccessorsResponse]
    """

    kwargs = _get_kwargs(
        quantum_processor_id=quantum_processor_id,
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
    quantum_processor_id: str,
    *,
    client: httpx.AsyncClient,
    page_size: Union[Unset, None, int] = 10,
    page_token: Union[Unset, None, str] = UNSET,
    httpx_request_kwargs: Dict[str, Any] = {},
) -> Response[ListQuantumProcessorAccessorsResponse]:

    kwargs = _get_kwargs(
        quantum_processor_id=quantum_processor_id,
        client=client,
        page_size=page_size,
        page_token=page_token,
    )
    kwargs.update(httpx_request_kwargs)
    response = client.request(
        **kwargs,
    )

    return _build_response(response=response)
