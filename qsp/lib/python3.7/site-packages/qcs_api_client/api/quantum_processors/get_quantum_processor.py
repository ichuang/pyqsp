from typing import Any, Dict

import httpx
from retrying import retry

from ...models.quantum_processor import QuantumProcessor
from ...types import Response
from ...util.errors import QCSHTTPStatusError, raise_for_status
from ...util.retry import DEFAULT_RETRY_ARGUMENTS


def _get_kwargs(
    quantum_processor_id: str,
    *,
    client: httpx.Client,
) -> Dict[str, Any]:
    url = "{}/v1/quantumProcessors/{quantumProcessorId}".format(
        client.base_url, quantumProcessorId=quantum_processor_id
    )

    headers = {k: v for (k, v) in client.headers.items()}
    cookies = {k: v for (k, v) in client.cookies}

    return {
        "method": "get",
        "url": url,
        "headers": headers,
        "cookies": cookies,
        "timeout": client.timeout,
    }


def _parse_response(*, response: httpx.Response) -> QuantumProcessor:
    raise_for_status(response)
    if response.status_code == 200:
        response_200 = QuantumProcessor.from_dict(response.json())

        return response_200
    else:
        raise QCSHTTPStatusError(
            f"Unexpected response: status code {response.status_code}", response=response, error=None
        )


def _build_response(*, response: httpx.Response) -> Response[QuantumProcessor]:
    """
    Construct the Response class from the raw ``httpx.Response``.
    """
    return Response.build_from_httpx_response(response=response, parse_function=_parse_response)


@retry(**DEFAULT_RETRY_ARGUMENTS)
def sync(
    quantum_processor_id: str,
    *,
    client: httpx.Client,
    httpx_request_kwargs: Dict[str, Any] = {},
) -> Response[QuantumProcessor]:
    """Get Quantum Processor

     Retrieve a single QuantumProcessor by ID.

    Args:
        quantum_processor_id (str): Public identifier for a quantum processor [example: Aspen-1]

    Returns:
        Response[QuantumProcessor]
    """

    kwargs = _get_kwargs(
        quantum_processor_id=quantum_processor_id,
        client=client,
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
    httpx_request_kwargs: Dict[str, Any] = {},
) -> Response[QuantumProcessor]:

    kwargs = _get_kwargs(
        quantum_processor_id=quantum_processor_id,
        client=client,
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
    httpx_request_kwargs: Dict[str, Any] = {},
) -> Response[QuantumProcessor]:
    """Get Quantum Processor

     Retrieve a single QuantumProcessor by ID.

    Args:
        quantum_processor_id (str): Public identifier for a quantum processor [example: Aspen-1]

    Returns:
        Response[QuantumProcessor]
    """

    kwargs = _get_kwargs(
        quantum_processor_id=quantum_processor_id,
        client=client,
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
    httpx_request_kwargs: Dict[str, Any] = {},
) -> Response[QuantumProcessor]:

    kwargs = _get_kwargs(
        quantum_processor_id=quantum_processor_id,
        client=client,
    )
    kwargs.update(httpx_request_kwargs)
    response = client.request(
        **kwargs,
    )

    return _build_response(response=response)
