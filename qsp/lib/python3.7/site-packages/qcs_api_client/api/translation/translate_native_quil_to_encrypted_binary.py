from typing import Any, Dict

import httpx
from retrying import retry

from ...models.translate_native_quil_to_encrypted_binary_request import TranslateNativeQuilToEncryptedBinaryRequest
from ...models.translate_native_quil_to_encrypted_binary_response import TranslateNativeQuilToEncryptedBinaryResponse
from ...types import Response
from ...util.errors import QCSHTTPStatusError, raise_for_status
from ...util.retry import DEFAULT_RETRY_ARGUMENTS


def _get_kwargs(
    quantum_processor_id: str,
    *,
    client: httpx.Client,
    json_body: TranslateNativeQuilToEncryptedBinaryRequest,
) -> Dict[str, Any]:
    url = "{}/v1/quantumProcessors/{quantumProcessorId}:translateNativeQuilToEncryptedBinary".format(
        client.base_url, quantumProcessorId=quantum_processor_id
    )

    headers = {k: v for (k, v) in client.headers.items()}
    cookies = {k: v for (k, v) in client.cookies}

    json_json_body = json_body.to_dict()

    return {
        "method": "post",
        "url": url,
        "headers": headers,
        "cookies": cookies,
        "timeout": client.timeout,
        "json": json_json_body,
    }


def _parse_response(*, response: httpx.Response) -> TranslateNativeQuilToEncryptedBinaryResponse:
    raise_for_status(response)
    if response.status_code == 200:
        response_200 = TranslateNativeQuilToEncryptedBinaryResponse.from_dict(response.json())

        return response_200
    else:
        raise QCSHTTPStatusError(
            f"Unexpected response: status code {response.status_code}", response=response, error=None
        )


def _build_response(*, response: httpx.Response) -> Response[TranslateNativeQuilToEncryptedBinaryResponse]:
    """
    Construct the Response class from the raw ``httpx.Response``.
    """
    return Response.build_from_httpx_response(response=response, parse_function=_parse_response)


@retry(**DEFAULT_RETRY_ARGUMENTS)
def sync(
    quantum_processor_id: str,
    *,
    client: httpx.Client,
    json_body: TranslateNativeQuilToEncryptedBinaryRequest,
    httpx_request_kwargs: Dict[str, Any] = {},
) -> Response[TranslateNativeQuilToEncryptedBinaryResponse]:
    """Translate Native Quil To Encrypted Binary

     Compile Rigetti-native Quil code to encrypted binary form, ready for execution on a
    Rigetti Quantum Processor.

    Args:
        quantum_processor_id (str): Public identifier for a quantum processor [example: Aspen-1]
        json_body (TranslateNativeQuilToEncryptedBinaryRequest):

    Returns:
        Response[TranslateNativeQuilToEncryptedBinaryResponse]
    """

    kwargs = _get_kwargs(
        quantum_processor_id=quantum_processor_id,
        client=client,
        json_body=json_body,
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
    json_body_dict: Dict,
    httpx_request_kwargs: Dict[str, Any] = {},
) -> Response[TranslateNativeQuilToEncryptedBinaryResponse]:
    json_body = TranslateNativeQuilToEncryptedBinaryRequest.from_dict(json_body_dict)

    kwargs = _get_kwargs(
        quantum_processor_id=quantum_processor_id,
        client=client,
        json_body=json_body,
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
    json_body: TranslateNativeQuilToEncryptedBinaryRequest,
    httpx_request_kwargs: Dict[str, Any] = {},
) -> Response[TranslateNativeQuilToEncryptedBinaryResponse]:
    """Translate Native Quil To Encrypted Binary

     Compile Rigetti-native Quil code to encrypted binary form, ready for execution on a
    Rigetti Quantum Processor.

    Args:
        quantum_processor_id (str): Public identifier for a quantum processor [example: Aspen-1]
        json_body (TranslateNativeQuilToEncryptedBinaryRequest):

    Returns:
        Response[TranslateNativeQuilToEncryptedBinaryResponse]
    """

    kwargs = _get_kwargs(
        quantum_processor_id=quantum_processor_id,
        client=client,
        json_body=json_body,
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
    json_body_dict: Dict,
    httpx_request_kwargs: Dict[str, Any] = {},
) -> Response[TranslateNativeQuilToEncryptedBinaryResponse]:
    json_body = TranslateNativeQuilToEncryptedBinaryRequest.from_dict(json_body_dict)

    kwargs = _get_kwargs(
        quantum_processor_id=quantum_processor_id,
        client=client,
        json_body=json_body,
    )
    kwargs.update(httpx_request_kwargs)
    response = client.request(
        **kwargs,
    )

    return _build_response(response=response)
