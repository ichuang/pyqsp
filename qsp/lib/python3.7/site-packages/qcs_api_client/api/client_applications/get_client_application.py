from typing import Any, Dict

import httpx
from retrying import retry

from ...models.client_application import ClientApplication
from ...types import Response
from ...util.errors import QCSHTTPStatusError, raise_for_status
from ...util.retry import DEFAULT_RETRY_ARGUMENTS


def _get_kwargs(
    client_application_name: str,
    *,
    client: httpx.Client,
) -> Dict[str, Any]:
    url = "{}/v1/clientApplications/{clientApplicationName}".format(
        client.base_url, clientApplicationName=client_application_name
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


def _parse_response(*, response: httpx.Response) -> ClientApplication:
    raise_for_status(response)
    if response.status_code == 200:
        response_200 = ClientApplication.from_dict(response.json())

        return response_200
    else:
        raise QCSHTTPStatusError(
            f"Unexpected response: status code {response.status_code}", response=response, error=None
        )


def _build_response(*, response: httpx.Response) -> Response[ClientApplication]:
    """
    Construct the Response class from the raw ``httpx.Response``.
    """
    return Response.build_from_httpx_response(response=response, parse_function=_parse_response)


@retry(**DEFAULT_RETRY_ARGUMENTS)
def sync(
    client_application_name: str,
    *,
    client: httpx.Client,
    httpx_request_kwargs: Dict[str, Any] = {},
) -> Response[ClientApplication]:
    """Get Client Application

     Get details of a specific Rigetti system component along with its latest and minimum supported
    versions.

    Args:
        client_application_name (str):

    Returns:
        Response[ClientApplication]
    """

    kwargs = _get_kwargs(
        client_application_name=client_application_name,
        client=client,
    )
    kwargs.update(httpx_request_kwargs)
    response = client.request(
        **kwargs,
    )

    return _build_response(response=response)


@retry(**DEFAULT_RETRY_ARGUMENTS)
def sync_from_dict(
    client_application_name: str,
    *,
    client: httpx.Client,
    httpx_request_kwargs: Dict[str, Any] = {},
) -> Response[ClientApplication]:

    kwargs = _get_kwargs(
        client_application_name=client_application_name,
        client=client,
    )
    kwargs.update(httpx_request_kwargs)
    response = client.request(
        **kwargs,
    )

    return _build_response(response=response)


@retry(**DEFAULT_RETRY_ARGUMENTS)
async def asyncio(
    client_application_name: str,
    *,
    client: httpx.AsyncClient,
    httpx_request_kwargs: Dict[str, Any] = {},
) -> Response[ClientApplication]:
    """Get Client Application

     Get details of a specific Rigetti system component along with its latest and minimum supported
    versions.

    Args:
        client_application_name (str):

    Returns:
        Response[ClientApplication]
    """

    kwargs = _get_kwargs(
        client_application_name=client_application_name,
        client=client,
    )
    kwargs.update(httpx_request_kwargs)
    response = await client.request(
        **kwargs,
    )

    return _build_response(response=response)


@retry(**DEFAULT_RETRY_ARGUMENTS)
async def asyncio_from_dict(
    client_application_name: str,
    *,
    client: httpx.AsyncClient,
    httpx_request_kwargs: Dict[str, Any] = {},
) -> Response[ClientApplication]:

    kwargs = _get_kwargs(
        client_application_name=client_application_name,
        client=client,
    )
    kwargs.update(httpx_request_kwargs)
    response = client.request(
        **kwargs,
    )

    return _build_response(response=response)
