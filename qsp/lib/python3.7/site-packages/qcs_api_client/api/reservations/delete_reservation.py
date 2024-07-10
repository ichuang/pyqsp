from typing import Any, Dict

import httpx
from retrying import retry

from ...models.reservation import Reservation
from ...types import Response
from ...util.errors import QCSHTTPStatusError, raise_for_status
from ...util.retry import DEFAULT_RETRY_ARGUMENTS


def _get_kwargs(
    reservation_id: int,
    *,
    client: httpx.Client,
) -> Dict[str, Any]:
    url = "{}/v1/reservations/{reservationId}".format(client.base_url, reservationId=reservation_id)

    headers = {k: v for (k, v) in client.headers.items()}
    cookies = {k: v for (k, v) in client.cookies}

    return {
        "method": "delete",
        "url": url,
        "headers": headers,
        "cookies": cookies,
        "timeout": client.timeout,
    }


def _parse_response(*, response: httpx.Response) -> Reservation:
    raise_for_status(response)
    if response.status_code == 200:
        response_200 = Reservation.from_dict(response.json())

        return response_200
    else:
        raise QCSHTTPStatusError(
            f"Unexpected response: status code {response.status_code}", response=response, error=None
        )


def _build_response(*, response: httpx.Response) -> Response[Reservation]:
    """
    Construct the Response class from the raw ``httpx.Response``.
    """
    return Response.build_from_httpx_response(response=response, parse_function=_parse_response)


@retry(**DEFAULT_RETRY_ARGUMENTS)
def sync(
    reservation_id: int,
    *,
    client: httpx.Client,
    httpx_request_kwargs: Dict[str, Any] = {},
) -> Response[Reservation]:
    """Delete Reservation

     Cancel an existing reservation for the user.

    Args:
        reservation_id (int):

    Returns:
        Response[Reservation]
    """

    kwargs = _get_kwargs(
        reservation_id=reservation_id,
        client=client,
    )
    kwargs.update(httpx_request_kwargs)
    response = client.request(
        **kwargs,
    )

    return _build_response(response=response)


@retry(**DEFAULT_RETRY_ARGUMENTS)
def sync_from_dict(
    reservation_id: int,
    *,
    client: httpx.Client,
    httpx_request_kwargs: Dict[str, Any] = {},
) -> Response[Reservation]:

    kwargs = _get_kwargs(
        reservation_id=reservation_id,
        client=client,
    )
    kwargs.update(httpx_request_kwargs)
    response = client.request(
        **kwargs,
    )

    return _build_response(response=response)


@retry(**DEFAULT_RETRY_ARGUMENTS)
async def asyncio(
    reservation_id: int,
    *,
    client: httpx.AsyncClient,
    httpx_request_kwargs: Dict[str, Any] = {},
) -> Response[Reservation]:
    """Delete Reservation

     Cancel an existing reservation for the user.

    Args:
        reservation_id (int):

    Returns:
        Response[Reservation]
    """

    kwargs = _get_kwargs(
        reservation_id=reservation_id,
        client=client,
    )
    kwargs.update(httpx_request_kwargs)
    response = await client.request(
        **kwargs,
    )

    return _build_response(response=response)


@retry(**DEFAULT_RETRY_ARGUMENTS)
async def asyncio_from_dict(
    reservation_id: int,
    *,
    client: httpx.AsyncClient,
    httpx_request_kwargs: Dict[str, Any] = {},
) -> Response[Reservation]:

    kwargs = _get_kwargs(
        reservation_id=reservation_id,
        client=client,
    )
    kwargs.update(httpx_request_kwargs)
    response = client.request(
        **kwargs,
    )

    return _build_response(response=response)
