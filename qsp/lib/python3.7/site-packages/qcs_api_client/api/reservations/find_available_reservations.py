import datetime
from typing import Any, Dict

import httpx
from retrying import retry
from rfc3339 import rfc3339

from ...models.find_available_reservations_response import FindAvailableReservationsResponse
from ...types import UNSET, Response
from ...util.errors import QCSHTTPStatusError, raise_for_status
from ...util.retry import DEFAULT_RETRY_ARGUMENTS


def _get_kwargs(
    *,
    client: httpx.Client,
    quantum_processor_id: str,
    start_time_from: datetime.datetime,
    duration: str,
) -> Dict[str, Any]:
    url = "{}/v1/reservations:findAvailable".format(client.base_url)

    headers = {k: v for (k, v) in client.headers.items()}
    cookies = {k: v for (k, v) in client.cookies}

    params: Dict[str, Any] = {}
    params["quantumProcessorId"] = quantum_processor_id

    assert start_time_from.tzinfo is not None, "Datetime must have timezone information"
    json_start_time_from = rfc3339(start_time_from)

    params["startTimeFrom"] = json_start_time_from

    params["duration"] = duration

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    return {
        "method": "get",
        "url": url,
        "headers": headers,
        "cookies": cookies,
        "timeout": client.timeout,
        "params": params,
    }


def _parse_response(*, response: httpx.Response) -> FindAvailableReservationsResponse:
    raise_for_status(response)
    if response.status_code == 200:
        response_200 = FindAvailableReservationsResponse.from_dict(response.json())

        return response_200
    else:
        raise QCSHTTPStatusError(
            f"Unexpected response: status code {response.status_code}", response=response, error=None
        )


def _build_response(*, response: httpx.Response) -> Response[FindAvailableReservationsResponse]:
    """
    Construct the Response class from the raw ``httpx.Response``.
    """
    return Response.build_from_httpx_response(response=response, parse_function=_parse_response)


@retry(**DEFAULT_RETRY_ARGUMENTS)
def sync(
    *,
    client: httpx.Client,
    quantum_processor_id: str,
    start_time_from: datetime.datetime,
    duration: str,
    httpx_request_kwargs: Dict[str, Any] = {},
) -> Response[FindAvailableReservationsResponse]:
    """Find Available Reservations

     List currently available reservations on the requested Rigetti quantum computer.

    Args:
        quantum_processor_id (str):
        start_time_from (datetime.datetime):
        duration (str): Formatted as specified for golang
            https://golang.org/pkg/time/#ParseDuration.

    Returns:
        Response[FindAvailableReservationsResponse]
    """

    kwargs = _get_kwargs(
        client=client,
        quantum_processor_id=quantum_processor_id,
        start_time_from=start_time_from,
        duration=duration,
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
    quantum_processor_id: str,
    start_time_from: datetime.datetime,
    duration: str,
    httpx_request_kwargs: Dict[str, Any] = {},
) -> Response[FindAvailableReservationsResponse]:

    kwargs = _get_kwargs(
        client=client,
        quantum_processor_id=quantum_processor_id,
        start_time_from=start_time_from,
        duration=duration,
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
    quantum_processor_id: str,
    start_time_from: datetime.datetime,
    duration: str,
    httpx_request_kwargs: Dict[str, Any] = {},
) -> Response[FindAvailableReservationsResponse]:
    """Find Available Reservations

     List currently available reservations on the requested Rigetti quantum computer.

    Args:
        quantum_processor_id (str):
        start_time_from (datetime.datetime):
        duration (str): Formatted as specified for golang
            https://golang.org/pkg/time/#ParseDuration.

    Returns:
        Response[FindAvailableReservationsResponse]
    """

    kwargs = _get_kwargs(
        client=client,
        quantum_processor_id=quantum_processor_id,
        start_time_from=start_time_from,
        duration=duration,
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
    quantum_processor_id: str,
    start_time_from: datetime.datetime,
    duration: str,
    httpx_request_kwargs: Dict[str, Any] = {},
) -> Response[FindAvailableReservationsResponse]:

    kwargs = _get_kwargs(
        client=client,
        quantum_processor_id=quantum_processor_id,
        start_time_from=start_time_from,
        duration=duration,
    )
    kwargs.update(httpx_request_kwargs)
    response = client.request(
        **kwargs,
    )

    return _build_response(response=response)
