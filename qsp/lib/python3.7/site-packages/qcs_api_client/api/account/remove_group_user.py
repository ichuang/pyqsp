from typing import Any, Dict, cast

import httpx
from retrying import retry

from ...models.remove_group_user_request import RemoveGroupUserRequest
from ...types import Response
from ...util.errors import QCSHTTPStatusError, raise_for_status
from ...util.retry import DEFAULT_RETRY_ARGUMENTS


def _get_kwargs(
    *,
    client: httpx.Client,
    json_body: RemoveGroupUserRequest,
) -> Dict[str, Any]:
    url = "{}/v1/groups:removeUser".format(client.base_url)

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


def _parse_response(*, response: httpx.Response) -> Any:
    raise_for_status(response)
    if response.status_code == 204:
        response_204 = cast(Any, None)
        return response_204
    else:
        raise QCSHTTPStatusError(
            f"Unexpected response: status code {response.status_code}", response=response, error=None
        )


def _build_response(*, response: httpx.Response) -> Response[Any]:
    """
    Construct the Response class from the raw ``httpx.Response``.
    """
    return Response.build_from_httpx_response(response=response, parse_function=_parse_response)


@retry(**DEFAULT_RETRY_ARGUMENTS)
def sync(
    *,
    client: httpx.Client,
    json_body: RemoveGroupUserRequest,
    httpx_request_kwargs: Dict[str, Any] = {},
) -> Response[Any]:
    """Remove user from a group.

     Remove a user from a group. Note, group membership may take several minutes to update within our
    identity provider. After removing a user from a group, please allow up to 60 minutes for changes to
    be reflected.

    Args:
        json_body (RemoveGroupUserRequest): Must provide either `userId` or `userEmail` and
            `groupId` or `groupName`.

    Returns:
        Response[Any]
    """

    kwargs = _get_kwargs(
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
    *,
    client: httpx.Client,
    json_body_dict: Dict,
    httpx_request_kwargs: Dict[str, Any] = {},
) -> Response[Any]:
    json_body = RemoveGroupUserRequest.from_dict(json_body_dict)

    kwargs = _get_kwargs(
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
    *,
    client: httpx.AsyncClient,
    json_body: RemoveGroupUserRequest,
    httpx_request_kwargs: Dict[str, Any] = {},
) -> Response[Any]:
    """Remove user from a group.

     Remove a user from a group. Note, group membership may take several minutes to update within our
    identity provider. After removing a user from a group, please allow up to 60 minutes for changes to
    be reflected.

    Args:
        json_body (RemoveGroupUserRequest): Must provide either `userId` or `userEmail` and
            `groupId` or `groupName`.

    Returns:
        Response[Any]
    """

    kwargs = _get_kwargs(
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
    *,
    client: httpx.AsyncClient,
    json_body_dict: Dict,
    httpx_request_kwargs: Dict[str, Any] = {},
) -> Response[Any]:
    json_body = RemoveGroupUserRequest.from_dict(json_body_dict)

    kwargs = _get_kwargs(
        client=client,
        json_body=json_body,
    )
    kwargs.update(httpx_request_kwargs)
    response = client.request(
        **kwargs,
    )

    return _build_response(response=response)
