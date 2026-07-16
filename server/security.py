import hmac
import ipaddress
import logging
import socket
from functools import wraps
from urllib.parse import urlsplit

from flask import request, jsonify

from config import LAB_ADMIN_TOKEN, TRAIN_ALLOWED_HOSTS

logger = logging.getLogger(__name__)

# Only these schemes may be fetched server-side. file://, ftp://, gopher://, etc.
# are rejected to prevent local file access and protocol smuggling.
ALLOWED_SCHEMES = {"https"}


def require_admin_token(view):
    """
    Protect a route with a static admin bearer token.

    The token is read from the `Authorization` header (raw value or `Bearer <token>`)
    and compared against LAB_ADMIN_TOKEN in constant time.
    """

    @wraps(view)
    def wrapper(*args, **kwargs):
        if not LAB_ADMIN_TOKEN:
            logger.error("LAB_ADMIN_TOKEN is not configured; refusing protected request")
            return jsonify({'error': 'Server misconfigured: authentication unavailable'}), 503

        header = request.headers.get('Authorization', '').strip()
        if header.lower().startswith('bearer '):
            provided = header[len('bearer '):].strip()
        else:
            provided = header

        if not provided or not hmac.compare_digest(provided, LAB_ADMIN_TOKEN):
            logger.warning("Unauthorized request on %s", request.path)
            return jsonify({'error': 'Unauthorized'}), 401

        return view(*args, **kwargs)

    return wrapper


class UnsafeUrlError(ValueError):
    """Raised when a user-supplied URL fails SSRF validation."""


def validate_public_url(url):
    """
    Validate a user-supplied URL before it is fetched server-side (SSRF guard).

    Enforces:
      - allowed scheme (https only)
      - host present in the TRAIN_ALLOWED_HOSTS allowlist
      - every resolved IP is public (no loopback/private/link-local/reserved)

    Raises:
        UnsafeUrlError: if the URL is not safe to fetch.

    Returns:
        str: the validated URL.
    """
    if not url or not isinstance(url, str):
        raise UnsafeUrlError("endpoint is required")

    parts = urlsplit(url)

    if parts.scheme not in ALLOWED_SCHEMES:
        raise UnsafeUrlError(f"scheme '{parts.scheme}' is not allowed")

    host = parts.hostname
    if not host:
        raise UnsafeUrlError("endpoint has no host")

    if host not in TRAIN_ALLOWED_HOSTS:
        raise UnsafeUrlError(f"host '{host}' is not in the allowlist")

    # Restrict to the default HTTPS port: an allowlisted host must not be reachable
    # on arbitrary ports (would expose other internal services bound to that host).
    if parts.port is not None and parts.port != 443:
        raise UnsafeUrlError(f"port '{parts.port}' is not allowed")

    # Resolve DNS and ensure no resolved address points to a non-public range.
    try:
        infos = socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise UnsafeUrlError(f"could not resolve host '{host}'") from exc

    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            raise UnsafeUrlError(f"host '{host}' resolves to a non-public address")

    return url
