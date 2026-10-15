"""GET-only public-IP-pinned gateway. Never forward browser cookies or bodies."""
import hashlib
import http.client
import time
import zlib
from dataclasses import dataclass
from urllib.parse import urlsplit, urljoin

from .capture import (CaptureError, normalize_url, _public_addresses,
                      _PinnedHTTP, _PinnedHTTPS, _body)

MAX_RESOURCE_BYTES = 8 * 1024 * 1024


@dataclass
class Resource:
    url: str
    status: int
    headers: dict
    body: bytes


def fetch_public_resource(url, timeout=10, max_bytes=MAX_RESOURCE_BYTES):
    """One HTTP hop; redirect targets are validated again by the caller."""
    url = normalize_url(url)
    p = urlsplit(url)
    port = p.port or (443 if p.scheme == "https" else 80)
    if port not in {80, 443}:
        raise CaptureError("Browser capture permits public ports 80 and 443 only.")
    addresses = _public_addresses(p.hostname, port)
    deadline = time.monotonic() + timeout
    cls = _PinnedHTTPS if p.scheme == "https" else _PinnedHTTP
    conn = cls(p.hostname, port, addresses[0], timeout)
    try:
        host = f"[{p.hostname}]" if ":" in p.hostname else p.hostname
        if p.port:
            host += f":{p.port}"
        conn.request("GET", p.path + ("?" + p.query if p.query else ""), headers={
            "Host": host, "User-Agent": "ResearchHTMLJSDetector/2.0 (bounded browser capture)",
            "Accept": "*/*", "Accept-Encoding": "gzip", "Connection": "close"})
        response = conn.getresponse()
        headers = {}
        # Preserve security/CORS behavior. No cookies, refresh headers or credentials.
        keep = {"content-type", "content-security-policy", "content-security-policy-report-only",
                "access-control-allow-origin", "access-control-allow-credentials",
                "access-control-expose-headers", "cross-origin-resource-policy",
                "x-content-type-options", "x-frame-options", "referrer-policy"}
        for name in keep:
            value = response.getheader(name)
            if value is not None:
                headers[name] = value
        if response.status in {301, 302, 303, 307, 308}:
            location = response.getheader("Location")
            if not location:
                raise CaptureError("HTTP redirect has no destination.")
            target = normalize_url(urljoin(url, location))
            q = urlsplit(target)
            _public_addresses(q.hostname, q.port or (443 if q.scheme == "https" else 80))
            headers["location"] = target
            return Resource(url, response.status, headers, b"")
        if response.status != 200:
            raise CaptureError(f"Resource returned HTTP {response.status}.")
        body, _ = _body(response, deadline, max_bytes)
        return Resource(url, response.status, headers, body)
    except CaptureError:
        raise
    except (OSError, http.client.HTTPException, zlib.error) as e:
        raise CaptureError(f"Resource capture failed: {type(e).__name__}: {e}") from e
    finally:
        conn.close()


def fingerprint(resource):
    return {"url": resource.url, "status": resource.status, "bytes": len(resource.body),
            "body_sha256": hashlib.sha256(resource.body).hexdigest()}
