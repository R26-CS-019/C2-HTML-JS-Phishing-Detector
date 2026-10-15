"""Bounded HTTP source capture; page JavaScript is never executed here."""
import codecs
import hashlib
import http.client
import ipaddress
import re
import socket
import ssl
import time
import zlib
from dataclasses import dataclass
from email.message import Message
from urllib.parse import quote, urljoin, urlsplit, urlunsplit

MAX_HTML_BYTES = 32 * 1024 * 1024
MAX_REDIRECTS = 5
TOTAL_TIMEOUT = 45


class CaptureError(ValueError):
    """A failed/incomplete capture must not become a benign prediction."""


def normalize_url(value):
    if not isinstance(value, str) or not value.strip():
        raise CaptureError("Enter a complete HTTP or HTTPS URL.")
    value = value.strip()
    if len(value) > 8192 or any(ord(c) <= 32 or ord(c) == 127 for c in value):
        raise CaptureError("URL is too long or contains whitespace/control characters.")
    try:
        p = urlsplit(value)
        if p.scheme.lower() not in {"http", "https"} or not p.hostname or p.username is not None or p.password is not None:
            raise CaptureError("Use an HTTP(S) URL without embedded credentials.")
        host = p.hostname.rstrip(".").lower()
        if "%" in host or not host:
            raise CaptureError("Invalid hostname.")
        try:
            address = ipaddress.ip_address(host)
            host = address.compressed
            authority = f"[{host}]" if address.version == 6 else host
        except ValueError:
            host = host.encode("idna").decode("ascii")
            if len(host) > 253 or not re.fullmatch(r"[a-z0-9.-]+", host) or ".." in host:
                raise CaptureError("Invalid hostname.")
            if any(not label or len(label) > 63 or label[0] == "-" or label[-1] == "-" for label in host.split(".")):
                raise CaptureError("Invalid hostname.")
            authority = host
        port = p.port
        if port is not None and not 1 <= port <= 65535:
            raise CaptureError("Invalid port.")
        if port is not None:
            authority += f":{port}"
        path = quote(p.path or "/", safe="/%:@!$&'()*+,;=-._~")
        query = quote(p.query, safe="/%?:@!$&'()*+,;=-._~[]")
        return urlunsplit((p.scheme.lower(), authority, path, query, ""))
    except (UnicodeError, ValueError) as e:
        if isinstance(e, CaptureError):
            raise
        raise CaptureError("Invalid URL.") from e


def _public_addresses(host, port):
    try:
        results = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as e:
        raise CaptureError("Could not resolve the website hostname.") from e
    addresses = list(dict.fromkeys(row[4][0] for row in results))
    if not addresses:
        raise CaptureError("Hostname has no usable IP addresses.")
    for value in addresses:
        address = ipaddress.ip_address(value)
        embedded = getattr(address, "ipv4_mapped", None)
        if not address.is_global or address.is_multicast or (embedded and not embedded.is_global):
            raise CaptureError("URL resolves to a private, local, reserved or multicast address. Use a public website.")
    return addresses


class _PinnedHTTP(http.client.HTTPConnection):
    def __init__(self, host, port, address, timeout):
        super().__init__(host, port, timeout=timeout)
        self.address = address

    def connect(self):
        self.sock = socket.create_connection((self.address, self.port), self.timeout)
        if ipaddress.ip_address(self.sock.getpeername()[0]) != ipaddress.ip_address(self.address):
            self.sock.close()
            raise CaptureError("Connected address differs from the validated address.")


class _PinnedHTTPS(_PinnedHTTP):
    def connect(self):
        super().connect()
        try:
            self.sock = ssl.create_default_context().wrap_socket(self.sock, server_hostname=self.host)
        except BaseException:
            self.sock.close()
            raise


def _body(response, deadline, max_bytes):
    encoding = (response.getheader("Content-Encoding") or "identity").strip().lower()
    if encoding not in {"identity", "gzip"}:
        raise CaptureError("Unsupported content encoding. Capture is incomplete.")
    length = response.getheader("Content-Length")
    if length:
        try:
            if int(length) < 0 or int(length) > max_bytes:
                raise CaptureError("Response exceeds the capture limit. No model prediction was made.")
        except ValueError as e:
            if isinstance(e, CaptureError):
                raise
            raise CaptureError("Invalid response length.") from e
    decoder = zlib.decompressobj(16 + zlib.MAX_WBITS) if encoding == "gzip" else None
    data = bytearray()
    transferred = 0
    while True:
        if time.monotonic() > deadline:
            raise CaptureError("Capture timed out before the complete HTML arrived.")
        chunk = response.read1(64 * 1024)
        if not chunk:
            break
        transferred += len(chunk)
        if transferred > max_bytes:
            raise CaptureError("Response exceeds the capture limit. No model prediction was made.")
        if decoder:
            decoded = decoder.decompress(chunk, max_bytes - len(data) + 1)
            if decoder.unconsumed_tail or len(data) + len(decoded) > max_bytes:
                raise CaptureError("Decompressed HTML exceeds the capture limit.")
            data.extend(decoded)
        else:
            data.extend(chunk)
        if len(data) > max_bytes:
            raise CaptureError("HTML exceeds the capture limit.")
    if length and transferred != int(length):
        raise CaptureError("Response ended before the declared content length. No model prediction was made.")
    if decoder:
        if not decoder.eof or decoder.unused_data:
            raise CaptureError("Incomplete or unsupported gzip response.")
    return bytes(data), transferred


def decode_html(data, content_type="text/html", allow_replacement=False):
    header = Message()
    header["content-type"] = content_type
    charset = header.get_content_charset()
    source = "http_header" if charset else "default_utf8"
    if data.startswith(codecs.BOM_UTF8):
        charset, source = "utf-8-sig", "bom"
    elif data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        charset, source = "utf-16", "bom"
    elif not charset:
        prefix = data[:4096].decode("ascii", errors="ignore")
        match = re.search(r"<meta\b[^>]*\bcharset\s*=\s*['\"]?\s*([a-zA-Z0-9._-]+)", prefix, re.I)
        if match:
            charset, source = match.group(1), "html_meta"
    charset = charset or "utf-8"
    try:
        html = data.decode(charset, errors="strict")
        replacement = False
    except (LookupError, UnicodeError) as e:
        if not allow_replacement:
            raise CaptureError("HTML encoding could not be decoded completely. No model prediction was made.") from e
        html = data.decode("utf-8", errors="replace")
        charset, replacement = "utf-8", True
    if not html.strip():
        raise CaptureError("The website returned empty HTML.")
    return html, {"character_encoding": charset, "encoding_basis": source,
                  "replacement_decoding_used": replacement}


@dataclass
class Capture:
    html: str
    metadata: dict


def capture_url(url, max_bytes=MAX_HTML_BYTES, timeout=TOTAL_TIMEOUT):
    initial = current = normalize_url(url)
    deadline = time.monotonic() + timeout
    redirects = []
    seen = set()
    for hop in range(MAX_REDIRECTS + 1):
        if current in seen:
            raise CaptureError("Website has a redirect loop.")
        seen.add(current)
        p = urlsplit(current)
        port = p.port or (443 if p.scheme == "https" else 80)
        addresses = _public_addresses(p.hostname, port)  # Check every redirect, then pin its connection.
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise CaptureError("Capture timed out.")
        connection_type = _PinnedHTTPS if p.scheme == "https" else _PinnedHTTP
        connection = connection_type(p.hostname, port, addresses[0], min(12, remaining))
        try:
            host = f"[{p.hostname}]" if ":" in p.hostname else p.hostname
            if p.port:
                host += f":{p.port}"
            target = p.path + ("?" + p.query if p.query else "")
            connection.request("GET", target, headers={
                "Host": host, "User-Agent": "ResearchHTMLJSDetector/1.0 (static HTML capture)",
                "Accept": "text/html,application/xhtml+xml", "Accept-Encoding": "gzip",
                "Connection": "close"})
            response = connection.getresponse()
            if response.status in {301, 302, 303, 307, 308}:
                location = response.getheader("Location")
                if not location or hop == MAX_REDIRECTS:
                    raise CaptureError("Missing redirect destination or redirect limit reached.")
                following = normalize_url(urljoin(current, location))
                redirects.append({"url": current, "status": response.status, "destination": following})
                current = following
                continue
            if response.status != 200:
                raise CaptureError(f"Website returned HTTP {response.status}. No model prediction was made.")
            content_type = response.getheader("Content-Type") or ""
            mime = content_type.split(";", 1)[0].strip().lower()
            if mime not in {"text/html", "application/xhtml+xml"}:
                raise CaptureError("Website did not return an HTML content type. No model prediction was made.")
            data, transferred = _body(response, deadline, max_bytes)
            html, encoding = decode_html(data, content_type)
            return Capture(html, {"requested_url": initial, "final_url": current,
                "capture_basis": "http_source", "capture_complete": True,
                "status_code": response.status, "content_type": content_type,
                "html_bytes": len(data), "transferred_bytes": transferred,
                "response_body_sha256": hashlib.sha256(data).hexdigest(),
                "redirects": redirects, "elapsed_seconds": time.monotonic() - (deadline - timeout),
                "javascript_executed": False, "external_resources_requested": False, **encoding})
        except CaptureError:
            raise
        except (OSError, http.client.HTTPException, zlib.error) as e:
            raise CaptureError(f"Could not capture complete HTML: {type(e).__name__}: {e}") from e
        finally:
            connection.close()
    raise CaptureError("Redirect limit reached.")
