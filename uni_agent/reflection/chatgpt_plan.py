"""ChatGPT plan usage for a hosted reflector, through Sign in with ChatGPT's open-source OAuth flow.

A one-time browser login writes a credentials file (mode 0600); every call then takes a fresh access token from it.
Refresh tokens rotate on each use, so the renewal runs under an exclusive lock on the file: whichever process holds
it renews and rewrites the file, and the others read the new token.

    python -m uni_agent.reflection.chatgpt_plan login [--out ~/.config/agentic-sdpo/chatgpt_plan.json]

Stdlib only, so the login runs on any machine with a browser.
"""
from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import json
import os
import secrets
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

AUTHORIZE_URL = "https://auth.openai.com/api/accounts/authorize"
TOKEN_URL = "https://auth.openai.com/api/accounts/oauth/token"
RESOURCE = "https://api.openai.com/v1"
SCOPES = "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"
APP_NAME = "uni-agent"
PORT = 1455
DEFAULT_PATH = "~/.config/agentic-sdpo/chatgpt_plan.json"
#: a token this close to expiry is renewed before use
MARGIN_S = 300.0


class PlanUnavailable(RuntimeError):
    """The plan cannot serve calls until the user signs in again (the refresh was refused or the file is unusable)."""


def _post_form(url: str, fields: dict, timeout: float = 30.0) -> dict:
    request = urllib.request.Request(url, data=urllib.parse.urlencode(fields).encode(),
                                     headers={"Content-Type": "application/x-www-form-urlencoded"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        try:
            code = json.loads(body).get("error")
        except json.JSONDecodeError:
            code = body[:200]
        raise PlanUnavailable(f"token endpoint refused ({exc.code}): {code}") from exc


def _write(path: Path, creds: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        json.dump(creds, fh, indent=1)
    os.replace(tmp, path)


def _with_tokens(creds: dict, tokens: dict) -> dict:
    return {**creds, "access_token": tokens["access_token"],
            "refresh_token": tokens.get("refresh_token") or creds.get("refresh_token"),
            "expires_at": time.time() + float(tokens.get("expires_in") or 3600)}


def access_token(path: str | os.PathLike) -> str:
    """A usable access token from the credentials file, renewing it (under the lock) when it is about to expire."""
    path = Path(os.path.expanduser(str(path)))
    if not path.is_file():
        raise PlanUnavailable(f"no ChatGPT plan credentials at {path}")
    with open(path.with_name(path.name + ".lock"), "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            creds = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise PlanUnavailable(f"unreadable ChatGPT plan credentials at {path}: {exc}") from exc
        if float(creds.get("expires_at") or 0) - time.time() > MARGIN_S:
            return creds["access_token"]
        tokens = _post_form(TOKEN_URL, {"grant_type": "refresh_token", "client_id": creds["client_id"],
                                        "refresh_token": creds["refresh_token"], "resource": RESOURCE})
        creds = _with_tokens(creds, tokens)
        _write(path, creds)
        return creds["access_token"]


def _pkce() -> tuple[str, str]:
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(48)).rstrip(b"=").decode()
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    return verifier, challenge


def login(out: str) -> Path:
    """Browser sign-in: consent to plan usage for this app, then write the credentials file."""
    path = Path(os.path.expanduser(out))
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = json.loads(path.read_text()) if path.is_file() else {}
    host_id = previous.get("ext_agent_host_id") or f"urn:uuid:{uuid.uuid4()}"
    verifier, challenge = _pkce()
    state, nonce = secrets.token_urlsafe(24), secrets.token_urlsafe(24)
    redirect = f"http://127.0.0.1:{PORT}/callback"
    params = {"client_id": previous.get("client_id") or "dynamic_agent_client", "ext_agent_host_id": host_id,
              "response_type": "code", "redirect_uri": redirect, "scope": SCOPES, "resource": RESOURCE,
              "state": state, "nonce": nonce, "code_challenge_method": "S256", "code_challenge": challenge}
    if not previous.get("client_id"):
        params["agent_name_hint"] = APP_NAME
    received: dict = {}

    class Callback(BaseHTTPRequestHandler):
        def do_GET(self):
            query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            received.update({k: v[0] for k, v in query.items()})
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Signed in; you can close this tab.")

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", PORT), Callback)
    url = f"{AUTHORIZE_URL}?{urllib.parse.urlencode(params)}"
    print(f"Opening the ChatGPT consent page; if no browser opens, visit:\n{url}")
    webbrowser.open(url)
    while "code" not in received and "error" not in received:
        server.handle_request()
    server.server_close()
    if "error" in received:
        raise SystemExit(f"sign-in refused: {received.get('error')}: {received.get('error_description', '')}")
    if received.get("state") != state:
        raise SystemExit("sign-in answer does not match this attempt (state mismatch); nothing written")
    client_id = received.get("client_id") or previous.get("client_id")
    if not client_id:
        raise SystemExit("the callback carried no issued client_id; nothing written")
    tokens = _post_form(TOKEN_URL, {"grant_type": "authorization_code", "client_id": client_id,
                                    "code": received["code"], "code_verifier": verifier, "redirect_uri": redirect,
                                    "resource": RESOURCE})
    granted = set((tokens.get("scope") or "").split())
    if "chatgpt.tokens.use.direct" not in granted:
        raise SystemExit(f"plan usage was not granted (scopes: {sorted(granted)}); nothing written")
    _write(path, _with_tokens({"client_id": client_id, "ext_agent_host_id": host_id, "scope": tokens.get("scope")},
                              tokens))
    print(f"ChatGPT plan credentials written to {path} (keep it private; copy it to the training host as is)")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["login"])
    parser.add_argument("--out", default=DEFAULT_PATH)
    args = parser.parse_args()
    login(args.out)


if __name__ == "__main__":
    main()
