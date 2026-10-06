"""ChatGPT plan credentials: a fresh token is used as is, a stale one is renewed once under the lock with its rotated
refresh token written back privately, and a refused renewal or a missing file turns the plan off."""

import base64
import hashlib
import json
import stat
import time

import pytest

from uni_agent.reflection import chatgpt_plan


def write_creds(tmp_path, expires_in):
    path = tmp_path / "chatgpt_plan.json"
    path.write_text(json.dumps({"client_id": "oaiapp_1", "ext_agent_host_id": "urn:uuid:h", "access_token": "old",
                                "refresh_token": "r1", "expires_at": time.time() + expires_in}))
    return path


def test_a_fresh_token_is_used_without_renewal(tmp_path, monkeypatch):
    monkeypatch.setattr(chatgpt_plan, "_post_form", lambda *a, **k: pytest.fail("renewed a fresh token"))
    assert chatgpt_plan.access_token(write_creds(tmp_path, 3000)) == "old"


def test_a_stale_token_is_renewed_and_the_rotated_refresh_token_kept(tmp_path, monkeypatch):
    calls = []

    def post(url, fields, timeout=30.0):
        calls.append((url, fields))
        return {"access_token": "new", "refresh_token": "r2", "expires_in": 3600}

    monkeypatch.setattr(chatgpt_plan, "_post_form", post)
    path = write_creds(tmp_path, 60)
    assert chatgpt_plan.access_token(path) == "new"
    assert calls == [(chatgpt_plan.TOKEN_URL, {"grant_type": "refresh_token", "client_id": "oaiapp_1",
                                                "refresh_token": "r1", "resource": chatgpt_plan.RESOURCE})]
    creds = json.loads(path.read_text())
    assert creds["refresh_token"] == "r2" and creds["access_token"] == "new" and creds["expires_at"] > time.time() + 3000
    assert creds["ext_agent_host_id"] == "urn:uuid:h" and stat.S_IMODE(path.stat().st_mode) == 0o600
    # the next caller reads the renewed token without renewing again
    assert chatgpt_plan.access_token(path) == "new" and len(calls) == 1


def test_a_refused_renewal_leaves_the_file_and_turns_the_plan_off(tmp_path, monkeypatch):
    def refused(url, fields, timeout=30.0):
        raise chatgpt_plan.PlanUnavailable("token endpoint refused (400): refresh_token_reused")

    monkeypatch.setattr(chatgpt_plan, "_post_form", refused)
    path = write_creds(tmp_path, 60)
    before = path.read_text()
    with pytest.raises(chatgpt_plan.PlanUnavailable, match="refresh_token_reused"):
        chatgpt_plan.access_token(path)
    assert path.read_text() == before


def test_missing_credentials_turn_the_plan_off(tmp_path):
    with pytest.raises(chatgpt_plan.PlanUnavailable, match="no ChatGPT plan credentials"):
        chatgpt_plan.access_token(tmp_path / "absent.json")


def test_pkce_challenge_is_the_verifiers_s256():
    verifier, challenge = chatgpt_plan._pkce()
    expected = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    assert challenge == expected and 43 <= len(verifier) <= 128 and "=" not in verifier
