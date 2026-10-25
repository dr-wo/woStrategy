"""Offline compatibility and synthetic auth failures; never read user tokens.

Run against the exact FastF13.9 wheel in the isolated validation environment.
All authentication managers use a pytest temporary path and non-secret dummy
values. No login endpoint, browser, credentials, or production token file is used.
"""
import base64
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time
import zlib

import fastf1
from fastf1 import _api
from fastf1.core import SessionResults
from fastf1.livetiming.client import SignalRClient
import pandas as pd
import pytest

from wodata.live.replay.archive import FastF1ArchiveClient
from wostrategy.core.session_loader import _add_session_result_rank


def test_consumed_private_and_live_api_call_shapes_remain_supported():
    from fastf1.internals.f1account import token_manager
    inspect.signature(_api.fetch_page).bind("/session/", "timing_data")
    inspect.signature(_api.parse).bind("payload", zipped=True)
    inspect.signature(_api.car_data).bind("/session/")
    inspect.signature(_api.position_data).bind("/session/")
    inspect.signature(SignalRClient).bind(filename="/dev/null", timeout=0,
                                         logger=None, no_auth=False)
    assert callable(fastf1.get_session)
    assert callable(fastf1.set_auth_method)
    assert callable(token_manager.get_stored)
    assert callable(fastf1.Cache.disabled)


@pytest.mark.parametrize("topic,page", [("CarData.z", "car_data"), ("Position.z", "position")])
def test_archive_adapter_uses_real_fastf1_compressed_payload_parser(monkeypatch, topic, page):
    payload = {"SyntheticCompatibility": True, "Values": [1, 2]}
    compressor = zlib.compressobj(wbits=-15)
    compressed = compressor.compress(json.dumps(payload).encode()) + compressor.flush()
    encoded = base64.b64encode(compressed).decode()
    calls = []
    monkeypatch.setattr(_api, "fetch_page", lambda path, name:
                        calls.append((path, name)) or ["00:00:01:250" + encoded])
    records = FastF1ArchiveClient().fetch_topic("/session/", topic)
    assert calls == [("/session/", page)]
    assert records[0].payload == payload
    assert records[0].session_time_seconds == 1.25


@pytest.mark.parametrize("status", [401, 404])
def test_low_level_missing_page_requires_separate_access_diagnostics(monkeypatch, status):
    # This third-party API deliberately loses HTTP status on failure; application
    # preflight must not classify a generic empty page as a parser defect.
    monkeypatch.setattr(fastf1.Cache, "requests_get", lambda *args, **kwargs:
                        SimpleNamespace(status_code=status))
    assert _api.fetch_page("/session/", "timing_data") is None
    assert FastF1ArchiveClient().fetch_topic("/session/", "TimingData") == []


def test_configure_keyword_contract_and_cache_disabled_context(tmp_path, monkeypatch):
    if not hasattr(fastf1.Cache, "configure"):
        pytest.skip("Cache.configure requires FastF1>=3.9")
    signature = inspect.signature(fastf1.Cache.configure)
    assert signature.parameters["cache_dir"].kind is inspect.Parameter.KEYWORD_ONLY
    fields = ("_CACHE_DIR", "_FORCE_RENEW", "_IGNORE_VERSION", "_requests_session_cached")
    for name in fields:
        monkeypatch.setattr(fastf1.Cache, name, getattr(fastf1.Cache, name))
    fastf1.Cache.configure(cache_dir=str(tmp_path), force_renew=True,
                          use_requests_cache=False)
    assert Path(fastf1.Cache._CACHE_DIR) == tmp_path
    assert fastf1.Cache._FORCE_RENEW is True
    assert fastf1.Cache._requests_session_cached is None
    before = fastf1.Cache._http_tmp_disabled, fastf1.Cache._func_tmp_disabled
    with fastf1.Cache.disabled(disable_http_cache=True, disable_func_cache=False):
        assert fastf1.Cache._http_tmp_disabled is True
        assert fastf1.Cache._func_tmp_disabled is False
    assert (fastf1.Cache._http_tmp_disabled, fastf1.Cache._func_tmp_disabled) == before


def test_cache_sessions_expose_scoped_response_hooks(tmp_path, monkeypatch):
    if not hasattr(fastf1.Cache, "configure"):
        pytest.skip("Cache.configure requires FastF1>=3.9")
    import requests
    for name in ("_CACHE_DIR", "_FORCE_RENEW", "_IGNORE_VERSION", "_requests_session_cached"):
        monkeypatch.setattr(fastf1.Cache, name, getattr(fastf1.Cache, name))
    fastf1.Cache.configure(cache_dir=str(tmp_path))
    sessions = (fastf1.Cache._requests_session, fastf1.Cache._requests_session_cached)
    assert all(isinstance(session, requests.Session) for session in sessions)
    original = [list(session.hooks.get("response", ())) for session in sessions]
    hook = lambda response, **kwargs: response
    try:
        for session in sessions:
            session.hooks.setdefault("response", []).append(hook)
            assert session.hooks["response"][-1] is hook
    finally:
        for session, saved in zip(sessions, original):
            session.hooks["response"][:] = saved
        fastf1.Cache._requests_session_cached.close()


def test_result_consumer_keeps_numeric_position_when_classification_is_n():
    results = SessionResults(pd.DataFrame({
        "Abbreviation": ["FAST", "SLOW"], "Position": [1., 22.],
        "ClassifiedPosition": ["1", "N"],
        #3.9 retains Q1 timing even when it is outside the107% classification.
        "Q1": pd.to_timedelta([90., 110.], unit="s"),
    }))
    laps = pd.DataFrame({"Driver": ["FAST", "SLOW"], "LapTimeSeconds": [90., 110.]})
    output = _add_session_result_rank(laps, SimpleNamespace(results=results))
    assert output.set_index("Driver")["SessionResultRank"].to_dict() == {"FAST": 1., "SLOW": 22.}
    assert len(output) == len(laps)
    assert results.loc[1, "Q1"] == pd.Timedelta(seconds=110.)


def test_new_practice_result_times_and_positions_do_not_replace_lap_measurements():
    results = SessionResults(pd.DataFrame({"Abbreviation": ["AAA", "BBB"],
        "Position": [2., 1.], "Time": pd.to_timedelta([91., 90.], unit="s")}))
    laps = pd.DataFrame({"Driver": ["AAA", "BBB"], "LapTimeSeconds": [94., 95.]})
    output = _add_session_result_rank(laps, SimpleNamespace(results=results))
    assert output["LapTimeSeconds"].tolist() == [94., 95.]
    assert output["SessionResultRank"].tolist() == [2., 1.]


@pytest.fixture
def auth_types():
    module = pytest.importorskip("fastf1.internals.f1account.tokens")
    from fastf1.exceptions import F1AuthError
    return module.TokenManager, module.SubscriptionToken, F1AuthError


@pytest.mark.parametrize("state", ["missing", "expired", "corrupt"])
def test_isolated_token_states_need_explicit_reauthentication(tmp_path, auth_types, state):
    TokenManager, SubscriptionToken, F1AuthError = auth_types
    calls = []
    def deny_login():
        calls.append("requested")
        raise F1AuthError("Synthetic authentication required; use the official local login.")
    path = tmp_path / "synthetic-auth.json"
    manager = TokenManager(auth_func=deny_login, path=path)
    if state == "expired":
        manager.store(SubscriptionToken(value="synthetic-test-only", expires_at=0.))
    elif state == "corrupt":
        path.write_text("not-json")
    # Passive status checks must never launch an authenticator.
    stored = manager.get_stored()
    assert stored is None or not stored.is_valid()
    assert calls == []
    with pytest.raises(F1AuthError, match="Synthetic authentication required"):
        manager.get()
    assert calls == ["requested"]


def test_synthetic_token_persists_between_managers_without_login(tmp_path, auth_types):
    TokenManager, SubscriptionToken, _ = auth_types
    def unexpected_login():
        pytest.fail("Valid isolated stored token should not trigger login")
    path = tmp_path / "synthetic-auth.json"
    token = SubscriptionToken(value="synthetic-test-only", expires_at=time.time() + 86400.)
    TokenManager(auth_func=unexpected_login, path=path).store(token)
    recovered = TokenManager(auth_func=unexpected_login, path=path).get()
    assert recovered.is_valid()
    assert recovered.expires_at == token.expires_at
    assert "synthetic-test-only" not in repr(recovered)
    if os.name != "nt":  # FastF1 uses per-user AppData isolation on Windows.
        assert path.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("renew_ok", [True, False])
def test_rejected_synthetic_token_has_bounded_reauthentication(tmp_path, auth_types, monkeypatch, renew_ok):
    import requests
    from requests.adapters import HTTPAdapter
    from fastf1.internals.f1account.transport import LivetimingAuthAdapter

    TokenManager, SubscriptionToken, F1AuthError = auth_types
    renewals, sends = [], []
    def authenticate():
        renewals.append(True)
        if not renew_ok:
            raise F1AuthError("Synthetic reauthentication unavailable")
        return SubscriptionToken(value="synthetic-renewed-only", expires_at=time.time() + 86400.)
    manager = TokenManager(auth_func=authenticate, path=tmp_path / "synthetic-auth.json")
    manager.store(SubscriptionToken(value="synthetic-rejected-only", expires_at=time.time() + 86400.))
    def send(adapter, request, **kwargs):
        sends.append("Authorization" in request.headers)
        response = requests.Response()
        response.status_code = 401 if len(sends) == 1 else 200
        response._content = b""
        response._content_consumed = True
        response.url = "https://livetiming.formula1.com/static/synthetic.json"
        return response
    monkeypatch.setattr(HTTPAdapter, "send", send)
    original = requests.Request("GET", "https://livetiming.formula1.com/static/synthetic.json").prepare()
    response = LivetimingAuthAdapter(manager).send(original)
    assert len(renewals) == 1
    assert len(sends) == (2 if renew_ok else 1)
    assert response.status_code == (200 if renew_ok else 401)
    assert "Authorization" not in original.headers
