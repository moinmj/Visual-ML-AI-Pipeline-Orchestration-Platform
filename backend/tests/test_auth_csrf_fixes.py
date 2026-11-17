import pytest
from unittest.mock import patch, MagicMock
from fastapi import Request, HTTPException
from backend.app.core.config import settings
from backend.app.core.security import (
    get_current_user,
    resolve_session,
    _extract_sid,
    _csrf_token_for,
    TokenData,
    _load_session,
)


@pytest.mark.asyncio
async def test_extract_sid_prioritizes_header_over_cookie():
    # Case 1: Header provided -> from_cookie must be False
    scope = {
        "type": "http",
        "headers": [(b"x-session-id", b"test-header-sid")],
        "query_string": b"",
    }
    req = Request(scope)
    sid, from_cookie = _extract_sid(req)
    assert sid == "test-header-sid"
    assert from_cookie is False

    # Case 2: Only cookie provided -> from_cookie must be True
    scope_cookie = {
        "type": "http",
        "headers": [(b"cookie", b"sid=test-cookie-sid")],
        "query_string": b"",
    }
    req_cookie = Request(scope_cookie)
    sid_c, from_cookie_c = _extract_sid(req_cookie)
    assert sid_c == "test-cookie-sid"
    assert from_cookie_c is True


@pytest.mark.asyncio
async def test_dev_session_resolves_in_development_mode_when_bypass_enabled():
    with patch.object(settings, "ENVIRONMENT", "development"), patch.object(settings, "DEV_AUTH_BYPASS", True):
        user = await resolve_session("dev")
        assert user is not None
        assert user.sub == "1"
        assert user.tenant_id == 1
        assert "Super Admin" in user.roles
        assert user.has_role("Super Admin")
        assert user.has_permission("*")


@pytest.mark.asyncio
async def test_dev_session_rejected_in_production():
    # In production, DEV_AUTH_BYPASS is False and ENVIRONMENT is production -> 'dev' token cannot bypass
    with patch.object(settings, "ENVIRONMENT", "production"), patch.object(settings, "DEV_AUTH_BYPASS", False):
        with patch("backend.app.core.security._um_sm", side_effect=Exception("UM DB Offline")):
            with pytest.raises(HTTPException):
                await resolve_session("dev")


@pytest.mark.asyncio
async def test_header_auth_skips_csrf_check():
    # A POST request with X-Session-Id header (explicit API client) must succeed without CSRF token
    with patch.object(settings, "ENVIRONMENT", "development"), patch.object(settings, "DEV_AUTH_BYPASS", True):
        with patch.object(settings, "SESSION_CSRF_ENABLED", True):
            scope = {
                "type": "http",
                "method": "POST",
                "headers": [
                    (b"x-session-id", b"dev"),
                    (b"cookie", b"sid=dev"),  # browser may send cookie too
                ],
                "query_string": b"",
            }
            req = Request(scope)
            user = await get_current_user(request=req)
            assert user is not None
            assert user.sub == "1"


@pytest.mark.asyncio
async def test_swagger_docs_referer_exempt_from_csrf():
    # Even if authenticated via cookie only, Swagger UI (/docs) requests are exempt from CSRF
    with patch.object(settings, "ENVIRONMENT", "development"), patch.object(settings, "DEV_AUTH_BYPASS", True):
        with patch.object(settings, "SESSION_CSRF_ENABLED", True):
            scope = {
                "type": "http",
                "method": "POST",
                "headers": [
                    (b"cookie", b"sid=dev"),
                    (b"referer", b"https://ml-crt.aiiotgeeks.com/docs"),
                ],
                "query_string": b"",
            }
            req = Request(scope)
            user = await get_current_user(request=req)
            assert user is not None
            assert user.sub == "1"


@pytest.mark.asyncio
async def test_cross_site_cookie_post_requires_csrf():
    # An untrusted cross-origin request using cookie must fail if CSRF token is missing
    with patch.object(settings, "ENVIRONMENT", "development"), patch.object(settings, "DEV_AUTH_BYPASS", True):
        with patch.object(settings, "SESSION_CSRF_ENABLED", True):
            scope = {
                "type": "http",
                "method": "POST",
                "headers": [
                    (b"cookie", b"sid=dev"),
                    (b"sec-fetch-site", b"cross-site"),
                    (b"referer", b"https://malicious-site.com"),
                ],
                "query_string": b"",
            }
            req = Request(scope)
            with pytest.raises(HTTPException) as exc_info:
                await get_current_user(request=req)
            assert exc_info.value.status_code == 403
            assert "Missing or invalid CSRF token" in exc_info.value.detail
