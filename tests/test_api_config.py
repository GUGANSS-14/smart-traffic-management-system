"""
Unit Tests for API Configuration & Secrets Manager
===================================================
Tests 3-tier fallback logic, masking, and environment variable handling.
"""

import os
from unittest.mock import patch
import streamlit as st

from modules.api_config import get_api_key, is_api_key_configured, mask_api_key


def test_get_api_key_empty():
    """When no key is configured in session, secrets, or env, returns None."""
    st.session_state.pop("user_api_key", None)
    st.session_state.pop("api_key_api_key", None)

    with patch.dict(os.environ, {}, clear=True):
        with patch.object(st, "secrets", {}, create=True):
            assert get_api_key("NON_EXISTENT_KEY") is None
            assert is_api_key_configured("NON_EXISTENT_KEY") is False


def test_get_api_key_from_env():
    """Tests retrieval from environment variables."""
    st.session_state.pop("user_api_key", None)
    st.session_state.pop("api_key_test_service", None)

    with patch.dict(os.environ, {"TEST_SERVICE": "env_secret_key_12345"}):
        with patch.object(st, "secrets", {}, create=True):
            assert get_api_key("TEST_SERVICE") == "env_secret_key_12345"
            assert is_api_key_configured("TEST_SERVICE") is True


def test_get_api_key_from_session_state():
    """Tests that user input in session state takes precedence."""
    st.session_state["user_api_key"] = "ui_entered_key_abcde"

    with patch.dict(os.environ, {"API_KEY": "env_fallback_key"}):
        assert get_api_key("API_KEY") == "ui_entered_key_abcde"

    del st.session_state["user_api_key"]


def test_mask_api_key():
    """Tests key masking for UI display."""
    assert mask_api_key(None) == "Not Configured"
    assert mask_api_key("") == "Not Configured"
    assert mask_api_key("short") == "••••••••"
    assert mask_api_key("sk-1234567890abcdef") == "sk-1••••••••cdef"
