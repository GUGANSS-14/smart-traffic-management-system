"""
API Configuration & Secrets Manager
===================================
Provides centralized, secure retrieval of API keys and credentials with a 3-tier fallback:
1. User Interface session state (st.session_state)
2. Streamlit Secrets (.streamlit/secrets.toml or Streamlit Cloud Secrets)
3. Operating System Environment Variables (.env / os.environ)

Security:
- Never prints or logs raw API keys.
- Ensures keys are masked when presented in user interfaces.
"""

import os
from typing import Optional
import streamlit as st


def get_api_key(key_name: str = "API_KEY") -> Optional[str]:
    """
    Retrieves an API key with 3-tier fallback:
    1. UI input in st.session_state (e.g. st.session_state['user_api_key'])
    2. Streamlit native secrets (.streamlit/secrets.toml)
    3. OS environment variable (os.getenv)

    Args:
        key_name: Name of the key to look for (default: 'API_KEY')

    Returns:
        The API key string if found, or None if not configured.
    """
    # Tier 1: Check session state
    session_key = f"api_key_{key_name.lower()}"
    if session_key in st.session_state and st.session_state[session_key]:
        return str(st.session_state[session_key]).strip()
    if "user_api_key" in st.session_state and st.session_state["user_api_key"]:
        return str(st.session_state["user_api_key"]).strip()

    # Tier 2: Check Streamlit secrets
    try:
        if hasattr(st, "secrets") and st.secrets is not None:
            if key_name in st.secrets:
                return str(st.secrets[key_name]).strip()
            for k in st.secrets:
                if k.lower() == key_name.lower():
                    return str(st.secrets[k]).strip()
                if isinstance(st.secrets[k], dict) and key_name.lower() in [sub_k.lower() for sub_k in st.secrets[k]]:
                    return str(st.secrets[k].get(key_name, "")).strip()
    except Exception:
        pass

    # Tier 3: Check OS environment variables
    env_val = os.getenv(key_name) or os.getenv(key_name.upper()) or os.getenv(key_name.lower())
    if env_val:
        return env_val.strip()

    return None


def is_api_key_configured(key_name: str = "API_KEY") -> bool:
    """Checks whether an API key is currently present and non-empty."""
    key = get_api_key(key_name)
    return bool(key and len(key.strip()) > 0)


def mask_api_key(key: Optional[str]) -> str:
    """
    Returns a masked version of the key for safe UI display (e.g., 'sk-12...89ab').
    """
    if not key:
        return "Not Configured"
    key = key.strip()
    if len(key) <= 8:
        return "••••••••"
    return f"{key[:4]}••••••••{key[-4:]}"
