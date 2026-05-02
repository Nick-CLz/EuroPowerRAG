"""Tests for anthropic_client.py."""

import os
from unittest.mock import MagicMock, patch

import pytest


def test_missing_api_key_raises_environment_error(monkeypatch):
    """Should raise EnvironmentError when ANTHROPIC_API_KEY is missing."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # Reset singleton
    import src.utils.anthropic_client as ac
    ac._client = None
    
    with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
        ac._get_client()


def test_placeholder_api_key_raises(monkeypatch):
    """Should reject placeholder keys starting with 'your_'."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "your_key_here")
    import src.utils.anthropic_client as ac
    ac._client = None
    
    with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
        ac._get_client()


def test_singleton_reuses_client(monkeypatch):
    """Calling _get_client() twice should return the same object."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-valid-key")
    import src.utils.anthropic_client as ac
    ac._client = None
    
    with patch("src.utils.anthropic_client.Anthropic") as MockAnthro:
        mock_instance = MagicMock()
        MockAnthro.return_value = mock_instance
        
        client1 = ac._get_client()
        client2 = ac._get_client()
        
        assert client1 is client2
        MockAnthro.assert_called_once()  # Only instantiated once
    
    ac._client = None  # Clean up
