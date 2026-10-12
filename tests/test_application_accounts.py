from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from mcp_email_server.adapters.metadata import LocalMetadataBackend
from mcp_email_server.application import accounts as accounts_module
from mcp_email_server.application import limits as limits_module
from mcp_email_server.application.accounts import (
    EffectiveAccountQueryService,
    EffectiveConfiguration,
    EffectiveConfigurationLimitError,
    EffectiveConfigurationQueryService,
)
from mcp_email_server.config import AccountAttributes, EmailServer, EmailSettings, ProviderSettings


def _account() -> EmailSettings:
    return EmailSettings(
        account_name="work",
        full_name="Work User",
        email_address="user@example.test",
        incoming=EmailServer(
            user_name="user@example.test",
            password=SecretStr("secret"),
            host="imap.example.test",
            port=993,
        ),
    )


def test_effective_account_query_service_uses_injected_source() -> None:
    source = MagicMock()
    source.list_effective_accounts.return_value = [_account().masked()]

    accounts = EffectiveAccountQueryService(source).execute()

    assert accounts[0].account_name == "work"
    source.list_effective_accounts.assert_called_once_with()


def test_effective_account_discovery_projects_stable_non_secret_capabilities() -> None:
    source = MagicMock()
    account = _account().model_copy(update={"outgoing": _account().incoming})
    source.list_effective_accounts.return_value = [account]
    service = EffectiveAccountQueryService(source)

    discovered = service.discover()
    selected = service.discover_one("work")

    assert discovered == [selected]
    assert discovered[0].model_dump(mode="json") == {
        "account_name": "work",
        "account_type": "email",
        "description": "",
        "email_address": "user@example.test",
        "can_receive": True,
        "can_send": True,
    }
    assert "incoming" not in discovered[0].model_dump()
    assert "updated_at" not in discovered[0].model_dump()


def test_effective_account_discovery_rejects_oversized_description(monkeypatch) -> None:
    source = MagicMock()
    source.list_effective_accounts.return_value = [_account().model_copy(update={"description": "éé"})]
    monkeypatch.setattr(
        accounts_module,
        "APPLICATION_LIMITS",
        replace(accounts_module.APPLICATION_LIMITS, account_description_bytes=3),
    )

    with pytest.raises(EffectiveConfigurationLimitError, match="account description"):
        EffectiveAccountQueryService(source).discover()


def test_effective_account_discovery_projects_provider_accounts_without_send_or_receive() -> None:
    source = MagicMock()
    provider = ProviderSettings(account_name="openai", provider_name="openai", api_key="sk-test")
    source.list_effective_accounts.return_value = [provider]

    discovered = EffectiveAccountQueryService(source).discover()

    assert discovered[0].model_dump(mode="json") == {
        "account_name": "openai",
        "account_type": "provider",
        "description": "",
        "email_address": None,
        "can_receive": False,
        "can_send": False,
    }


def test_effective_account_discovery_projects_neither_email_nor_provider_as_unknown() -> None:
    """account_type="unknown" is a declared member of AvailableAccount's Literal
    contract, not dead code — any account that is neither EmailSettings nor
    ProviderSettings must still be discoverable, just with no capabilities."""
    source = MagicMock()
    bare_account = AccountAttributes(account_name="mystery")
    source.list_effective_accounts.return_value = [bare_account]

    discovered = EffectiveAccountQueryService(source).discover()

    assert discovered[0].model_dump(mode="json") == {
        "account_name": "mystery",
        "account_type": "unknown",
        "description": "",
        "email_address": None,
        "can_receive": False,
        "can_send": False,
    }


def test_effective_account_query_rejects_count_above_shared_limit(monkeypatch) -> None:
    source = MagicMock()
    source.list_effective_accounts.return_value = [_account().masked(), _account().masked()]
    monkeypatch.setattr(
        accounts_module,
        "APPLICATION_LIMITS",
        replace(accounts_module.APPLICATION_LIMITS, configured_accounts=1),
    )

    with pytest.raises(EffectiveConfigurationLimitError, match="limit_exceeded"):
        EffectiveAccountQueryService(source).execute()


def test_effective_account_query_rejects_oversized_canonical_result(monkeypatch) -> None:
    source = MagicMock()
    source.list_effective_accounts.return_value = [_account().masked()]
    monkeypatch.setattr(
        limits_module,
        "APPLICATION_LIMITS",
        replace(limits_module.APPLICATION_LIMITS, serialized_response_bytes=1),
    )

    with pytest.raises(EffectiveConfigurationLimitError, match="limit_exceeded"):
        EffectiveAccountQueryService(source).execute()


def test_effective_account_discovery_rejects_oversized_canonical_result(monkeypatch) -> None:
    """discover()'s own AvailableAccount-projection size check is independent of
    execute()'s earlier AccountAttributes-size check; it must still reject an
    oversized discovery payload even when the input accounts passed execute()."""
    source = MagicMock()
    source.list_effective_accounts.return_value = [_account().masked()]
    monkeypatch.setattr(accounts_module, "validate_serialized_result", MagicMock(side_effect=[None, ValueError]))

    with pytest.raises(EffectiveConfigurationLimitError, match="account discovery result"):
        EffectiveAccountQueryService(source).discover()


def test_effective_configuration_query_returns_configuration_within_limits() -> None:
    source = MagicMock()
    expected = EffectiveConfiguration(
        accounts=(_account(),),
        allowed_recipients=("first@example.test",),
        allowed_senders=(),
    )
    source.effective_configuration.return_value = expected

    configuration = EffectiveConfigurationQueryService(source).execute()

    assert configuration == expected


def test_effective_configuration_rejects_oversized_canonical_result(monkeypatch) -> None:
    source = MagicMock()
    source.effective_configuration.return_value = EffectiveConfiguration(
        accounts=(),
        allowed_recipients=("first@example.test",),
        allowed_senders=(),
    )
    monkeypatch.setattr(
        limits_module,
        "APPLICATION_LIMITS",
        replace(limits_module.APPLICATION_LIMITS, serialized_response_bytes=1),
    )

    with pytest.raises(EffectiveConfigurationLimitError, match="effective configuration result"):
        EffectiveConfigurationQueryService(source).execute()


def test_effective_configuration_rejects_policy_count_above_shared_limit(monkeypatch) -> None:
    source = MagicMock()
    source.effective_configuration.return_value = EffectiveConfiguration(
        accounts=(),
        allowed_recipients=("first@example.test", "second@example.test"),
        allowed_senders=(),
    )
    monkeypatch.setattr(
        accounts_module,
        "APPLICATION_LIMITS",
        replace(accounts_module.APPLICATION_LIMITS, policy_entries=1),
    )

    with pytest.raises(EffectiveConfigurationLimitError, match="limit_exceeded"):
        EffectiveConfigurationQueryService(source).execute()


@pytest.mark.parametrize("mode, reload", [("legacy", False), ("managed", True)])
def test_local_account_adapter_uses_selected_mode_authority_and_masks_secrets(mode: str, reload: bool) -> None:
    settings = MagicMock()
    settings.get_accounts.return_value = [_account()]
    with (
        patch("mcp_email_server.adapters.metadata.process_bootstrap", return_value=SimpleNamespace(mode=mode)),
        patch("mcp_email_server.adapters.metadata.get_settings", return_value=settings) as get_settings,
    ):
        accounts = LocalMetadataBackend().list_effective_accounts()

    assert isinstance(accounts[0], EmailSettings)
    assert accounts[0].account_name == "work"
    assert accounts[0].incoming.password.get_secret_value() == "********"
    get_settings.assert_called_once_with(reload=reload)
