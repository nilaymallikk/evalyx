"""Hermetic tests for connection configuration, request mapping, and
response extraction (Phase 15 Steps 2, 4, 5, 18)."""

import pytest
from pydantic import ValidationError

from evalyx.application.connection import (
    AuthConfig,
    ConnectionConfig,
    RequestMapping,
    build_request_body,
    extract_answer,
)
from evalyx.application.ssrf import FORBIDDEN_HEADER_NAMES

PUBLIC = "https://93.184.216.34/v1/chat"


def _connection(**overrides) -> ConnectionConfig:
    values = {"endpoint": PUBLIC}
    values.update(overrides)
    return ConnectionConfig(**values)


# -- endpoint validation ------------------------------------------------------


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://localhost/v1/chat",
        "http://127.0.0.1/v1/chat",
        "http://169.254.169.254/latest/meta-data",
        "ftp://93.184.216.34/x",
        "https://user:pass@93.184.216.34/x",
        "not a url",
    ],
)
def test_endpoint_blocked(endpoint: str):
    with pytest.raises(ValidationError):
        _connection(endpoint=endpoint)


def test_valid_connection_defaults():
    config = _connection()
    assert config.method == "POST"
    assert config.auth.type == "none"
    assert config.response_path == "answer"
    assert config.timeout_seconds == 30.0
    assert config.max_attempts == 3


def test_timeout_and_retry_bounds():
    with pytest.raises(ValidationError):
        _connection(timeout_seconds=0.5)
    with pytest.raises(ValidationError):
        _connection(timeout_seconds=500.0)
    with pytest.raises(ValidationError):
        _connection(max_attempts=50)


# -- auth configuration -------------------------------------------------------


def test_bearer_and_api_key_modes():
    assert _connection(auth={"type": "bearer"}).auth.requires_secret
    assert _connection(auth={"type": "api_key"}).auth.header_name == "X-API-Key"
    assert not _connection().auth.requires_secret


def test_api_key_header_validation():
    with pytest.raises(ValidationError):
        _connection(auth={"type": "api_key", "header_name": "Host"})
    with pytest.raises(ValidationError):
        _connection(auth={"type": "api_key", "header_name": "Bad Header!"})
    with pytest.raises(ValidationError):
        _connection(auth={"type": "api_key", "header_name": "Authorization"})
    with pytest.raises(ValidationError):
        # header_name only meaningful for api_key mode
        _connection(auth={"type": "none", "header_name": "X-Custom"})


# -- headers -------------------------------------------------------------------


def test_security_sensitive_headers_rejected():
    for name in FORBIDDEN_HEADER_NAMES:
        with pytest.raises(ValidationError):
            _connection(headers={name: "override"})


def test_custom_headers_accepted():
    config = _connection(headers={"X-Tenant": "team-1"})
    assert config.headers["X-Tenant"] == "team-1"


def test_header_conflicting_with_auth_header_rejected():
    with pytest.raises(ValidationError):
        _connection(
            auth={"type": "api_key", "header_name": "X-API-Key"},
            headers={"X-Api-Key": "static"},
        )


# -- request mapping -----------------------------------------------------------


def test_field_mode_default():
    body = build_request_body(RequestMapping(), "What is supervised learning?")
    assert body == {"input": "What is supervised learning?"}


def test_field_mode_custom_field_and_extras():
    mapping = RequestMapping(input_field="question", extra_fields={"user_id": "u-1"})
    body = build_request_body(mapping, "hi")
    assert body == {"user_id": "u-1", "question": "hi"}


def test_template_mode_substitution():
    mapping = RequestMapping(
        mode="template", body_template={"question": "{{input}}", "top_k": 3}
    )
    body = build_request_body(mapping, "hello world")
    assert body == {"question": "hello world", "top_k": 3}


def test_template_nested_substitution():
    mapping = RequestMapping(
        mode="template",
        body_template={"messages": [{"role": "user", "content": "{{input}}"}]},
    )
    body = build_request_body(mapping, "hi")
    assert body["messages"][0]["content"] == "hi"


def test_template_unknown_variable_rejected():
    with pytest.raises(ValidationError):
        RequestMapping(mode="template", body_template={"q": "{{user_secret}}"})


def test_template_without_input_rejected():
    with pytest.raises(ValidationError):
        RequestMapping(mode="template", body_template={"q": "static"})


def test_template_requires_dict_in_template_mode():
    with pytest.raises(ValidationError):
        RequestMapping(mode="template")


def test_template_rejected_in_field_mode():
    with pytest.raises(ValidationError):
        RequestMapping(body_template={"q": "{{input}}"})


# -- response extraction --------------------------------------------------------


def test_extract_flat():
    assert extract_answer({"answer": "supervised"}, "answer") == "supervised"


def test_extract_nested():
    body = {"data": {"response": "ok"}}
    assert extract_answer(body, "data.response") == "ok"


def test_extract_list_index():
    body = {"choices": [{"message": {"content": "hello"}}]}
    assert extract_answer(body, "choices.0.message.content") == "hello"


@pytest.mark.parametrize(
    "body,path",
    [
        ({"answer": "x"}, "missing"),
        ({"data": {}}, "data.response"),
        ({"choices": []}, "choices.0.message.content"),
        ({"answer": ""}, "answer"),
        ({"answer": 42}, "answer"),
        ("a plain string", "answer"),
        (None, "answer"),
    ],
)
def test_extraction_failures(body, path):
    from evalyx.application.connection import ResponseExtractionError

    with pytest.raises(ResponseExtractionError):
        extract_answer(body, path)


def test_response_path_validation():
    with pytest.raises(ValidationError):
        _connection(response_path="with spaces")
    with pytest.raises(ValidationError):
        _connection(response_path="a." * 20 + "end")  # too many segments
    with pytest.raises(ValidationError):
        _connection(response_path="")


def test_auth_config_defaults_valid():
    assert AuthConfig().type == "none"