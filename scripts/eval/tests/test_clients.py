from unittest.mock import MagicMock, patch
import pytest
from scripts.eval.clients import ChatClient, BackendSpec


def _mock_response(content: str):
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content=content))]
    resp.usage = MagicMock(total_tokens=10, prompt_tokens=5, completion_tokens=5)
    return resp


def test_chat_client_dashscope_uses_correct_base_url():
    spec = BackendSpec(name="dashscope", model="qwen3-8b")
    with patch("scripts.eval.clients.OpenAI") as MockOpenAI:
        MockOpenAI.return_value.chat.completions.create.return_value = _mock_response("\\boxed{A}")
        client = ChatClient(spec, api_key="fake-key")
        client.chat([{"role": "user", "content": "hi"}], max_tokens=8)
        MockOpenAI.assert_called_with(
            api_key="fake-key",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )


def test_chat_client_deepseek_uses_correct_base_url():
    spec = BackendSpec(name="deepseek", model="deepseek-v4-pro")
    with patch("scripts.eval.clients.OpenAI") as MockOpenAI:
        MockOpenAI.return_value.chat.completions.create.return_value = _mock_response("hi")
        client = ChatClient(spec, api_key="fake-key")
        client.chat([{"role": "user", "content": "x"}], max_tokens=4)
        MockOpenAI.assert_called_with(api_key="fake-key", base_url="https://api.deepseek.com")


def test_chat_client_local_vllm_uses_provided_base_url():
    spec = BackendSpec(name="openai", model="qwen3-8b-tom",
                       base_url="http://localhost:8000/v1")
    with patch("scripts.eval.clients.OpenAI") as MockOpenAI:
        MockOpenAI.return_value.chat.completions.create.return_value = _mock_response("ok")
        client = ChatClient(spec, api_key="dummy")
        client.chat([{"role": "user", "content": "x"}], max_tokens=4)
        MockOpenAI.assert_called_with(api_key="dummy", base_url="http://localhost:8000/v1")


def test_chat_client_passes_extra_body_for_thinking():
    spec = BackendSpec(name="dashscope", model="qwen3-8b",
                       extra_body={"enable_thinking": False})
    with patch("scripts.eval.clients.OpenAI") as MockOpenAI:
        MockOpenAI.return_value.chat.completions.create.return_value = _mock_response("x")
        client = ChatClient(spec, api_key="fake")
        client.chat([{"role": "user", "content": "x"}], max_tokens=4)
        kwargs = MockOpenAI.return_value.chat.completions.create.call_args.kwargs
        assert kwargs["extra_body"] == {"enable_thinking": False}


def test_chat_client_retries_on_failure():
    spec = BackendSpec(name="deepseek", model="deepseek-v4-pro")
    with patch("scripts.eval.clients.OpenAI") as MockOpenAI:
        mock_create = MockOpenAI.return_value.chat.completions.create
        mock_create.side_effect = [
            RuntimeError("transient"),
            RuntimeError("transient"),
            _mock_response("\\boxed{B}"),
        ]
        client = ChatClient(spec, api_key="fake", max_retries=3)
        result = client.chat([{"role": "user", "content": "x"}], max_tokens=4)
        assert result.content == "\\boxed{B}"
        assert mock_create.call_count == 3


def test_chat_client_raises_after_max_retries():
    spec = BackendSpec(name="deepseek", model="deepseek-v4-pro")
    with patch("scripts.eval.clients.OpenAI") as MockOpenAI:
        MockOpenAI.return_value.chat.completions.create.side_effect = RuntimeError("dead")
        client = ChatClient(spec, api_key="fake", max_retries=2)
        with pytest.raises(RuntimeError):
            client.chat([{"role": "user", "content": "x"}], max_tokens=4)
