"""Multi-LLM client abstraction for OpenAI, Claude, Grok, and Gemini."""

from __future__ import annotations

import asyncio
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncGenerator, Dict, List, Optional

logger = logging.getLogger(__name__)

PROVIDER_MODELS = {
    "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "o3-mini"],
    "claude": ["claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
    "grok": ["grok-3", "grok-3-mini", "grok-3-fast"],
    "gemini": ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash"],
}

PROVIDER_DISPLAY = {
    "openai": {"name": "OpenAI", "color": "#10a37f", "icon": "O"},
    "claude": {"name": "Claude", "color": "#d4a574", "icon": "C"},
    "grok": {"name": "Grok", "color": "#1da1f2", "icon": "G"},
    "gemini": {"name": "Gemini", "color": "#8e75ff", "icon": "Ge"},
}

PROVIDER_ENV_KEYS = {
    "openai": "OPENAI_API_KEY",
    "claude": "ANTHROPIC_API_KEY",
    "grok": "XAI_API_KEY",
    "gemini": "GOOGLE_API_KEY",
}


@dataclass
class ProviderConfig:
    api_key: str = ""
    model: str = ""
    enabled: bool = False


class LLMProvider(ABC):
    """Base class for LLM providers."""

    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model

    @abstractmethod
    async def complete_stream(
        self, messages: List[Dict[str, str]],
    ) -> AsyncGenerator[str, None]:
        """Stream completion tokens."""
        ...

    async def complete(self, messages: List[Dict[str, str]]) -> str:
        """Non-streaming completion."""
        chunks = []
        async for chunk in self.complete_stream(messages):
            chunks.append(chunk)
        return "".join(chunks)


class OpenAIProvider(LLMProvider):
    async def complete_stream(
        self, messages: List[Dict[str, str]],
    ) -> AsyncGenerator[str, None]:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self.api_key)
        try:
            stream = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                max_tokens=1024,
                temperature=0.7,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content
        except Exception as e:
            logger.error("OpenAI error: %s", e)
            yield f"[OpenAI Error: {e}]"


class ClaudeProvider(LLMProvider):
    """Claude provider: uses CLI subprocess for OAuth, SDK for API keys."""

    def _is_oauth_token(self) -> bool:
        return self.api_key.startswith("sk-ant-oat")

    async def complete_stream(
        self, messages: List[Dict[str, str]],
    ) -> AsyncGenerator[str, None]:
        if self._is_oauth_token():
            async for chunk in self._stream_via_cli(messages):
                yield chunk
        else:
            async for chunk in self._stream_via_sdk(messages):
                yield chunk

    async def _stream_via_cli(
        self, messages: List[Dict[str, str]],
    ) -> AsyncGenerator[str, None]:
        """Use `claude -p` subprocess for OAuth token auth."""
        import json as _json

        # Build a single prompt from messages
        parts = []
        for m in messages:
            if m["role"] == "system":
                parts.append(f"[System instructions]\n{m['content']}\n")
            elif m["role"] == "user":
                parts.append(f"User: {m['content']}")
            elif m["role"] == "assistant":
                parts.append(f"Assistant: {m['content']}")
        prompt = "\n\n".join(parts)

        cmd = [
            "claude", "-p", prompt,
            "--model", self.model,
            "--output-format", "stream-json",
            "--verbose",
            "--max-turns", "1",
        ]

        try:
            proc = await asyncio.subprocess.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            async for line in proc.stdout:
                line = line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    data = _json.loads(line)
                except _json.JSONDecodeError:
                    continue

                # Extract text from assistant message events
                if data.get("type") == "assistant":
                    msg = data.get("message", {})
                    for block in msg.get("content", []):
                        if block.get("type") == "text" and block.get("text"):
                            yield block["text"]

            await proc.wait()
        except FileNotFoundError:
            yield "[Claude Error: 'claude' CLI not found. Install Claude Code first.]"
        except Exception as e:
            logger.error("Claude CLI error: %s", e)
            yield f"[Claude CLI Error: {e}]"

    async def _stream_via_sdk(
        self, messages: List[Dict[str, str]],
    ) -> AsyncGenerator[str, None]:
        """Use Anthropic SDK for standard API key auth."""
        from anthropic import AsyncAnthropic

        client = AsyncAnthropic(api_key=self.api_key, auth_token=None)

        system = ""
        conv_messages = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                conv_messages.append(m)

        try:
            async with client.messages.stream(
                model=self.model,
                max_tokens=1024,
                system=system if system else "You are a helpful assistant.",
                messages=conv_messages,
            ) as stream:
                async for text in stream.text_stream:
                    yield text
        except Exception as e:
            logger.error("Claude SDK error: %s", e)
            yield f"[Claude Error: {e}]"


class GrokProvider(LLMProvider):
    """Grok uses an OpenAI-compatible API at api.x.ai."""

    async def complete_stream(
        self, messages: List[Dict[str, str]],
    ) -> AsyncGenerator[str, None]:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=self.api_key,
            base_url="https://api.x.ai/v1",
        )
        try:
            stream = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                max_tokens=1024,
                temperature=0.7,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content
        except Exception as e:
            logger.error("Grok error: %s", e)
            yield f"[Grok Error: {e}]"


class GeminiProvider(LLMProvider):
    async def complete_stream(
        self, messages: List[Dict[str, str]],
    ) -> AsyncGenerator[str, None]:
        import google.generativeai as genai

        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model)

        # Convert messages to Gemini format
        system_text = ""
        history = []
        for m in messages:
            if m["role"] == "system":
                system_text = m["content"]
            elif m["role"] == "user":
                history.append({"role": "user", "parts": [m["content"]]})
            elif m["role"] == "assistant":
                history.append({"role": "model", "parts": [m["content"]]})

        # Prepend system to first user message if exists
        if system_text and history and history[0]["role"] == "user":
            history[0]["parts"][0] = system_text + "\n\n" + history[0]["parts"][0]

        try:
            chat = model.start_chat(history=history[:-1] if len(history) > 1 else [])
            last_msg = history[-1]["parts"][0] if history else ""

            response = await chat.send_message_async(last_msg, stream=True)
            async for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            logger.error("Gemini error: %s", e)
            yield f"[Gemini Error: {e}]"


PROVIDER_CLASSES = {
    "openai": OpenAIProvider,
    "claude": ClaudeProvider,
    "grok": GrokProvider,
    "gemini": GeminiProvider,
}


class LLMManager:
    """Manages all LLM providers and their configurations."""

    def __init__(self) -> None:
        self.configs: Dict[str, ProviderConfig] = {}
        self._load_from_env()

    def _load_from_env(self) -> None:
        """Load API keys from environment variables."""
        for provider, env_key in PROVIDER_ENV_KEYS.items():
            api_key = os.environ.get(env_key, "")

            # Claude: also check CLAUDE_CODE_OAUTH_TOKEN as fallback
            if provider == "claude" and not api_key:
                api_key = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "")

            default_model = PROVIDER_MODELS[provider][0] if PROVIDER_MODELS[provider] else ""
            self.configs[provider] = ProviderConfig(
                api_key=api_key,
                model=default_model,
                enabled=bool(api_key),
            )

    def configure(
        self, provider: str, api_key: str = "", model: str = "", enabled: bool = True,
    ) -> None:
        """Update a provider's configuration."""
        if provider not in PROVIDER_MODELS:
            return
        cfg = self.configs.get(provider, ProviderConfig())
        if api_key:
            cfg.api_key = api_key
        if model:
            cfg.model = model
        cfg.enabled = enabled and bool(cfg.api_key)
        self.configs[provider] = cfg
        logger.info("Configured %s: model=%s, enabled=%s", provider, cfg.model, cfg.enabled)

    def get_provider(self, name: str) -> Optional[LLMProvider]:
        """Create a provider instance if configured and enabled."""
        cfg = self.configs.get(name)
        if not cfg or not cfg.enabled or not cfg.api_key:
            return None
        cls = PROVIDER_CLASSES.get(name)
        if not cls:
            return None
        return cls(api_key=cfg.api_key, model=cfg.model)

    def get_active_providers(self) -> Dict[str, LLMProvider]:
        """Return all active provider instances."""
        result = {}
        for name in PROVIDER_MODELS:
            p = self.get_provider(name)
            if p:
                result[name] = p
        return result

    def get_status(self) -> Dict:
        """Return current status of all providers for the UI."""
        status = {}
        for name in PROVIDER_MODELS:
            cfg = self.configs.get(name, ProviderConfig())
            status[name] = {
                "enabled": cfg.enabled,
                "model": cfg.model,
                "has_key": bool(cfg.api_key),
                "models": PROVIDER_MODELS[name],
                "display": PROVIDER_DISPLAY[name],
            }
        return status
