"""Multi-model discussion orchestrator.

Manages multi-round discussions where AI models can see and respond
to each other's answers, producing a natural debate-style conversation.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import AsyncGenerator, Callable, Dict, List, Optional, Tuple

from btcdump.web.llm import LLMManager, LLMProvider, PROVIDER_DISPLAY

logger = logging.getLogger(__name__)


@dataclass
class DiscussionMessage:
    provider: str
    model: str
    round: int
    content: str
    display_name: str = ""
    color: str = ""


def _build_market_context(signal_data: Dict) -> str:
    """Format signal data as a clear market briefing for LLMs."""
    if not signal_data:
        return "No market data available."

    s = signal_data
    lines = [
        f"CURRENT {signal_data.get('symbol', 'BTCUSDT').replace('USDT', '/USDT')} MARKET STATE",
        "=" * 40,
        f"Price:          ${s.get('current_price', 0):,.2f}",
        f"AI Prediction:  ${s.get('predicted_price', 0):,.2f} ({s.get('change_pct', 0):+.2f}%)",
        f"Signal:         {s.get('direction', 'N/A')} (Confidence: {s.get('confidence', 0):.1f}%)",
        "",
        "TECHNICAL INDICATORS",
        f"RSI(14):        {s.get('rsi', 'N/A')}",
        f"MACD:           {'bullish (above signal)' if s.get('macd_bullish') else 'bearish (below signal)'}",
        f"Stochastic:     {s.get('stoch_k', 'N/A')}",
        f"ADX:            {s.get('adx', 'N/A')}",
        f"ATR:            ${s.get('atr', 0):,.2f}",
        f"Volume Ratio:   {s.get('volume_ratio', 'N/A')}x",
        "",
        "ML MODEL DETAILS",
        f"Ensemble MAPE:  {s.get('mape', 0):.2f}%",
        f"Model Agreement:{s.get('model_agreement', 0):.0%}",
        f"Confluence:     {s.get('indicator_confluence', 0)}/5 indicators confirming",
        f"Risk/Reward:    {s.get('risk_reward', 0):.2f}",
    ]
    return "\n".join(lines)


SYSTEM_PROMPT_TEMPLATE = """You are {display_name}, participating in a multi-AI discussion about Bitcoin trading analysis.

RULES:
- Be concise and specific (max 150 words per response)
- Reference specific indicator values from the market data
- When responding to other models, reference them by name
- Be honest about uncertainty
- You can change your mind if another model makes a good point

{market_context}"""

ROUND_2_PLUS_TEMPLATE = """The user asked: "{question}"

Here's what the other AI models said in the previous round:

{previous_responses}

Now respond to their points. You can:
- Agree and add nuance ("I agree with [Model]'s point about RSI because...")
- Disagree and explain ("I disagree with [Model] here because...")
- Highlight something others missed
- Change your position if convinced ("[Model] made me reconsider...")

Keep it concise and natural. Be specific."""


class DiscussionEngine:
    """Orchestrates multi-round AI discussions."""

    def __init__(self, llm_manager: LLMManager) -> None:
        self._llm = llm_manager
        self._history: List[DiscussionMessage] = []

    async def run_discussion(
        self,
        question: str,
        signal_data: Dict,
        num_rounds: int = 3,
        on_chunk: Optional[Callable] = None,
    ) -> List[DiscussionMessage]:
        """Run a full multi-round discussion.

        Args:
            question: User's question.
            signal_data: Current market/signal data.
            num_rounds: Number of discussion rounds.
            on_chunk: Async callback(provider, round, chunk, done) for streaming.

        Returns:
            List of all discussion messages.
        """
        providers = self._llm.get_active_providers()
        if not providers:
            if on_chunk:
                await on_chunk("system", 0, "No AI models are configured. Add API keys in Settings.", True)
            return []

        market_context = _build_market_context(signal_data)
        all_messages: List[DiscussionMessage] = []

        for round_num in range(1, num_rounds + 1):
            if on_chunk:
                await on_chunk("system", round_num, f"__ROUND_START__{round_num}", True)

            round_messages = await self._run_round(
                question=question,
                market_context=market_context,
                round_num=round_num,
                previous_messages=all_messages,
                providers=providers,
                on_chunk=on_chunk,
            )
            all_messages.extend(round_messages)

        self._history.extend(all_messages)
        return all_messages

    async def _run_round(
        self,
        question: str,
        market_context: str,
        round_num: int,
        previous_messages: List[DiscussionMessage],
        providers: Dict[str, LLMProvider],
        on_chunk: Optional[Callable],
    ) -> List[DiscussionMessage]:
        """Run one round of discussion with all providers in parallel."""

        async def _get_response(
            provider_name: str, provider: LLMProvider,
        ) -> DiscussionMessage:
            display = PROVIDER_DISPLAY.get(provider_name, {})
            display_name = display.get("name", provider_name)

            system = SYSTEM_PROMPT_TEMPLATE.format(
                display_name=display_name,
                market_context=market_context,
            )

            messages = [{"role": "system", "content": system}]

            if round_num == 1:
                messages.append({"role": "user", "content": question})
            else:
                # Build context with previous responses
                prev_texts = []
                for msg in previous_messages:
                    if msg.round == round_num - 1:
                        prev_texts.append(f"[{msg.display_name}]: {msg.content}")

                round_prompt = ROUND_2_PLUS_TEMPLATE.format(
                    question=question,
                    previous_responses="\n\n".join(prev_texts),
                )
                messages.append({"role": "user", "content": round_prompt})

            # Stream response
            full_content = []
            try:
                async for chunk in provider.complete_stream(messages):
                    full_content.append(chunk)
                    if on_chunk:
                        await on_chunk(provider_name, round_num, chunk, False)
            except Exception as e:
                error_msg = f"[Error: {e}]"
                full_content.append(error_msg)
                if on_chunk:
                    await on_chunk(provider_name, round_num, error_msg, False)

            if on_chunk:
                await on_chunk(provider_name, round_num, "", True)

            return DiscussionMessage(
                provider=provider_name,
                model=provider.model,
                round=round_num,
                content="".join(full_content),
                display_name=display_name,
                color=display.get("color", "#888"),
            )

        # Run all providers in parallel
        tasks = [
            _get_response(name, prov)
            for name, prov in providers.items()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        messages = []
        for r in results:
            if isinstance(r, DiscussionMessage):
                messages.append(r)
            else:
                logger.error("Discussion round error: %s", r)

        return messages

    @property
    def history(self) -> List[DiscussionMessage]:
        return list(self._history)

    def clear_history(self) -> None:
        self._history.clear()
