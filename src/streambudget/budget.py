from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass

from .config import BudgetConfig, Prices
from .trace import Trace


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Ticket:
    number: int
    quote_usd: float | None
    active: bool = True


class Ledger:
    """Reserve estimated cost before each attempt, reconcile usage after.

    Estimated input tokens cannot guarantee an invoice cap for every provider.
    Unknown/failed usage remains provisionally charged, never silently free.
    """

    def __init__(self, config: BudgetConfig, trace: Trace):
        self.config, self.trace = config, trace
        self.requests = 0
        self.reported_usd = 0.0
        self.provisional_usd = 0.0
        self.reserved_usd = 0.0
        self.unpriced_attempts = 0
        self.unknown_attempts = 0
        self.input_tokens = self.output_tokens = self.cached_input_tokens = 0
        self._lock = asyncio.Lock()

    async def reserve(self, quote: float | None) -> Ticket:
        if quote is not None and (not math.isfinite(quote) or quote < 0):
            raise ValueError("Cost reservation must be finite and nonnegative")
        async with self._lock:
            if self.requests >= self.config.max_requests:
                raise BudgetExceeded("Request-attempt budget exhausted")
            if self.config.max_usd is not None:
                if quote is None:
                    raise BudgetExceeded("Cannot enforce a dollar budget without configured prices")
                if self.reported_usd + self.provisional_usd + self.reserved_usd + quote > self.config.max_usd:
                    raise BudgetExceeded("Estimated dollar budget exhausted")
            self.requests += 1
            self.reserved_usd += quote or 0
            return Ticket(self.requests, quote)

    async def settle(self, ticket: Ticket, *, role: str, prices: Prices,
                     usage: dict | None, status: str, synthetic: bool = False,
                     measured_media_usd: float | None = None,
                     provider_cost_usd: float | None = None) -> None:
        async with self._lock:
            if not ticket.active:
                raise RuntimeError("Budget ticket settled twice")
            ticket.active = False
            self.reserved_usd -= ticket.quote_usd or 0
            # Provider usage is an untrusted schema. Invalid/missing fields do not
            # become zero-dollar usage and must not leave an unsettled reservation.
            usage_valid = False
            normalized_usage = None
            if isinstance(usage, dict):
                try:
                    inp, out = usage["prompt_tokens"], usage["completion_tokens"]
                    details = usage.get("prompt_tokens_details") or {}
                    cached = details.get("cached_tokens", 0)
                    if not all(isinstance(v, int) and not isinstance(v, bool) and v >= 0
                               for v in (inp, out, cached)) or cached > inp:
                        raise ValueError("Invalid token counts")
                    normalized_usage = {"prompt_tokens": inp, "completion_tokens": out,
                                        "prompt_tokens_details": {"cached_tokens": cached}}
                    usage_valid = True
                except (KeyError, AttributeError, TypeError, ValueError):
                    pass
            provider_cost_valid = (isinstance(provider_cost_usd, (int, float))
                                   and not isinstance(provider_cost_usd, bool)
                                   and math.isfinite(provider_cost_usd) and provider_cost_usd >= 0)
            reported = None
            if synthetic:
                reported = 0.0
            elif measured_media_usd is not None:
                reported = measured_media_usd
            elif usage_valid:
                self.input_tokens += inp
                self.output_tokens += out
                self.cached_input_tokens += cached
                if prices.known:
                    cp = prices.cached_input_per_million
                    cp = prices.input_per_million if cp is None else cp
                    reported = ((inp - cached) * prices.input_per_million
                                + cached * cp + out * prices.output_per_million) / 1e6
            if provider_cost_valid and not synthetic and measured_media_usd is None:
                # Aggregator routing can change prices relative to the catalog.
                # Use its request charge when present; this is not an invoice audit.
                reported = provider_cost_usd
            if reported is None:
                if ticket.quote_usd is None:
                    self.unpriced_attempts += 1
                else:
                    self.provisional_usd += ticket.quote_usd
                if not usage_valid:
                    self.unknown_attempts += 1
            else:
                self.reported_usd += reported
            self.trace.emit("model_attempt", attempt=ticket.number, role=role, status=status,
                            synthetic=synthetic, provider_usage=normalized_usage, reported_usd=reported,
                            provider_usage_invalid=usage is not None and not usage_valid,
                            estimated_reservation_usd=ticket.quote_usd,
                            billing_basis=("media_duration_and_configured_rate" if measured_media_usd is not None
                                           else "provider_reported_cost" if provider_cost_valid else "tokens"),
                            provider_cost_invalid=provider_cost_usd is not None and not provider_cost_valid,
                            billing_uncertain=not synthetic and not usage_valid and measured_media_usd is None)
            if self.config.max_usd is not None and self.reported_usd + self.provisional_usd > self.config.max_usd:
                self.trace.emit("budget_overrun", reason="Actual/provisional usage exceeded estimated reservation")

    def summary(self) -> dict:
        return {"request_attempts": self.requests, "reported_usd": self.reported_usd,
                "complete_reported_usd": self.reported_usd if not (self.unpriced_attempts or self.unknown_attempts or self.reserved_usd > 1e-12) else None,
                "provisional_usd": self.provisional_usd, "reserved_usd": max(0, self.reserved_usd),
                "unpriced_attempts": self.unpriced_attempts, "unknown_usage_attempts": self.unknown_attempts,
                "provider_input_tokens": self.input_tokens, "provider_output_tokens": self.output_tokens,
                "provider_cached_input_tokens": self.cached_input_tokens,
                "gpu_seconds": None, "gpu_measurement": "not_measured",
                "cost_note": "Reported dollars use validated provider costs when supported, otherwise configured rates and usage; not invoice reconciliation."}
