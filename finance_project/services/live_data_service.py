import datetime as dt
import json
import os
import re
import xml.etree.ElementTree as ET
from typing import Any, Optional
from urllib.parse import quote_plus

import requests

from finance_project.core.llm.client import generate_response
from finance_project.core.logging.logger import log_event
from finance_project.services.live_data_search import search_web


_LIVE_ENABLED = os.getenv("LIVE_DATA_ENABLED", "true").lower() not in {"0", "false", "off", "no"}
_HTTP_TIMEOUT_SECONDS = float(os.getenv("LIVE_DATA_HTTP_TIMEOUT_SECONDS", "5"))
_MAX_HEADLINES = int(os.getenv("LIVE_DATA_MAX_HEADLINES", "3"))
_WEB_FALLBACK_ENABLED = os.getenv("LIVE_DATA_WEB_FALLBACK_ENABLED", "true").lower() not in {
    "0",
    "false",
    "off",
    "no",
}
_NEWS_WEB_ENABLED = os.getenv("LIVE_DATA_NEWS_WEB_ENABLED", "true").lower() not in {
    "0",
    "false",
    "off",
    "no",
}
_CRYPTO_ENABLED = os.getenv("LIVE_DATA_ENABLE_CRYPTO", "true").lower() not in {"0", "false", "off", "no"}
_GOLD_ENABLED = os.getenv("LIVE_DATA_ENABLE_GOLD", "true").lower() not in {"0", "false", "off", "no"}
_FX_ENABLED = os.getenv("LIVE_DATA_ENABLE_FX", "true").lower() not in {"0", "false", "off", "no"}

_ALLOWED_LIVE_KINDS = {
    "weather",
    "stock",
    "stock_history",
    "news",
    "crypto",
    "crypto_history",
    "gold",
    "gold_history",
    "commodity",
    "commodity_history",
    "fx",
}
_PROVIDER_SUCCESS_STATUSES = {"ok", "blocked"}


def _extract_json_object(raw_text: str) -> dict | None:
    text = str(raw_text or "").strip()
    if not text:
        return None

    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None

    try:
        payload = json.loads(match.group(0))
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        return None


def _llm_extract_json(prompt: str, operation: str = "turn_control") -> dict | None:
    try:
        raw = generate_response(prompt, operation=operation)
    except Exception:
        return None
    return _extract_json_object(raw)


def _build_payload(
    *,
    kind: str,
    status: str,
    message: str | None = None,
    facts: Optional[list[str]] = None,
    sources: Optional[list[str]] = None,
    route: str = "structured",
    provider: str = "unknown",
    ambiguity_clarification: bool = False,
    failure_reason: str | None = None,
    provider_attempts: Optional[list[dict[str, str]]] = None,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "status": status,
        "message": message,
        "facts": facts or [],
        "sources": sources or [],
        "as_of": _now_utc_iso(),
        "route": route,
        "provider": provider,
        "ambiguity_clarification": ambiguity_clarification,
        "failure_reason": failure_reason,
        "provider_attempts": provider_attempts or [],
    }


def _log_live_route(query: str, live_data: dict[str, Any]) -> None:
    try:
        log_event(
            event="live_data_route_selected",
            metadata={
                "kind": live_data.get("kind"),
                "status": live_data.get("status"),
                "route": live_data.get("route", "structured"),
                "provider": live_data.get("provider", "unknown"),
                "ambiguity_clarification": bool(live_data.get("ambiguity_clarification")),
                "failure_reason": live_data.get("failure_reason"),
                "provider_attempts_count": len(live_data.get("provider_attempts") or []),
                "provider_attempts": live_data.get("provider_attempts") or [],
                "sources_count": len(live_data.get("sources") or []),
                "query_len": len(str(query or "")),
            },
        )
    except Exception:
        return


def classify_live_data_kind(query: str) -> Optional[str]:
    text = (query or "").strip()
    if not text:
        return None

    prompt = (
        "You classify whether a user asks for live data updates.\n"
        "Classify the current message into exactly one kind: weather, stock, stock_history, news, crypto, crypto_history, gold, fx, none.\n"
        "Support multilingual and transliterated input.\n"
        "Choose weather/stock/crypto/gold/fx/news only for current live data asks.\n"
        "Choose stock_history or crypto_history only for explicit historical performance asks.\n"
        "If the message is educational, conversational, planning, or unclear, choose none.\n\n"
        f"User message: {text}\n\n"
        "Return STRICT JSON only with key 'kind' and value in [weather, stock, stock_history, news, crypto, crypto_history, gold, fx, none]."
    )
    payload = _llm_extract_json(prompt, operation="turn_control")
    if not payload:
        return None

    kind = str(payload.get("kind") or "").strip().lower()
    if kind in _ALLOWED_LIVE_KINDS:
        return kind
    return None


_INR_CONVERT_KINDS = {"gold", "commodity", "crypto"}


def _get_usd_inr_rate() -> float | None:
    """Fetch current USD/INR spot rate from frankfurter."""
    try:
        resp = requests.get(
            "https://api.frankfurter.app/latest",
            params={"from": "USD", "to": "INR"},
            timeout=max(4.0, _HTTP_TIMEOUT_SECONDS),
        )
        resp.raise_for_status()
        rate = (resp.json() or {}).get("rates", {}).get("INR")
        return float(rate) if rate else None
    except Exception:
        return None


def _append_inr_price(payload: dict) -> dict:
    """Add 'Last price INR:' fact to payload when a USD 'Last price:' fact exists."""
    facts = list(payload.get("facts") or [])
    usd_fact = next((f for f in facts if f.startswith("Last price:")), None)
    if not usd_fact:
        return payload
    price_str = usd_fact[len("Last price:"):].strip()
    m = re.match(r"^([0-9]+(?:\.[0-9]*)?)", price_str)
    if not m:
        return payload
    try:
        price_usd = float(m.group(1))
    except ValueError:
        return payload
    rate = _get_usd_inr_rate()
    if not rate:
        return payload
    price_inr = round(price_usd * rate)
    new_facts = list(facts)
    new_facts.append(f"Last price INR: {price_inr:,}")
    if "troy ounce" in price_str.lower():
        price_per_10g_inr = round(price_usd * rate / 31.1035 * 10)
        new_facts.append(f"Last price INR per 10g: {price_per_10g_inr:,}")
    payload = dict(payload)
    payload["facts"] = new_facts
    return payload


def maybe_fetch_live_data(
    query: str,
    profile: Optional[dict] = None,
    live_kind: Optional[str] = None,
    slots: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    if not _LIVE_ENABLED:
        return None

    kind = str(live_kind or "").strip().lower() if live_kind else classify_live_data_kind(query)
    if not kind or kind not in _ALLOWED_LIVE_KINDS:
        return None

    if _looks_unsafe(query):
        blocked = _build_payload(
            kind=kind,
            status="blocked",
            message="I can help with finance and safe informational updates, but I can't help with that request.",
            route="structured",
            provider="safety",
        )
        _log_live_route(query, blocked)
        return blocked

    slots = slots or {}

    structured = _run_structured_provider_chain(
        kind=kind,
        query=query,
        profile=profile or {},
        slots=slots,
    )
    if structured and str(structured.get("status") or "").lower() in _PROVIDER_SUCCESS_STATUSES:
        if kind in _INR_CONVERT_KINDS:
            structured = _append_inr_price(structured)
        _log_live_route(query, structured)
        return structured

    if _WEB_FALLBACK_ENABLED:
        fallback = _fetch_web_fallback(query=query, kind=kind)
        if fallback:
            fallback["provider_attempts"] = list(structured.get("provider_attempts") or []) + list(
                fallback.get("provider_attempts") or []
            )
            _log_live_route(query, fallback)
            return fallback

    if structured:
        _log_live_route(query, structured)
    return structured


def _run_structured_provider_chain(
    kind: str,
    query: str,
    profile: dict,
    slots: dict[str, Any],
) -> dict[str, Any]:
    providers = _provider_registry(kind)
    attempts: list[dict[str, str]] = []
    if not providers:
        return _build_payload(
            kind=kind,
            status="error",
            message="I couldn't find a structured provider for that live data request.",
            route="structured",
            provider="provider_chain",
            failure_reason="provider_unavailable",
            provider_attempts=attempts,
        )

    last_error: dict[str, Any] | None = None
    for provider_name, provider_fn in providers:
        provider_slots = dict(slots)
        provider_slots["_kind"] = kind
        try:
            result = provider_fn(query=query, profile=profile, slots=provider_slots)
        except Exception:
            result = _build_payload(
                kind=kind,
                status="error",
                message="I couldn't fetch that live update right now.",
                route="structured",
                provider=provider_name,
                failure_reason="provider_exception",
            )

        if not result:
            attempts.append({"provider": provider_name, "status": "none"})
            continue

        status = str(result.get("status") or "error").strip().lower()
        attempts.append({"provider": provider_name, "status": status})
        result["route"] = "structured"
        result["provider"] = str(result.get("provider") or provider_name)
        result["provider_attempts"] = list(attempts)

        if status in _PROVIDER_SUCCESS_STATUSES:
            return result

        if status == "error":
            result["failure_reason"] = str(result.get("failure_reason") or "provider_failed")
            last_error = result

    if last_error:
        last_error["provider_attempts"] = list(attempts)
        last_error["failure_reason"] = str(last_error.get("failure_reason") or "provider_failed")
        return last_error

    return _build_payload(
        kind=kind,
        status="error",
        message="I couldn't fetch that live update right now.",
        route="structured",
        provider="provider_chain",
        failure_reason="provider_failed",
        provider_attempts=attempts,
    )


def _provider_registry(kind: str) -> list[tuple[str, Any]]:
    if kind == "weather":
        return [("open_meteo", _provider_weather_open_meteo)]
    if kind == "stock":
        return [("equity_router", _provider_stock_router)]
    if kind == "stock_history":
        return [("yahoo_finance", _provider_stock_history_yahoo), ("stooq_history", _provider_stock_history_stooq)]
    if kind == "crypto":
        return [("coingecko", _provider_crypto_coingecko)] if _CRYPTO_ENABLED else [("feature_flag", _provider_disabled)]
    if kind == "crypto_history":
        return [("coingecko", _provider_crypto_history_coingecko)] if _CRYPTO_ENABLED else [("feature_flag", _provider_disabled)]
    if kind == "gold":
        return [("yahoo_finance", _provider_gold_yahoo), ("stooq", _provider_gold_stooq)] if _GOLD_ENABLED else [("feature_flag", _provider_disabled)]
    if kind == "gold_history":
        return [("stooq_history", _provider_gold_history)] if _GOLD_ENABLED else [("feature_flag", _provider_disabled)]
    if kind == "commodity":
        return [("stooq", _provider_commodity_stooq)]
    if kind == "commodity_history":
        return [("stooq_history", _provider_commodity_history_stooq)]
    if kind == "fx":
        return [("frankfurter", _provider_fx_frankfurter)] if _FX_ENABLED else [("feature_flag", _provider_disabled)]
    if kind == "news":
        providers: list[tuple[str, Any]] = []
        if _NEWS_WEB_ENABLED:
            providers.append(("duckduckgo_news", _provider_news_web))
        providers.append(("google_news_rss", _provider_news_rss))
        return providers
    return []


def _provider_disabled(query: str, profile: dict, slots: dict[str, Any]) -> dict[str, Any]:
    _ = query
    _ = profile
    kind = str(slots.get("_kind") or "live_data")
    label = kind.replace("_", " ")
    return _build_payload(
        kind=kind,
        status="error",
        message=f"{label} live data is disabled right now.",
        provider="feature_flag",
        failure_reason="provider_disabled",
    )


def _provider_weather_open_meteo(query: str, profile: dict, slots: dict[str, Any]) -> dict[str, Any]:
    return _fetch_weather(query=query, profile=profile, location_hint=slots.get("location"))


def _provider_stock_router(query: str, profile: dict, slots: dict[str, Any]) -> dict[str, Any]:
    _ = profile
    return _fetch_stock_quote(
        query=query,
        symbol_hint=slots.get("ticker"),
        company_hint=slots.get("company"),
    )


def _provider_stock_history_yahoo(query: str, profile: dict, slots: dict[str, Any]) -> dict[str, Any]:
    _ = profile
    return _fetch_stock_history(
        query=query,
        symbol_hint=slots.get("ticker"),
        company_hint=slots.get("company"),
        window_days_hint=slots.get("window_days"),
    )


def _provider_stock_history_stooq(query: str, profile: dict, slots: dict[str, Any]) -> dict[str, Any]:
    _ = profile
    return _fetch_stock_history_stooq(
        query=query,
        symbol_hint=slots.get("ticker"),
        company_hint=slots.get("company"),
        window_days_hint=slots.get("window_days"),
    )


def _provider_crypto_coingecko(query: str, profile: dict, slots: dict[str, Any]) -> dict[str, Any]:
    _ = profile
    return _fetch_crypto_quote(query=query, coin_hint=slots.get("coin"))


def _provider_crypto_history_coingecko(query: str, profile: dict, slots: dict[str, Any]) -> dict[str, Any]:
    _ = profile
    return _fetch_crypto_history(
        query=query,
        coin_hint=slots.get("coin"),
        window_days_hint=slots.get("window_days"),
    )


def _provider_gold_yahoo(query: str, profile: dict, slots: dict[str, Any]) -> dict[str, Any]:
    _ = profile
    _ = slots
    return _fetch_gold_spot(query=query)


def _provider_gold_stooq(query: str, profile: dict, slots: dict[str, Any]) -> dict[str, Any]:
    _ = profile
    _ = slots
    return _fetch_gold_spot_stooq(query=query)


def _provider_gold_history(query: str, profile: dict, slots: dict[str, Any]) -> dict[str, Any]:
    _ = profile
    return _fetch_gold_history_stooq(query=query, window_days_hint=slots.get("window_days"))


def _provider_commodity_history_stooq(query: str, profile: dict, slots: dict[str, Any]) -> dict[str, Any]:
    _ = profile
    ticker = _to_text(slots.get("ticker"))
    if not ticker:
        return _build_payload(
            kind="commodity_history",
            status="error",
            message="I couldn't determine the commodity symbol.",
            provider="stooq_history",
            failure_reason="no_ticker",
        )
    return _fetch_commodity_history_stooq(
        query=query, ticker=ticker, window_days_hint=slots.get("window_days")
    )


def _fetch_commodity_history_stooq(
    query: str, ticker: str, window_days_hint: Optional[int] = None
) -> dict[str, Any]:
    symbol = ticker.lower()
    window_days = _normalize_history_window_days(
        window_days_hint or _extract_history_window_days(query), default_days=1095
    )
    end_date = dt.datetime.utcnow().date()
    start_date = end_date - dt.timedelta(days=window_days)

    try:
        resp = requests.get(
            "https://stooq.com/q/d/l/",
            params={"s": symbol, "i": "d"},
            timeout=max(8.0, _HTTP_TIMEOUT_SECONDS),
        )
        resp.raise_for_status()
        lines = [line.strip() for line in (resp.text or "").splitlines() if line.strip()]
        if len(lines) < 3:
            return _build_payload(
                kind="commodity_history",
                status="error",
                message=f"I couldn't fetch historical data for {ticker.upper()}.",
                sources=["https://stooq.com/"],
                provider="stooq_history",
            )

        closes: list[float] = []
        filtered_dates: list[str] = []
        for row in lines[1:]:
            parts = [p.strip() for p in row.split(",")]
            if len(parts) < 5:
                continue
            date_str, close_str = parts[0], parts[4]
            if close_str in {"", "N/D"}:
                continue
            try:
                row_date = dt.datetime.strptime(date_str, "%Y-%m-%d").date()
                close_value = float(close_str)
            except Exception:
                continue
            if row_date < start_date or row_date > end_date:
                continue
            closes.append(close_value)
            filtered_dates.append(row_date.isoformat())

        if len(closes) < 2:
            return _build_payload(
                kind="commodity_history",
                status="error",
                message=f"I couldn't fetch enough historical points for {ticker.upper()}.",
                sources=["https://stooq.com/"],
                provider="stooq_history",
            )

        start_price = closes[0]
        end_price = closes[-1]
        total_return = ((end_price - start_price) / start_price) * 100 if start_price else 0.0
        facts = [
            f"Instrument: {ticker.upper()}",
            f"Period: {filtered_dates[0]} to {filtered_dates[-1]}",
            f"Start price: {round(start_price, 2)}",
            f"End price: {round(end_price, 2)}",
            f"Total return: {round(total_return, 2)}%",
        ]
        return _build_payload(
            kind="commodity_history",
            status="ok",
            facts=facts,
            sources=["https://stooq.com/"],
            provider="stooq_history",
        )
    except Exception:
        return _build_payload(
            kind="commodity_history",
            status="error",
            message=f"I couldn't fetch historical data for {ticker.upper()} right now.",
            sources=["https://stooq.com/"],
            provider="stooq_history",
        )


def _provider_commodity_stooq(query: str, profile: dict, slots: dict[str, Any]) -> dict[str, Any]:
    _ = query
    _ = profile
    ticker = _to_text(slots.get("ticker"))
    if not ticker:
        return _build_payload(
            kind="commodity",
            status="error",
            message="I couldn't determine the commodity symbol.",
            provider="stooq",
            failure_reason="no_ticker",
        )
    symbol = ticker.lower()
    try:
        resp = requests.get(
            "https://stooq.com/q/l/",
            params={"s": symbol, "f": "sd2t2ohlcv", "h": "", "e": "csv"},
            timeout=max(6.0, _HTTP_TIMEOUT_SECONDS),
        )
        resp.raise_for_status()
        lines = [line.strip() for line in (resp.text or "").splitlines() if line.strip()]
        if len(lines) < 2:
            return _build_payload(kind="commodity", status="error", provider="stooq", failure_reason="provider_failed")
        parts = [p.strip() for p in lines[1].split(",")]
        if len(parts) < 7 or parts[6] in {"", "N/D"}:
            return _build_payload(kind="commodity", status="error", provider="stooq", failure_reason="provider_failed")
        out_symbol, date_s, time_s, open_s, high_s, low_s, close_s = parts[:7]
        facts = [
            f"Instrument: {out_symbol}",
            f"Last price: {close_s} USD per troy ounce",
            f"Open/High/Low: {open_s}/{high_s}/{low_s}",
            f"Trade date/time: {date_s} {time_s}",
        ]
        return _build_payload(kind="commodity", status="ok", facts=facts, sources=["https://stooq.com/"], provider="stooq")
    except Exception:
        return _build_payload(kind="commodity", status="error", provider="stooq", failure_reason="provider_failed")


def _provider_fx_frankfurter(query: str, profile: dict, slots: dict[str, Any]) -> dict[str, Any]:
    _ = profile
    return _fetch_fx_rate(
        query=query,
        pair_hint=slots.get("pair"),
        base_hint=slots.get("base_currency"),
        quote_hint=slots.get("quote_currency"),
    )


def _provider_news_web(query: str, profile: dict, slots: dict[str, Any]) -> dict[str, Any]:
    _ = profile
    topic = _to_text(slots.get("topic")) or _extract_news_topic(query) or "india finance"
    return _fetch_news_via_web(topic=topic)


def _provider_news_rss(query: str, profile: dict, slots: dict[str, Any]) -> dict[str, Any]:
    _ = profile
    topic = _to_text(slots.get("topic")) or _extract_news_topic(query) or "india finance"
    return _fetch_headlines_rss(topic=topic)


def _fetch_weather(query: str, profile: dict, location_hint: Optional[str] = None) -> dict[str, Any]:
    location = _to_text(location_hint) or _extract_location(query) or _to_text(profile.get("city"))
    if not location:
        return _build_payload(
            kind="weather",
            status="needs_input",
            message="I can check that. Please share your city name.",
            provider="open_meteo",
        )

    try:
        resolved_location, top = _geocode_with_retry(location)
        if not top:
            return _build_payload(
                kind="weather",
                status="needs_input",
                message=f"I couldn't find '{location}'. Please share a nearby city name.",
                sources=["https://open-meteo.com/"],
                provider="open_meteo",
            )

        forecast = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": top.get("latitude"),
                "longitude": top.get("longitude"),
                "current": "temperature_2m,apparent_temperature,relative_humidity_2m,wind_speed_10m",
                "timezone": "auto",
            },
            timeout=_HTTP_TIMEOUT_SECONDS,
        )
        forecast.raise_for_status()
        current = (forecast.json() or {}).get("current") or {}

        place_name = top.get("name") or resolved_location
        country = top.get("country_code") or top.get("country")
        facts = [
            f"Location: {place_name}{', ' + country if country else ''}",
            f"Temperature: {current.get('temperature_2m')} C",
            f"Feels like: {current.get('apparent_temperature')} C",
            f"Humidity: {current.get('relative_humidity_2m')}%",
            f"Wind speed: {current.get('wind_speed_10m')} km/h",
        ]
        return _build_payload(
            kind="weather",
            status="ok",
            facts=facts,
            sources=["https://open-meteo.com/"],
            provider="open_meteo",
        )
    except Exception:
        return _build_payload(
            kind="weather",
            status="error",
            message="I couldn't fetch live weather right now. Please try again shortly.",
            sources=["https://open-meteo.com/"],
            provider="open_meteo",
        )


def _geocode_with_retry(location: str) -> tuple[str, dict | None]:
    text = _to_text(location) or ""
    has_non_latin = bool(re.search(r"[\u0600-\u06FF\u0900-\u097F]", text))

    if has_non_latin:
        transliterated = _transliterate_location(location)
        if transliterated and transliterated.lower() != location.lower():
            retry = _geocode_location(transliterated)
            if retry:
                return transliterated, retry

    direct = _geocode_location(location)
    if direct:
        return location, direct

    transliterated = _transliterate_location(location)
    if transliterated and transliterated.lower() != location.lower():
        retry = _geocode_location(transliterated)
        if retry:
            return transliterated, retry

    return location, None


def _geocode_location(location: str) -> dict | None:
    geo = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": location, "count": 8, "language": "en", "format": "json"},
        timeout=_HTTP_TIMEOUT_SECONDS,
    )
    geo.raise_for_status()
    results = (geo.json() or {}).get("results") or []
    if not results:
        return None

    def _population(item: dict) -> float:
        try:
            return float(item.get("population") or 0)
        except (TypeError, ValueError):
            return 0.0

    return max(results, key=_population)


def _transliterate_location(location: str) -> str | None:
    text = _to_text(location)
    if not text:
        return None
    if re.search(r"[A-Za-z]", text) and not re.search(r"[\u0600-\u06FF\u0900-\u097F]", text):
        return text

    payload = _llm_extract_json(
        prompt=(
            "Normalize this city/place name into standard English (Latin script) for geocoding.\n"
            "Return STRICT JSON only with key 'location'.\n"
            f"Input: {text}"
        ),
        operation="turn_control",
    )
    if not isinstance(payload, dict):
        return None
    return _to_text(payload.get("location"))


def _fetch_stock_quote(query: str, symbol_hint: Optional[str] = None, company_hint: Optional[str] = None) -> dict[str, Any]:
    symbol = _normalize_symbol(_to_text(symbol_hint) or _extract_symbol(query))
    company = _to_text(company_hint) or _extract_company_hint(query)
    if not symbol and not company:
        return _build_payload(
            kind="stock",
            status="needs_input",
            message="Please share a stock ticker or company name (for example, TCS or RELIANCE).",
            provider="equity_router",
        )

    chosen_symbol = symbol
    chosen_nse_symbol = _nse_symbol_for_indian_symbol(symbol)
    if not chosen_symbol and company:
        resolution = _resolve_equity_symbol(company)
        if resolution.get("status") == "ambiguous":
            options = resolution.get("options") or []
            options_text = ", ".join(options[:3]) if options else "multiple similar symbols"
            return _build_payload(
                kind="stock",
                status="needs_input",
                message=f"I found multiple matches: {options_text}. Which one do you want?",
                provider="equity_resolver",
                ambiguity_clarification=True,
            )
        chosen_symbol = _to_text(resolution.get("symbol"))
        chosen_nse_symbol = _to_text(resolution.get("nse_symbol")) or _nse_symbol_for_indian_symbol(chosen_symbol)

    if not chosen_symbol and not chosen_nse_symbol:
        return _build_payload(
            kind="stock",
            status="needs_input",
            message="I couldn't identify that stock. Please share the exact ticker.",
            provider="equity_resolver",
        )

    try:
        if chosen_nse_symbol:
            nse_quote = _fetch_stock_from_nse(chosen_nse_symbol)
            if nse_quote:
                return nse_quote

        stooq_quote = _fetch_stock_from_stooq(chosen_symbol or chosen_nse_symbol)
        if stooq_quote:
            return stooq_quote

        return _build_payload(
            kind="stock",
            status="error",
            message=f"I couldn't fetch quote data for {chosen_symbol or chosen_nse_symbol}. Please try again shortly.",
            sources=["https://www.nseindia.com/", "https://stooq.com/"],
            provider="equity_router",
        )
    except Exception:
        return _build_payload(
            kind="stock",
            status="error",
            message="I couldn't fetch live stock data right now. Please try again shortly.",
            sources=["https://www.nseindia.com/", "https://stooq.com/"],
            provider="equity_router",
        )


def _fetch_stock_history(
    query: str,
    symbol_hint: Optional[str] = None,
    company_hint: Optional[str] = None,
    window_days_hint: Optional[int] = None,
) -> dict[str, Any]:
    symbol = _normalize_symbol(_to_text(symbol_hint) or _extract_symbol(query))
    company = _to_text(company_hint) or _extract_company_hint(query)
    if not symbol and not company:
        return _build_payload(
            kind="stock_history",
            status="needs_input",
            message="Please share a stock ticker or company name for historical performance.",
            provider="yahoo_finance",
        )

    chosen_symbol = symbol
    nse_symbol = _nse_symbol_for_indian_symbol(symbol)
    if not chosen_symbol and company:
        resolution = _resolve_equity_symbol(company)
        if resolution.get("status") == "ambiguous":
            options = resolution.get("options") or []
            options_text = ", ".join(options[:3]) if options else "multiple similar symbols"
            return _build_payload(
                kind="stock_history",
                status="needs_input",
                message=f"I found multiple matches: {options_text}. Which one do you want?",
                provider="equity_resolver",
                ambiguity_clarification=True,
            )
        chosen_symbol = _to_text(resolution.get("symbol"))
        nse_symbol = _to_text(resolution.get("nse_symbol")) or _nse_symbol_for_indian_symbol(chosen_symbol)

    yahoo_symbol = chosen_symbol
    if not yahoo_symbol and nse_symbol:
        yahoo_symbol = f"{nse_symbol}.NS"

    if not yahoo_symbol:
        return _build_payload(
            kind="stock_history",
            status="needs_input",
            message="I couldn't identify that stock. Please share the exact ticker.",
            provider="equity_resolver",
        )

    window_days = _normalize_history_window_days(window_days_hint or _extract_history_window_days(query), default_days=1095)
    end_dt = dt.datetime.utcnow()
    start_dt = end_dt - dt.timedelta(days=window_days)

    try:
        history = requests.get(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{quote_plus(yahoo_symbol)}",
            params={
                "interval": "1d",
                "period1": int(start_dt.timestamp()),
                "period2": int(end_dt.timestamp()),
                "events": "history",
            },
            timeout=max(6.0, _HTTP_TIMEOUT_SECONDS),
        )
        history.raise_for_status()
        result = (((history.json() or {}).get("chart") or {}).get("result")) or []
        if not result:
            return _build_payload(
                kind="stock_history",
                status="error",
                message=f"I couldn't fetch historical data for {yahoo_symbol}.",
                sources=["https://finance.yahoo.com/"],
                provider="yahoo_finance",
            )

        quote_data = ((((result[0] or {}).get("indicators") or {}).get("quote")) or [{}])[0]
        closes = quote_data.get("close") or []
        series = [float(v) for v in closes if isinstance(v, (int, float))]
        if len(series) < 2:
            return _build_payload(
                kind="stock_history",
                status="error",
                message=f"I couldn't fetch enough historical points for {yahoo_symbol}.",
                sources=["https://finance.yahoo.com/"],
                provider="yahoo_finance",
            )

        start_price = series[0]
        end_price = series[-1]
        total_return = ((end_price - start_price) / start_price) * 100 if start_price else 0.0
        meta = result[0].get("meta") or {}
        instrument = _to_text(meta.get("symbol")) or yahoo_symbol
        facts = [
            f"Instrument: {instrument}",
            f"Period: {start_dt.date().isoformat()} to {end_dt.date().isoformat()}",
            f"Start price: {round(start_price, 2)}",
            f"End price: {round(end_price, 2)}",
            f"Total return: {round(total_return, 2)}%",
        ]
        return _build_payload(
            kind="stock_history",
            status="ok",
            facts=facts,
            sources=["https://finance.yahoo.com/"],
            provider="yahoo_finance",
        )
    except Exception:
        return _build_payload(
            kind="stock_history",
            status="error",
            message="I couldn't fetch historical stock data right now. Please try again shortly.",
            sources=["https://finance.yahoo.com/"],
            provider="yahoo_finance",
        )


def _fetch_stock_history_stooq(
    query: str,
    symbol_hint: Optional[str] = None,
    company_hint: Optional[str] = None,
    window_days_hint: Optional[int] = None,
) -> dict[str, Any]:
    symbol = _normalize_symbol(_to_text(symbol_hint) or _extract_symbol(query))
    company = _to_text(company_hint) or _extract_company_hint(query)
    if not symbol and not company:
        return _build_payload(
            kind="stock_history",
            status="needs_input",
            message="Please share a stock ticker or company name for historical performance.",
            provider="stooq_history",
        )

    chosen_symbol = symbol
    nse_symbol = _nse_symbol_for_indian_symbol(symbol)
    if not chosen_symbol and company:
        resolution = _resolve_equity_symbol(company)
        if resolution.get("status") == "ambiguous":
            options = resolution.get("options") or []
            options_text = ", ".join(options[:3]) if options else "multiple similar symbols"
            return _build_payload(
                kind="stock_history",
                status="needs_input",
                message=f"I found multiple matches: {options_text}. Which one do you want?",
                provider="equity_resolver",
                ambiguity_clarification=True,
            )
        chosen_symbol = _to_text(resolution.get("symbol"))
        nse_symbol = _to_text(resolution.get("nse_symbol")) or _nse_symbol_for_indian_symbol(chosen_symbol)

    stooq_symbol = _to_stooq_symbol_for_history(chosen_symbol, nse_symbol=nse_symbol)
    if not stooq_symbol:
        return _build_payload(
            kind="stock_history",
            status="needs_input",
            message="I couldn't identify that stock. Please share the exact ticker.",
            provider="stooq_history",
        )

    window_days = _normalize_history_window_days(window_days_hint or _extract_history_window_days(query), default_days=1095)
    end_date = dt.datetime.utcnow().date()
    start_date = end_date - dt.timedelta(days=window_days)

    try:
        resp = requests.get(
            "https://stooq.com/q/d/l/",
            params={"s": stooq_symbol, "i": "d"},
            timeout=max(8.0, _HTTP_TIMEOUT_SECONDS),
        )
        resp.raise_for_status()
        lines = [line.strip() for line in (resp.text or "").splitlines() if line.strip()]
        if len(lines) < 3:
            return _build_payload(
                kind="stock_history",
                status="error",
                message=f"I couldn't fetch historical data for {stooq_symbol.upper()}.",
                sources=["https://stooq.com/"],
                provider="stooq_history",
            )

        closes: list[float] = []
        filtered_dates: list[str] = []
        for row in lines[1:]:
            parts = [part.strip() for part in row.split(",")]
            if len(parts) < 5:
                continue
            date_str = parts[0]
            close_str = parts[4]
            if close_str in {"", "N/D"}:
                continue
            try:
                row_date = dt.datetime.strptime(date_str, "%Y-%m-%d").date()
                close_value = float(close_str)
            except Exception:
                continue
            if row_date < start_date or row_date > end_date:
                continue
            closes.append(close_value)
            filtered_dates.append(row_date.isoformat())

        if len(closes) < 2:
            return _build_payload(
                kind="stock_history",
                status="error",
                message=f"I couldn't fetch enough historical points for {stooq_symbol.upper()}.",
                sources=["https://stooq.com/"],
                provider="stooq_history",
            )

        start_price = closes[0]
        end_price = closes[-1]
        total_return = ((end_price - start_price) / start_price) * 100 if start_price else 0.0
        facts = [
            f"Instrument: {stooq_symbol.upper()}",
            f"Period: {filtered_dates[0]} to {filtered_dates[-1]}",
            f"Start price: {round(start_price, 2)}",
            f"End price: {round(end_price, 2)}",
            f"Total return: {round(total_return, 2)}%",
        ]
        return _build_payload(
            kind="stock_history",
            status="ok",
            facts=facts,
            sources=["https://stooq.com/"],
            provider="stooq_history",
        )
    except Exception:
        return _build_payload(
            kind="stock_history",
            status="error",
            message="I couldn't fetch historical stock data right now. Please try again shortly.",
            sources=["https://stooq.com/"],
            provider="stooq_history",
        )


def _fetch_crypto_quote(query: str, coin_hint: Optional[str] = None) -> dict[str, Any]:
    coin = _to_text(coin_hint) or _extract_coin_hint(query)
    if not coin:
        return _build_payload(
            kind="crypto",
            status="needs_input",
            message="Please share a crypto name or ticker (for example, Bitcoin or BTC).",
            provider="coingecko",
        )

    resolution = _resolve_crypto_asset(coin)
    if resolution.get("status") == "ambiguous":
        options = resolution.get("options") or []
        options_text = ", ".join(options[:3]) if options else "multiple similar coins"
        return _build_payload(
            kind="crypto",
            status="needs_input",
            message=f"I found multiple matches: {options_text}. Which coin do you want?",
            provider="coingecko",
            ambiguity_clarification=True,
        )
    if resolution.get("status") != "ok":
        return _build_payload(
            kind="crypto",
            status="needs_input",
            message="I couldn't identify that crypto asset. Please share the exact coin name or ticker.",
            provider="coingecko",
        )

    coin_id = _to_text(resolution.get("id"))
    symbol = _to_text(resolution.get("symbol")) or coin.upper()
    name = _to_text(resolution.get("name")) or coin
    if not coin_id:
        return _build_payload(
            kind="crypto",
            status="error",
            message="I couldn't resolve that crypto asset right now.",
            provider="coingecko",
        )

    try:
        price_resp = requests.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={
                "ids": coin_id,
                "vs_currencies": "usd,inr",
                "include_24hr_change": "true",
            },
            timeout=_HTTP_TIMEOUT_SECONDS,
        )
        price_resp.raise_for_status()
        data = price_resp.json() or {}
        item = data.get(coin_id) or {}
        usd = item.get("usd")
        inr = item.get("inr")
        change = item.get("usd_24h_change")
        if usd is None:
            return _build_payload(
                kind="crypto",
                status="error",
                message="I couldn't fetch live crypto pricing right now. Please try again shortly.",
                sources=["https://www.coingecko.com/"],
                provider="coingecko",
            )

        facts = [
            f"Instrument: {name} ({symbol.upper()})",
            f"Last price: {round(float(usd), 4)} USD",
        ]
        if inr is not None:
            facts.append(f"Last price INR: {round(float(inr), 2)} INR")
        if isinstance(change, (int, float)):
            facts.append(f"24h change: {round(float(change), 2)}%")
        return _build_payload(
            kind="crypto",
            status="ok",
            facts=facts,
            sources=["https://www.coingecko.com/"],
            provider="coingecko",
        )
    except Exception:
        return _build_payload(
            kind="crypto",
            status="error",
            message="I couldn't fetch live crypto pricing right now. Please try again shortly.",
            sources=["https://www.coingecko.com/"],
            provider="coingecko",
        )


def _fetch_crypto_history(query: str, coin_hint: Optional[str], window_days_hint: Optional[int]) -> dict[str, Any]:
    coin = _to_text(coin_hint) or _extract_coin_hint(query)
    if not coin:
        return _build_payload(
            kind="crypto_history",
            status="needs_input",
            message="Please share a crypto name or ticker for historical performance.",
            provider="coingecko",
        )

    resolution = _resolve_crypto_asset(coin)
    if resolution.get("status") == "ambiguous":
        options = resolution.get("options") or []
        options_text = ", ".join(options[:3]) if options else "multiple similar coins"
        return _build_payload(
            kind="crypto_history",
            status="needs_input",
            message=f"I found multiple matches: {options_text}. Which coin do you want?",
            provider="coingecko",
            ambiguity_clarification=True,
        )
    if resolution.get("status") != "ok":
        return _build_payload(
            kind="crypto_history",
            status="needs_input",
            message="I couldn't identify that crypto asset. Please share the exact coin name or ticker.",
            provider="coingecko",
        )

    coin_id = _to_text(resolution.get("id"))
    symbol = _to_text(resolution.get("symbol")) or coin.upper()
    name = _to_text(resolution.get("name")) or coin
    if not coin_id:
        return _build_payload(
            kind="crypto_history",
            status="error",
            message="I couldn't resolve that crypto asset right now.",
            provider="coingecko",
        )

    window_days = _normalize_history_window_days(window_days_hint or _extract_history_window_days(query), default_days=1095)
    end_dt = dt.datetime.utcnow()
    start_dt = end_dt - dt.timedelta(days=window_days)

    try:
        history_resp = requests.get(
            f"https://api.coingecko.com/api/v3/coins/{quote_plus(coin_id)}/market_chart",
            params={
                "vs_currency": "usd",
                "days": window_days,
                "interval": "daily",
            },
            timeout=max(8.0, _HTTP_TIMEOUT_SECONDS),
        )
        history_resp.raise_for_status()
        prices = (history_resp.json() or {}).get("prices") or []
        series = []
        for item in prices:
            if not isinstance(item, list) or len(item) < 2:
                continue
            value = item[1]
            try:
                series.append(float(value))
            except (TypeError, ValueError):
                continue

        if len(series) < 2:
            return _build_payload(
                kind="crypto_history",
                status="error",
                message=f"I couldn't fetch enough historical points for {name}.",
                sources=["https://www.coingecko.com/"],
                provider="coingecko",
            )

        start_price = series[0]
        end_price = series[-1]
        total_return = ((end_price - start_price) / start_price) * 100 if start_price else 0.0
        facts = [
            f"Instrument: {name} ({symbol.upper()})",
            f"Period: {start_dt.date().isoformat()} to {end_dt.date().isoformat()}",
            f"Start price: {round(start_price, 4)}",
            f"End price: {round(end_price, 4)}",
            f"Total return: {round(total_return, 2)}%",
        ]
        return _build_payload(
            kind="crypto_history",
            status="ok",
            facts=facts,
            sources=["https://www.coingecko.com/"],
            provider="coingecko",
        )
    except Exception:
        return _build_payload(
            kind="crypto_history",
            status="error",
            message="I couldn't fetch historical crypto data right now. Please try again shortly.",
            sources=["https://www.coingecko.com/"],
            provider="coingecko",
        )


def _fetch_gold_spot(query: str) -> dict[str, Any]:
    _ = query
    try:
        resp = requests.get(
            "https://query1.finance.yahoo.com/v8/finance/chart/GC=F",
            params={"interval": "1d", "range": "5d"},
            timeout=max(6.0, _HTTP_TIMEOUT_SECONDS),
        )
        resp.raise_for_status()
        result = (((resp.json() or {}).get("chart") or {}).get("result")) or []
        if not result:
            return _build_payload(
                kind="gold",
                status="error",
                message="I couldn't fetch live gold pricing right now. Please try again shortly.",
                sources=["https://finance.yahoo.com/"],
                provider="yahoo_finance",
            )

        meta = result[0].get("meta") or {}
        quote_data = ((((result[0] or {}).get("indicators") or {}).get("quote")) or [{}])[0]
        closes = [float(v) for v in (quote_data.get("close") or []) if isinstance(v, (int, float))]
        if not closes:
            return _build_payload(
                kind="gold",
                status="error",
                message="I couldn't fetch live gold pricing right now. Please try again shortly.",
                sources=["https://finance.yahoo.com/"],
                provider="yahoo_finance",
            )

        last_price = closes[-1]
        previous = closes[-2] if len(closes) > 1 else float(meta.get("previousClose") or 0.0)
        change_pct = ((last_price - previous) / previous) * 100 if previous else None
        currency = _to_text(meta.get("currency")) or "USD"

        facts = [
            "Instrument: Gold futures (GC=F)",
            f"Last price: {round(last_price, 2)} {currency} per troy ounce",
        ]
        if isinstance(change_pct, (int, float)):
            facts.append(f"Change: {round(change_pct, 2)}% vs previous close")

        return _build_payload(
            kind="gold",
            status="ok",
            facts=facts,
            sources=["https://finance.yahoo.com/"],
            provider="yahoo_finance",
        )
    except Exception:
        return _build_payload(
            kind="gold",
            status="error",
            message="I couldn't fetch live gold pricing right now. Please try again shortly.",
            sources=["https://finance.yahoo.com/"],
            provider="yahoo_finance",
        )


def _fetch_gold_spot_stooq(query: str) -> dict[str, Any]:
    _ = query
    candidates = ["xauusd", "gold"]
    for symbol in candidates:
        try:
            resp = requests.get(
                "https://stooq.com/q/l/",
                params={"s": symbol, "f": "sd2t2ohlcv", "h": "", "e": "csv"},
                timeout=max(6.0, _HTTP_TIMEOUT_SECONDS),
            )
            resp.raise_for_status()
            lines = [line.strip() for line in (resp.text or "").splitlines() if line.strip()]
            if len(lines) < 2:
                continue
            parts = [part.strip() for part in lines[1].split(",")]
            if len(parts) < 8:
                continue
            out_symbol, date_s, time_s, open_s, high_s, low_s, close_s, _volume_s = parts[:8]
            if close_s in {"", "N/D"}:
                continue

            facts = [
                f"Instrument: {out_symbol}",
                f"Last price: {close_s} USD per troy ounce",
                f"Open/High/Low: {open_s}/{high_s}/{low_s}",
                f"Trade date/time: {date_s} {time_s}",
            ]
            return _build_payload(
                kind="gold",
                status="ok",
                facts=facts,
                sources=["https://stooq.com/"],
                provider="stooq",
            )
        except Exception:
            continue

    return _build_payload(
        kind="gold",
        status="error",
        message="I couldn't fetch live gold pricing right now. Please try again shortly.",
        sources=["https://stooq.com/"],
        provider="stooq",
        failure_reason="provider_failed",
    )


def _fetch_gold_history_stooq(query: str, window_days_hint: Optional[int] = None) -> dict[str, Any]:
    window_days = _normalize_history_window_days(
        window_days_hint or _extract_history_window_days(query), default_days=1095
    )
    end_date = dt.datetime.utcnow().date()
    start_date = end_date - dt.timedelta(days=window_days)

    try:
        resp = requests.get(
            "https://stooq.com/q/d/l/",
            params={"s": "xauusd", "i": "d"},
            timeout=max(8.0, _HTTP_TIMEOUT_SECONDS),
        )
        resp.raise_for_status()
        lines = [line.strip() for line in (resp.text or "").splitlines() if line.strip()]
        if len(lines) < 3:
            return _build_payload(
                kind="gold_history",
                status="error",
                message="I couldn't fetch historical gold data right now.",
                sources=["https://stooq.com/"],
                provider="stooq_history",
            )

        closes: list[float] = []
        filtered_dates: list[str] = []
        for row in lines[1:]:
            parts = [part.strip() for part in row.split(",")]
            if len(parts) < 5:
                continue
            date_str = parts[0]
            close_str = parts[4]
            if close_str in {"", "N/D"}:
                continue
            try:
                row_date = dt.datetime.strptime(date_str, "%Y-%m-%d").date()
                close_value = float(close_str)
            except Exception:
                continue
            if row_date < start_date or row_date > end_date:
                continue
            closes.append(close_value)
            filtered_dates.append(row_date.isoformat())

        if len(closes) < 2:
            return _build_payload(
                kind="gold_history",
                status="error",
                message="I couldn't fetch enough historical points for gold.",
                sources=["https://stooq.com/"],
                provider="stooq_history",
            )

        start_price = closes[0]
        end_price = closes[-1]
        total_return = ((end_price - start_price) / start_price) * 100 if start_price else 0.0
        facts = [
            "Instrument: XAUUSD",
            f"Period: {filtered_dates[0]} to {filtered_dates[-1]}",
            f"Start price: {round(start_price, 2)}",
            f"End price: {round(end_price, 2)}",
            f"Total return: {round(total_return, 2)}%",
        ]
        return _build_payload(
            kind="gold_history",
            status="ok",
            facts=facts,
            sources=["https://stooq.com/"],
            provider="stooq_history",
        )
    except Exception:
        return _build_payload(
            kind="gold_history",
            status="error",
            message="I couldn't fetch historical gold data right now.",
            sources=["https://stooq.com/"],
            provider="stooq_history",
        )


def _fetch_fx_rate(
    query: str,
    pair_hint: Optional[str] = None,
    base_hint: Optional[str] = None,
    quote_hint: Optional[str] = None,
) -> dict[str, Any]:
    base, quote = _resolve_fx_pair(query=query, pair_hint=pair_hint, base_hint=base_hint, quote_hint=quote_hint)
    if not base or not quote:
        return _build_payload(
            kind="fx",
            status="needs_input",
            message="Please share a currency pair like USD/INR or EUR/USD.",
            provider="frankfurter",
        )

    try:
        resp = requests.get(
            "https://api.frankfurter.app/latest",
            params={"from": base, "to": quote},
            timeout=_HTTP_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        payload = resp.json() or {}
        rates = payload.get("rates") or {}
        rate_value = rates.get(quote)
        if rate_value is None:
            return _build_payload(
                kind="fx",
                status="error",
                message=f"I couldn't fetch the latest {base}/{quote} rate right now.",
                sources=["https://www.frankfurter.app/"],
                provider="frankfurter",
            )

        facts = [
            f"Pair: {base}/{quote}",
            f"Rate: 1 {base} = {rate_value} {quote}",
            f"Date: {payload.get('date')}",
        ]
        return _build_payload(
            kind="fx",
            status="ok",
            facts=facts,
            sources=["https://www.frankfurter.app/"],
            provider="frankfurter",
        )
    except Exception:
        return _build_payload(
            kind="fx",
            status="error",
            message=f"I couldn't fetch the latest {base}/{quote} rate right now.",
            sources=["https://www.frankfurter.app/"],
            provider="frankfurter",
        )


def _fetch_news_via_web(topic: str) -> dict[str, Any]:
    results = search_web(f"latest {topic} news", max_results=_MAX_HEADLINES)
    if not results:
        return _build_payload(
            kind="news",
            status="error",
            message="I couldn't find fresh headlines for that topic right now.",
            sources=[],
            route="web_fallback",
            provider="duckduckgo",
            failure_reason="fallback_empty",
            provider_attempts=[{"provider": "duckduckgo", "status": "error"}],
        )

    facts = [f"Topic: {topic}"]
    for idx, item in enumerate(results[:_MAX_HEADLINES], start=1):
        title = _to_text(item.get("title"))
        if title:
            facts.append(f"Headline {idx}: {title}")

    sources = [str(item.get("url")) for item in results[:_MAX_HEADLINES] if str(item.get("url") or "").startswith("http")]
    if len(facts) <= 1 or not sources:
        return _build_payload(
            kind="news",
            status="error",
            message="I couldn't find enough reliable headlines for that topic right now.",
            sources=sources,
            route="web_fallback",
            provider="duckduckgo",
            failure_reason="fallback_empty",
            provider_attempts=[{"provider": "duckduckgo", "status": "error"}],
        )

    return _build_payload(
        kind="news",
        status="ok",
        facts=facts,
        sources=sources,
        route="web_fallback",
        provider="duckduckgo",
        failure_reason="fallback_ok",
        provider_attempts=[{"provider": "duckduckgo", "status": "ok"}],
    )


def _fetch_headlines_rss(topic: str) -> dict[str, Any]:
    try:
        rss_url = f"https://news.google.com/rss/search?q={quote_plus(topic)}"
        resp = requests.get(rss_url, timeout=_HTTP_TIMEOUT_SECONDS)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        items = root.findall(".//item")
        if not items:
            return _build_payload(
                kind="news",
                status="error",
                message="I couldn't find fresh headlines for that topic right now.",
                sources=["https://news.google.com/"],
                provider="google_news_rss",
            )

        headlines = []
        sources = []
        for item in items[:_MAX_HEADLINES]:
            title = _to_text(item.findtext("title"))
            link = _to_text(item.findtext("link"))
            if title:
                headlines.append(title)
            if link:
                sources.append(link)

        facts = [f"Topic: {topic}"] + [f"Headline {i+1}: {headline}" for i, headline in enumerate(headlines)]
        return _build_payload(
            kind="news",
            status="ok",
            facts=facts,
            sources=sources or ["https://news.google.com/"],
            provider="google_news_rss",
        )
    except Exception:
        return _build_payload(
            kind="news",
            status="error",
            message="I couldn't fetch live headlines right now. Please try again shortly.",
            sources=["https://news.google.com/"],
            provider="google_news_rss",
        )


def _fetch_web_fallback(query: str, kind: str) -> dict[str, Any] | None:
    results = search_web(query, max_results=_MAX_HEADLINES)
    if not results:
        return _build_payload(
            kind=kind,
            status="error",
            message="I couldn't fetch a reliable live update right now. Please try again shortly.",
            route="web_fallback",
            provider="duckduckgo",
            failure_reason="fallback_empty",
            provider_attempts=[{"provider": "duckduckgo", "status": "error"}],
        )

    sources = [str(item.get("url")) for item in results if str(item.get("url") or "").startswith("http")]
    facts: list[str] = []
    if kind == "news":
        for idx, item in enumerate(results[:_MAX_HEADLINES], start=1):
            title = _to_text(item.get("title"))
            if title:
                facts.append(f"Headline {idx}: {title}")
    else:
        for idx, item in enumerate(results[:_MAX_HEADLINES], start=1):
            title = _to_text(item.get("title"))
            snippet = _to_text(item.get("snippet"))
            if title and snippet:
                facts.append(f"Result {idx}: {title} - {snippet}")
            elif title:
                facts.append(f"Result {idx}: {title}")

    if not sources or not facts:
        return _build_payload(
            kind=kind,
            status="error",
            message="I couldn't fetch a reliable live update right now. Please try again shortly.",
            sources=sources,
            route="web_fallback",
            provider="duckduckgo",
            failure_reason="fallback_empty",
            provider_attempts=[{"provider": "duckduckgo", "status": "error"}],
        )

    return _build_payload(
        kind=kind,
        status="ok",
        facts=facts,
        sources=sources,
        route="web_fallback",
        provider="duckduckgo",
        failure_reason="fallback_ok",
        provider_attempts=[{"provider": "duckduckgo", "status": "ok"}],
    )


def _extract_location(query: str) -> Optional[str]:
    text = (query or "").strip()
    if not text:
        return None

    payload = _llm_extract_json(
        prompt=(
            "Extract the weather location from the user message.\n"
            "Return a location only if user explicitly mentions a city/place/region.\n"
            "If location is missing or unclear, return null.\n\n"
            f"User message: {text}\n\n"
            "Return STRICT JSON only with key 'location' (string or null)."
        ),
        operation="turn_control",
    )
    if not isinstance(payload, dict):
        return None

    location = _to_text(payload.get("location"))
    if not location:
        return None
    location = re.sub(r"\s+", " ", location).strip(" .,-")
    return location[:80] if len(location) > 80 else location


def _extract_symbol(query: str) -> Optional[str]:
    text = (query or "").strip()
    if not text:
        return None
    candidates = re.findall(r"\b[A-Z]{1,6}(?:\.(?:NS|BO))?\b", text)
    blocked = {"INR", "USD", "NSE", "BSE"}
    for candidate in candidates:
        if candidate in blocked:
            continue
        return candidate
    return None


def _extract_company_hint(query: str) -> Optional[str]:
    text = (query or "").strip()
    if not text:
        return None

    payload = _llm_extract_json(
        prompt=(
            "Extract company/entity for equity quote lookup from the user message.\n"
            "Return null if user did not provide any stock/company reference.\n"
            "Return STRICT JSON only with key 'company' (string or null).\n\n"
            f"User message: {text}"
        ),
        operation="turn_control",
    )
    if isinstance(payload, dict):
        company = _to_text(payload.get("company"))
        if company:
            return company

    patterns = [
        r"(?:stock|share|quote|price)\s+(?:of|for)\s+([a-zA-Z0-9 .&-]{2,60})",
        r"(?:how is|update on)\s+([a-zA-Z0-9 .&-]{2,60})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            hint = match.group(1).strip(" .,-")
            if hint:
                return hint
    return None


def _extract_coin_hint(query: str) -> Optional[str]:
    text = (query or "").strip()
    if not text:
        return None
    payload = _llm_extract_json(
        prompt=(
            "Extract the crypto coin name or ticker from the user message.\n"
            "Return null if no crypto asset is explicitly present.\n"
            "Return STRICT JSON only with key 'coin' (string or null).\n\n"
            f"User message: {text}"
        ),
        operation="turn_control",
    )
    if isinstance(payload, dict):
        return _to_text(payload.get("coin"))
    return None


def _extract_news_topic(query: str) -> Optional[str]:
    text = (query or "").strip()
    if not text:
        return None
    payload = _llm_extract_json(
        prompt=(
            "Extract a concise topic for live news lookup from this user message.\n"
            "If the user requests generic finance/business news, return 'india finance'.\n"
            "If no news request is present, return null.\n"
            "Return STRICT JSON only with key 'topic' (string or null).\n\n"
            f"User message: {text}"
        ),
        operation="turn_control",
    )
    if isinstance(payload, dict):
        topic = _to_text(payload.get("topic"))
        if topic:
            return topic
    return None


def _extract_history_window_days(query: str) -> int | None:
    text = (query or "").strip()
    if not text:
        return None
    payload = _llm_extract_json(
        prompt=(
            "Extract desired historical window for market performance.\n"
            "Map to one of these day values only: 1095 (3 years), 1825 (5 years), 3650 (10 years).\n"
            "If no explicit period is provided, return null.\n"
            "Return STRICT JSON only with key 'window_days' (integer or null).\n\n"
            f"User message: {text}"
        ),
        operation="turn_control",
    )
    if not isinstance(payload, dict):
        return None
    return _to_window_days(payload.get("window_days"))


def _resolve_equity_symbol(company_hint: str) -> dict[str, Any]:
    hint = _to_text(company_hint)
    if not hint:
        return {"status": "none", "symbol": None, "nse_symbol": None, "options": []}

    candidates: list[dict[str, Any]] = []
    hint_norm = _normalize_text(hint)
    try:
        resp = requests.get(
            "https://query1.finance.yahoo.com/v1/finance/search",
            params={"q": hint, "quotesCount": 6, "newsCount": 0},
            timeout=_HTTP_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        quotes = (resp.json() or {}).get("quotes") or []
        for idx, quote in enumerate(quotes):
            symbol = _to_text(quote.get("symbol"))
            if not symbol:
                continue
            name = _to_text(quote.get("shortname")) or _to_text(quote.get("longname")) or symbol
            score = _match_score(hint_norm, symbol=symbol, name=name, rank=idx)
            exchange = _to_text(quote.get("exchange")) or _to_text(quote.get("fullExchangeName")) or ""
            nse_symbol = _nse_symbol_for_indian_symbol(symbol)
            if "NSE" in exchange.upper() and nse_symbol:
                candidates.append(
                    {"symbol": symbol, "name": name, "score": score, "nse_symbol": nse_symbol}
                )
            else:
                candidates.append({"symbol": symbol, "name": name, "score": score})
    except Exception:
        pass

    nse_symbol = _resolve_nse_symbol(hint)
    if nse_symbol:
        score = _match_score(hint_norm, symbol=nse_symbol, name=nse_symbol, rank=0) + 10
        candidates.append({"symbol": f"{nse_symbol}.NS", "name": nse_symbol, "score": score, "nse_symbol": nse_symbol})

    if not candidates:
        return {"status": "none", "symbol": None, "nse_symbol": None, "options": []}

    dedup: dict[str, dict[str, Any]] = {}
    for item in candidates:
        symbol = str(item.get("symbol"))
        existing = dedup.get(symbol)
        if not existing or float(item.get("score", 0)) > float(existing.get("score", 0)):
            dedup[symbol] = item
    ranked = sorted(dedup.values(), key=lambda x: float(x.get("score", 0)), reverse=True)
    top = ranked[0]
    second = ranked[1] if len(ranked) > 1 else None
    top_score = float(top.get("score", 0))
    second_score = float(second.get("score", 0)) if second else -999.0

    options = [f"{item.get('name')} ({item.get('symbol')})" for item in ranked[:3]]
    if top_score < 35:
        return {"status": "none", "symbol": None, "nse_symbol": None, "options": options}
    if second and (top_score - second_score) <= 6:
        return {"status": "ambiguous", "symbol": None, "nse_symbol": None, "options": options}

    symbol = _to_text(top.get("symbol"))
    nse = _to_text(top.get("nse_symbol")) or _nse_symbol_for_indian_symbol(symbol)
    return {"status": "ok", "symbol": symbol, "nse_symbol": nse, "options": options}


def _resolve_crypto_asset(coin_hint: str) -> dict[str, Any]:
    hint = _to_text(coin_hint)
    if not hint:
        return {"status": "none", "id": None, "name": None, "symbol": None, "options": []}
    hint_norm = _normalize_text(hint)
    try:
        resp = requests.get(
            "https://api.coingecko.com/api/v3/search",
            params={"query": hint},
            timeout=_HTTP_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        coins = (resp.json() or {}).get("coins") or []
    except Exception:
        return {"status": "none", "id": None, "name": None, "symbol": None, "options": []}

    candidates = []
    for idx, item in enumerate(coins[:8]):
        coin_id = _to_text(item.get("id"))
        name = _to_text(item.get("name"))
        symbol = _to_text(item.get("symbol"))
        if not coin_id or not name or not symbol:
            continue
        score = _match_score(hint_norm, symbol=symbol, name=name, rank=idx)
        if _normalize_text(coin_id) == hint_norm:
            score += 20
        candidates.append({"id": coin_id, "name": name, "symbol": symbol, "score": score})

    if not candidates:
        return {"status": "none", "id": None, "name": None, "symbol": None, "options": []}

    ranked = sorted(candidates, key=lambda x: float(x.get("score", 0)), reverse=True)
    top = ranked[0]
    second = ranked[1] if len(ranked) > 1 else None
    options = [f"{item.get('name')} ({str(item.get('symbol')).upper()})" for item in ranked[:3]]
    if float(top.get("score", 0)) < 35:
        return {"status": "none", "id": None, "name": None, "symbol": None, "options": options}
    if second and (float(top.get("score", 0)) - float(second.get("score", 0))) <= 6:
        return {"status": "ambiguous", "id": None, "name": None, "symbol": None, "options": options}

    return {
        "status": "ok",
        "id": _to_text(top.get("id")),
        "name": _to_text(top.get("name")),
        "symbol": _to_text(top.get("symbol")),
        "options": options,
    }


def _resolve_fx_pair(
    query: str,
    pair_hint: Optional[str] = None,
    base_hint: Optional[str] = None,
    quote_hint: Optional[str] = None,
) -> tuple[str | None, str | None]:
    base = _normalize_ccy(base_hint)
    quote = _normalize_ccy(quote_hint)
    if base and quote:
        return base, quote

    pair = _to_text(pair_hint)
    if pair:
        parsed = _parse_pair(pair)
        if parsed:
            return parsed

    payload = _llm_extract_json(
        prompt=(
            "Extract forex pair from user message.\n"
            "Return ISO currency codes only.\n"
            "Return STRICT JSON with keys 'base_currency' and 'quote_currency' (string or null).\n\n"
            f"User message: {query}"
        ),
        operation="turn_control",
    )
    if isinstance(payload, dict):
        base = _normalize_ccy(payload.get("base_currency"))
        quote = _normalize_ccy(payload.get("quote_currency"))
        if base and quote:
            return base, quote

    parsed = _parse_pair(query)
    if parsed:
        return parsed
    return None, None


def _parse_pair(value: str) -> tuple[str, str] | None:
    text = str(value or "").strip().upper()
    if not text:
        return None

    slash = re.search(r"\b([A-Z]{3})\s*[/\-]\s*([A-Z]{3})\b", text)
    if slash:
        return slash.group(1), slash.group(2)

    to_form = re.search(r"\b([A-Z]{3})\s+TO\s+([A-Z]{3})\b", text)
    if to_form:
        return to_form.group(1), to_form.group(2)

    compact_token = re.search(r"\b([A-Z]{6})\b", text)
    if compact_token:
        token = compact_token.group(1)
        return token[:3], token[3:]

    return None


def _normalize_history_window_days(value: Any, default_days: int) -> int:
    allowed = [1095, 1825, 3650]
    parsed = _to_window_days(value)
    if parsed is None:
        return default_days
    if parsed <= 12:
        parsed *= 365
    return min(allowed, key=lambda x: abs(x - parsed))


def _to_window_days(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _normalize_symbol(symbol: Optional[str]) -> Optional[str]:
    token = _to_text(symbol)
    if not token:
        return None
    token = token.upper()
    if not re.fullmatch(r"[A-Z0-9]{1,8}(?:\.(?:NS|BO))?", token):
        return None
    return token


def _normalize_nse_symbol(symbol: Optional[str]) -> Optional[str]:
    token = _to_text(symbol)
    if not token:
        return None
    token = token.upper()
    if "." in token:
        token = token.split(".")[0]
    if not re.fullmatch(r"[A-Z0-9]{1,15}", token):
        return None
    return token


def _to_stooq_symbol_for_history(symbol: Optional[str], nse_symbol: Optional[str] = None) -> Optional[str]:
    token = _to_text(symbol)
    if token:
        upper = token.upper()
        if upper.endswith(".NS") or upper.endswith(".BO"):
            base = upper.split(".")[0].lower()
            return f"{base}.in"
        if "." in upper:
            return upper.lower()
        return f"{upper.lower()}.us"
    if nse_symbol:
        nse_token = _normalize_nse_symbol(nse_symbol)
        if nse_token:
            return f"{nse_token.lower()}.in"
    return None


def _nse_symbol_for_indian_symbol(symbol: Optional[str]) -> Optional[str]:
    token = _to_text(symbol)
    if not token:
        return None
    upper = token.upper()
    if upper.endswith(".NS") or upper.endswith(".BO"):
        return _normalize_nse_symbol(upper)
    return None


def _resolve_nse_symbol(company_hint: Optional[str]) -> Optional[str]:
    hint = _to_text(company_hint)
    if not hint:
        return None
    try:
        session, headers = _nse_session()
        resp = session.get(
            "https://www.nseindia.com/api/search/autocomplete",
            params={"q": hint},
            headers=headers,
            timeout=_HTTP_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        symbols = (resp.json() or {}).get("symbols") or []
        for item in symbols:
            sym = _to_text(item.get("symbol"))
            if sym:
                return sym.upper()
    except Exception:
        return None
    return None


def _fetch_stock_from_nse(symbol: str) -> Optional[dict[str, Any]]:
    try:
        session, headers = _nse_session()
        resp = session.get(
            "https://www.nseindia.com/api/quote-equity",
            params={"symbol": symbol},
            headers=headers,
            timeout=_HTTP_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        data = resp.json() or {}
        price_info = data.get("priceInfo") or {}
        info = data.get("info") or {}
        meta = data.get("metadata") or {}

        last_price = price_info.get("lastPrice")
        if last_price in (None, "", "N/A"):
            return None

        change = price_info.get("change")
        change_pct = price_info.get("pChange")
        company = info.get("companyName") or symbol
        last_update = meta.get("lastUpdateTime")
        facts = [
            f"Instrument: {company} ({symbol})",
            f"Last price: {last_price} INR",
            f"Change: {change} ({change_pct}%)",
        ]
        if last_update:
            facts.append(f"Last update: {last_update}")
        return _build_payload(
            kind="stock",
            status="ok",
            facts=facts,
            sources=["https://www.nseindia.com/"],
            provider="nse",
        )
    except Exception:
        return None


def _fetch_stock_from_stooq(symbol_or_hint: Optional[str]) -> Optional[dict[str, Any]]:
    token = _to_text(symbol_or_hint)
    if not token:
        return None
    symbol = re.sub(r"[^A-Za-z0-9.]", "", token).lower()
    if "." not in symbol:
        symbol = f"{symbol}.us"
    try:
        resp = requests.get(
            "https://stooq.com/q/l/",
            params={"s": symbol, "f": "sd2t2ohlcv", "h": "", "e": "csv"},
            timeout=max(6.0, _HTTP_TIMEOUT_SECONDS),
        )
        resp.raise_for_status()
        lines = [line.strip() for line in (resp.text or "").splitlines() if line.strip()]
        if len(lines) < 2:
            return None
        parts = [p.strip() for p in lines[1].split(",")]
        if len(parts) < 8:
            return None
        out_symbol, date_s, time_s, open_s, high_s, low_s, close_s, volume_s = parts[:8]
        if close_s in {"N/D", ""}:
            return None

        facts = [
            f"Instrument: {out_symbol}",
            f"Last price: {close_s} USD",
            f"Open/High/Low: {open_s}/{high_s}/{low_s}",
            f"Volume: {volume_s}",
            f"Trade date/time: {date_s} {time_s}",
        ]
        return _build_payload(
            kind="stock",
            status="ok",
            facts=facts,
            sources=["https://stooq.com/"],
            provider="stooq",
        )
    except Exception:
        return None


def _nse_session() -> tuple[requests.Session, dict[str, str]]:
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "application/json,text/plain,*/*",
        "Referer": "https://www.nseindia.com/",
    }
    session = requests.Session()
    session.get("https://www.nseindia.com", headers=headers, timeout=_HTTP_TIMEOUT_SECONDS)
    return session, headers


def _normalize_ccy(value: Any) -> str | None:
    token = _to_text(value)
    if not token:
        return None
    token = re.sub(r"[^A-Za-z]", "", token).upper()
    if not re.fullmatch(r"[A-Z]{3}", token):
        return None
    return token


def _normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def _match_score(hint_norm: str, symbol: str, name: str, rank: int) -> float:
    symbol_norm = _normalize_text(symbol)
    name_norm = _normalize_text(name)
    score = max(0.0, 20.0 - (rank * 2.0))
    if hint_norm == symbol_norm:
        score += 100
    elif hint_norm in symbol_norm:
        score += 65
    if hint_norm == name_norm:
        score += 95
    elif hint_norm and hint_norm in name_norm:
        score += 70
    elif name_norm and name_norm in hint_norm:
        score += 45
    return score


def _looks_unsafe(query: str) -> bool:
    text = (query or "").lower()
    blocked_terms = {
        "porn",
        "sex",
        "explicit",
        "nude",
        "suicide",
        "kill myself",
        "bomb",
        "make a weapon",
    }
    return any(term in text for term in blocked_terms)


def _to_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _now_utc_iso() -> str:
    return dt.datetime.utcnow().isoformat() + "Z"
