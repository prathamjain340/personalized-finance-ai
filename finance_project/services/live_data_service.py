import datetime as dt
import json
import os
import re
import xml.etree.ElementTree as ET
from typing import Any, Optional
from urllib.parse import quote_plus

import requests
from finance_project.core.llm.client import generate_response


_LIVE_ENABLED = os.getenv("LIVE_DATA_ENABLED", "true").lower() not in {"0", "false", "off", "no"}
_HTTP_TIMEOUT_SECONDS = float(os.getenv("LIVE_DATA_HTTP_TIMEOUT_SECONDS", "4"))
_MAX_HEADLINES = int(os.getenv("LIVE_DATA_MAX_HEADLINES", "3"))


_ALLOWED_LIVE_KINDS = {"weather", "stock", "stock_history", "news"}


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


def classify_live_data_kind(query: str) -> Optional[str]:
    text = (query or "").strip()
    if not text:
        return None

    prompt = (
        "You classify whether a user asks for live data updates.\n"
        "Classify the current message into exactly one kind: weather, stock, stock_history, news, none.\n"
        "Support multilingual and transliterated input.\n"
        "Choose weather/stock/news only when the user asks for current/live information.\n"
        "Choose stock_history only when the user asks about historical stock performance over a past period.\n"
        "If the message is educational, conversational, planning, or unclear, choose none.\n\n"
        f"User message: {text}\n\n"
        "Return STRICT JSON only with key 'kind' and value in [weather, stock, stock_history, news, none]."
    )

    try:
        raw = generate_response(prompt, operation="turn_control")
        payload = _extract_json_object(raw)
        if not isinstance(payload, dict):
            return None

        kind = str(payload.get("kind") or "").strip().lower()
        if kind in _ALLOWED_LIVE_KINDS:
            return kind
        return None
    except Exception:
        return None


def maybe_fetch_live_data(
    query: str,
    profile: Optional[dict] = None,
    live_kind: Optional[str] = None,
    slots: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    if not _LIVE_ENABLED:
        return None

    kind = str(live_kind or "").strip().lower() if live_kind else classify_live_data_kind(query)
    if not kind:
        return None
    if kind not in _ALLOWED_LIVE_KINDS:
        return None

    if _looks_unsafe(query):
        return {
            "kind": kind,
            "status": "blocked",
            "message": "I can help with finance and safe informational updates, but I can't help with that request.",
            "facts": [],
            "sources": [],
            "as_of": _now_utc_iso(),
        }

    slots = slots or {}

    if kind == "weather":
        return _fetch_weather(query=query, profile=profile or {}, location_hint=slots.get("location"))
    if kind == "stock":
        return _fetch_stock_quote(
            query=query,
            symbol_hint=slots.get("ticker"),
            company_hint=slots.get("company"),
        )
    if kind == "stock_history":
        return _fetch_stock_history(
            query=query,
            symbol_hint=slots.get("ticker"),
            company_hint=slots.get("company"),
            window_days_hint=slots.get("window_days"),
        )
    if kind == "news":
        return _fetch_headlines(query=query, topic_hint=slots.get("topic"))
    return None


def _fetch_weather(query: str, profile: dict, location_hint: Optional[str] = None) -> dict[str, Any]:
    location = _to_text(location_hint) or _extract_location(query) or _to_text(profile.get("city"))
    if not location:
        return {
            "kind": "weather",
            "status": "needs_input",
            "message": "I can check that. Please share your city name.",
            "facts": [],
            "sources": [],
            "as_of": _now_utc_iso(),
        }

    try:
        resolved_location, top = _geocode_with_retry(location)
        results = [top] if top else []
        if not results:
            return {
                "kind": "weather",
                "status": "needs_input",
                "message": f"I couldn't find '{location}'. Please share a nearby city name.",
                "facts": [],
                "sources": ["https://open-meteo.com/"],
                "as_of": _now_utc_iso(),
            }

        lat = top.get("latitude")
        lon = top.get("longitude")
        place_name = top.get("name") or resolved_location
        country = top.get("country_code") or top.get("country")

        forecast = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,apparent_temperature,relative_humidity_2m,wind_speed_10m",
                "timezone": "auto",
            },
            timeout=_HTTP_TIMEOUT_SECONDS,
        )
        forecast.raise_for_status()
        data = forecast.json()
        current = data.get("current") or {}

        facts = [
            f"Location: {place_name}{', ' + country if country else ''}",
            f"Temperature: {current.get('temperature_2m')} C",
            f"Feels like: {current.get('apparent_temperature')} C",
            f"Humidity: {current.get('relative_humidity_2m')}%",
            f"Wind speed: {current.get('wind_speed_10m')} km/h",
        ]

        return {
            "kind": "weather",
            "status": "ok",
            "message": None,
            "facts": facts,
            "sources": ["https://open-meteo.com/"],
            "as_of": _now_utc_iso(),
        }
    except Exception:
        return {
            "kind": "weather",
            "status": "error",
            "message": "I couldn't fetch live weather right now. Please try again shortly.",
            "facts": [],
            "sources": ["https://open-meteo.com/"],
            "as_of": _now_utc_iso(),
        }


def _geocode_with_retry(location: str) -> tuple[str, dict | None]:
    text = _to_text(location) or ""
    has_non_latin = bool(re.search(r"[؀-ۿऀ-ॿ]", text))

    # For non-Latin inputs, transliterate first to reduce false matches like Dal? (IR/CN) for Delhi.
    if has_non_latin:
        transliterated = _transliterate_location(location)
        if transliterated and transliterated.lower() != location.lower():
            retry = _geocode_location(transliterated)
            if retry:
                return transliterated, retry

    direct = _geocode_location(location)
    if direct:
        return location, direct

    # Latin inputs can still benefit from one normalization retry.
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
    geo_data = geo.json()
    results = geo_data.get("results") or []
    if not results:
        return None

    # Prefer highest-population match for ambiguous names.
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

    # Skip rewrite for already-Latin inputs.
    if re.search(r"[A-Za-z]", text) and not re.search(r"[\u0600-\u06FF\u0900-\u097F]", text):
        return text

    prompt = (
        "Normalize this city/place name into standard English (Latin script) for geocoding.\n"
        "Return STRICT JSON only with key 'location'.\n"
        f"Input: {text}"
    )
    try:
        raw = generate_response(prompt, operation="turn_control")
        payload = _extract_json_object(raw)
        if not isinstance(payload, dict):
            return None
        candidate = _to_text(payload.get("location"))
        return candidate
    except Exception:
        return None


def _fetch_stock_quote(
    query: str,
    symbol_hint: Optional[str] = None,
    company_hint: Optional[str] = None,
) -> dict[str, Any]:
    symbol = _to_text(symbol_hint) or _extract_symbol(query)
    company_hint = _to_text(company_hint) or _extract_company_hint(query)
    if not symbol and not company_hint:
        return {
            "kind": "stock",
            "status": "needs_input",
            "message": "Please share a stock ticker or company name (for example, TCS or RELIANCE).",
            "facts": [],
            "sources": [],
            "as_of": _now_utc_iso(),
        }

    try:
        # Prefer NSE route first for Indian equity coverage.
        nse_symbol = _normalize_nse_symbol(symbol) or _resolve_nse_symbol(company_hint)
        if nse_symbol:
            nse_quote = _fetch_stock_from_nse(nse_symbol)
            if nse_quote:
                return nse_quote

        resolved_symbol = symbol or _resolve_symbol(company_hint)
        stooq_quote = _fetch_stock_from_stooq(resolved_symbol or company_hint)
        if stooq_quote:
            return stooq_quote

        if not resolved_symbol and not nse_symbol:
            return {
                "kind": "stock",
                "status": "needs_input",
                "message": "I couldn't identify that stock. Please share the exact ticker.",
                "facts": [],
                "sources": ["https://finance.yahoo.com/"],
                "as_of": _now_utc_iso(),
            }

        return {
            "kind": "stock",
            "status": "needs_input",
            "message": f"I couldn't fetch quote data for {resolved_symbol}. Please verify the ticker.",
            "facts": [],
            "sources": ["https://www.nseindia.com/", "https://stooq.com/"],
            "as_of": _now_utc_iso(),
        }
    except Exception:
        return {
            "kind": "stock",
            "status": "error",
            "message": "I couldn't fetch live stock data right now. Please try again shortly.",
            "facts": [],
            "sources": ["https://www.nseindia.com/", "https://stooq.com/"],
            "as_of": _now_utc_iso(),
        }


def _coerce_window_days(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        days = int(float(value))
    except (TypeError, ValueError):
        return None
    if days < 7 or days > 3650:
        return None
    return days


def _fetch_stock_history(
    query: str,
    symbol_hint: Optional[str] = None,
    company_hint: Optional[str] = None,
    window_days_hint: Optional[int] = None,
) -> dict[str, Any]:
    symbol = _to_text(symbol_hint) or _extract_symbol(query)
    company = _to_text(company_hint) or _extract_company_hint(query)
    if not symbol and not company:
        return {
            "kind": "stock_history",
            "status": "needs_input",
            "message": "Please share a stock ticker or company name for historical performance (for example, TCS or RELIANCE).",
            "facts": [],
            "sources": [],
            "as_of": _now_utc_iso(),
        }

    try:
        nse_symbol = _normalize_nse_symbol(symbol) or _resolve_nse_symbol(company)
        resolved_symbol = _to_text(symbol) or _resolve_symbol(company)
        yahoo_symbol = resolved_symbol

        if not yahoo_symbol and nse_symbol:
            yahoo_symbol = f"{nse_symbol}.NS"
        elif nse_symbol and yahoo_symbol and "." not in yahoo_symbol and not yahoo_symbol.endswith("NS"):
            yahoo_symbol = f"{nse_symbol}.NS"

        if not yahoo_symbol:
            return {
                "kind": "stock_history",
                "status": "needs_input",
                "message": "I couldn't identify that stock. Please share the exact ticker.",
                "facts": [],
                "sources": ["https://finance.yahoo.com/"],
                "as_of": _now_utc_iso(),
            }

        window_days = _coerce_window_days(window_days_hint) or 1095
        end_dt = dt.datetime.utcnow()
        start_dt = end_dt - dt.timedelta(days=window_days)

        history = requests.get(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{quote_plus(yahoo_symbol)}",
            params={
                "interval": "1d",
                "period1": int(start_dt.timestamp()),
                "period2": int(end_dt.timestamp()),
                "events": "history",
            },
            timeout=max(6, _HTTP_TIMEOUT_SECONDS),
        )
        history.raise_for_status()
        payload = history.json()
        result = ((payload or {}).get("chart") or {}).get("result") or []
        if not result:
            return {
                "kind": "stock_history",
                "status": "needs_input",
                "message": f"I couldn't fetch historical data for {yahoo_symbol}. Please verify the ticker.",
                "facts": [],
                "sources": ["https://finance.yahoo.com/"],
                "as_of": _now_utc_iso(),
            }

        quote_data = ((((result[0] or {}).get("indicators") or {}).get("quote")) or [{}])[0]
        closes = quote_data.get("close") or []
        series = []
        for value in closes:
            try:
                if value is None:
                    continue
                series.append(float(value))
            except (TypeError, ValueError):
                continue

        if len(series) < 2:
            return {
                "kind": "stock_history",
                "status": "needs_input",
                "message": f"I couldn't fetch enough historical points for {yahoo_symbol}. Try another ticker or range.",
                "facts": [],
                "sources": ["https://finance.yahoo.com/"],
                "as_of": _now_utc_iso(),
            }

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
        return {
            "kind": "stock_history",
            "status": "ok",
            "message": None,
            "facts": facts,
            "sources": ["https://finance.yahoo.com/"],
            "as_of": _now_utc_iso(),
        }
    except Exception:
        return {
            "kind": "stock_history",
            "status": "error",
            "message": "I couldn't fetch historical stock data right now. Please try again shortly.",
            "facts": [],
            "sources": ["https://finance.yahoo.com/"],
            "as_of": _now_utc_iso(),
        }


def _fetch_headlines(query: str, topic_hint: Optional[str] = None) -> dict[str, Any]:
    topic = _to_text(topic_hint) or _extract_news_topic(query)
    if not topic:
        topic = "india finance"

    try:
        rss_url = f"https://news.google.com/rss/search?q={quote_plus(topic)}"
        resp = requests.get(rss_url, timeout=_HTTP_TIMEOUT_SECONDS)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)

        items = root.findall(".//item")
        headlines = []
        links = []
        for item in items[:_MAX_HEADLINES]:
            title = _to_text(item.findtext("title"))
            link = _to_text(item.findtext("link"))
            pub_date = _to_text(item.findtext("pubDate"))
            if title:
                headlines.append(f"{title}{f' ({pub_date})' if pub_date else ''}")
            if link:
                links.append(link)

        if not headlines:
            return {
                "kind": "news",
                "status": "needs_input",
                "message": "I couldn't find fresh headlines for that topic. Try a more specific query.",
                "facts": [],
                "sources": ["https://news.google.com/"],
                "as_of": _now_utc_iso(),
            }

        facts = [f"Topic: {topic}"] + [f"Headline {i+1}: {h}" for i, h in enumerate(headlines)]
        return {
            "kind": "news",
            "status": "ok",
            "message": None,
            "facts": facts,
            "sources": links[:_MAX_HEADLINES] or ["https://news.google.com/"],
            "as_of": _now_utc_iso(),
        }
    except Exception:
        return {
            "kind": "news",
            "status": "error",
            "message": "I couldn't fetch live headlines right now. Please try again shortly.",
            "facts": [],
            "sources": ["https://news.google.com/"],
            "as_of": _now_utc_iso(),
        }


def _extract_location(query: str) -> Optional[str]:
    text = (query or "").strip()
    if not text:
        return None

    prompt = (
        "Extract the weather location from the user message.\n"
        "Return a location only if user explicitly mentions a city/place/region.\n"
        "If the user asks generic weather or language preference/style (for example Hindi/English), return null.\n"
        "If location is missing or unclear, return null.\n\n"
        f"User message: {text}\n\n"
        "Return STRICT JSON only with key 'location' (string or null)."
    )

    try:
        raw = generate_response(prompt, operation="turn_control")
        payload = _extract_json_object(raw)
        if not isinstance(payload, dict):
            return None

        value = payload.get("location")
        if value is None:
            return None

        location = _to_text(value)
        if not location:
            return None

        location = re.sub(r"\s+", " ", location).strip(" .,-")
        if len(location) > 80:
            location = location[:80].strip()
        return location or None
    except Exception:
        return None


def _extract_symbol(query: str) -> Optional[str]:
    text = (query or "").strip()
    if not text:
        return None
    # Capture ticker-like tokens, including .NS / .BO suffixes.
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


def _resolve_symbol(company_hint: Optional[str]) -> Optional[str]:
    hint = _to_text(company_hint)
    if not hint:
        return None

    resp = requests.get(
        "https://query1.finance.yahoo.com/v1/finance/search",
        params={"q": hint, "quotesCount": 1, "newsCount": 0},
        timeout=_HTTP_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    data = resp.json()
    quotes = (data or {}).get("quotes") or []
    if not quotes:
        return None
    symbol = _to_text(quotes[0].get("symbol"))
    return symbol


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
        data = resp.json()
        symbols = data.get("symbols") or []
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
        data = resp.json()
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

        return {
            "kind": "stock",
            "status": "ok",
            "message": None,
            "facts": facts,
            "sources": ["https://www.nseindia.com/"],
            "as_of": _now_utc_iso(),
        }
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
            timeout=max(6, _HTTP_TIMEOUT_SECONDS),
        )
        resp.raise_for_status()
        lines = [line.strip() for line in resp.text.splitlines() if line.strip()]
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
        return {
            "kind": "stock",
            "status": "ok",
            "message": None,
            "facts": facts,
            "sources": ["https://stooq.com/"],
            "as_of": _now_utc_iso(),
        }
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


def _extract_news_topic(query: str) -> Optional[str]:
    text = (query or "").strip()
    if not text:
        return None
    patterns = [
        r"(?:news on|update on|headlines on)\s+(.+)$",
        r"(?:latest news about|latest news on)\s+(.+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            topic = match.group(1).strip(" .,-")
            if topic:
                return topic
    # If generic "latest news", keep finance-default topic.
    if "news" in text.lower():
        return "india finance"
    return None


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
