"""Benchmark sintético de tool calling com verificador determinístico (sem LLM juiz).

Tarefas geradas por sementes fixas a partir de 11 ferramentas com esquema JSON:
  single       uma chamada certa entre 4 ferramentas oferecidas (1 correta + 3 distratoras)
  parallel     duas chamadas da mesma ferramenta com valores diferentes ("Paris e Tóquio")
  irrelevance  nenhuma ferramenta serve -> o certo é NÃO chamar
Splits: dev (sementes 1000+i) e test (sementes 900000+i, TRANCADO: só via tool_eval --test, que
registra no ledger). Formato de chamada: o XML do chat template do Qwen3.5
(<tool_call><function=nome><parameter=arg>valor</parameter></function></tool_call>).
"""

from __future__ import annotations

import random
import re

CITIES = ["Paris", "Tokyo", "Lima", "Cairo", "Oslo", "Toronto", "Nairobi", "Lisbon", "Seoul", "Denver",
          "Recife", "Madrid", "Hanoi", "Dublin", "Santiago", "Mumbai", "Prague", "Austin", "Accra", "Perth"]
CURR = ["USD", "EUR", "BRL", "JPY", "GBP", "CAD", "INR", "CHF"]
TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "TSLA", "META", "PETR4", "VALE3", "NFLX"]
LABELS = ["gym", "meeting", "medicine", "flight", "school pickup", "standup"]
TITLES = ["Team sync", "Dentist", "Project review", "Lunch with Ana", "Budget planning", "Yoga class"]
RESTAURANTS = ["Mangiare", "Sushi Ko", "La Brasa", "Green Fork", "Casa Lima", "Blue Door"]
ROOMS = ["living room", "bedroom", "office", "kitchen"]
UNITS = [("km", "miles"), ("kg", "pounds"), ("liters", "gallons"), ("celsius", "fahrenheit"), ("meters", "feet")]


def _date(r):
    m = r.choice([12, 1, 2, 3])
    return f"{2026 if m == 12 else 2027}-{m:02d}-{r.randint(1, 28):02d}"


def _time(r):
    return f"{r.randint(5, 22):02d}:{r.choice(['00', '15', '30', '45'])}"


def _amount(r):
    return r.choice([r.randint(5, 900), r.randint(5, 900) + 0.5])


def _fmt(x):
    return str(int(x)) if float(x).is_integer() else str(x)


def P(t, desc, enum=None):
    d = {"type": t, "description": desc}
    if enum:
        d["enum"] = enum
    return d


# nome -> (descrição, propriedades, gerador(r) -> (args, [templates]))
TOOLS = {
    "get_weather": ("Get the current weather for a city.",
                    {"city": P("string", "City name"), "unit": P("string", "Temperature unit", ["celsius", "fahrenheit"])},
                    lambda r: ({"city": r.choice(CITIES), "unit": r.choice(["celsius", "fahrenheit"])},
                               ["What's the weather like in {city} right now? Give it in {unit}.",
                                "Tell me the current temperature in {city}, in {unit} please.",
                                "I'm heading to {city} today. How's the weather there? ({unit})"])),
    "convert_currency": ("Convert an amount of money between two currencies (ISO codes).",
                         {"amount": P("number", "Amount to convert"), "from_currency": P("string", "ISO code"),
                          "to_currency": P("string", "ISO code")},
                         lambda r: (dict(zip(["from_currency", "to_currency"], r.sample(CURR, 2)), amount=_amount(r)),
                                    ["How much is {amount} {from_currency} in {to_currency}?",
                                     "Convert {amount} {from_currency} to {to_currency}.",
                                     "I have {amount} {from_currency}. What is that worth in {to_currency}?"])),
    "set_alarm": ("Set an alarm at a time (HH:MM, 24h) with a label.",
                  {"time": P("string", "HH:MM, 24h"), "label": P("string", "Alarm label")},
                  lambda r: ({"time": _time(r), "label": r.choice(LABELS)},
                             ["Set an alarm for {time} labeled '{label}'.",
                              "Wake me up at {time}; call the alarm '{label}'.",
                              "Please create a '{label}' alarm at {time}."])),
    "search_flights": ("Search flights between two cities on a date (YYYY-MM-DD).",
                       {"origin": P("string", "Departure city"), "destination": P("string", "Arrival city"),
                        "date": P("string", "YYYY-MM-DD")},
                       lambda r: (dict(zip(["origin", "destination"], r.sample(CITIES, 2)), date=_date(r)),
                                  ["Find me flights from {origin} to {destination} on {date}.",
                                   "Are there flights {origin} -> {destination} on {date}?",
                                   "I need to fly from {origin} to {destination}, departing {date}."])),
    "calculate_tip": ("Compute the tip for a restaurant bill.",
                      {"bill_amount": P("number", "Bill total"), "tip_percent": P("number", "Tip percentage")},
                      lambda r: ({"bill_amount": _amount(r), "tip_percent": r.choice([10, 12, 15, 18, 20, 25])},
                                 ["What's a {tip_percent}% tip on a bill of {bill_amount}?",
                                  "Our bill came to {bill_amount}. How much is a {tip_percent} percent tip?"])),
    "create_event": ("Create a calendar event.",
                     {"title": P("string", "Event title"), "date": P("string", "YYYY-MM-DD"),
                      "duration_minutes": P("integer", "Duration in minutes")},
                     lambda r: ({"title": r.choice(TITLES), "date": _date(r), "duration_minutes": r.choice([15, 30, 45, 60, 90, 120])},
                                ["Schedule '{title}' on {date} for {duration_minutes} minutes.",
                                 "Add an event called '{title}' to my calendar on {date}, lasting {duration_minutes} minutes."])),
    "get_stock_price": ("Get the latest price of a stock by ticker.",
                        {"ticker": P("string", "Ticker symbol")},
                        lambda r: ({"ticker": r.choice(TICKERS)},
                                   ["What's {ticker} trading at right now?", "Get me the current stock price of {ticker}.",
                                    "How much is one share of {ticker} today?"])),
    "convert_units": ("Convert a value between measurement units.",
                      {"value": P("number", "Value"), "from_unit": P("string", "Source unit"), "to_unit": P("string", "Target unit")},
                      lambda r: (dict(zip(["from_unit", "to_unit"], r.choice(UNITS)), value=_amount(r)),
                                 ["Convert {value} {from_unit} to {to_unit}.", "How many {to_unit} is {value} {from_unit}?"])),
    "get_distance": ("Get travel distance between two places by mode.",
                     {"origin": P("string", "Start"), "destination": P("string", "End"),
                      "mode": P("string", "Travel mode", ["driving", "walking", "transit"])},
                     lambda r: (dict(zip(["origin", "destination"], r.sample(CITIES, 2)), mode=r.choice(["driving", "walking", "transit"])),
                                ["How far is it from {origin} to {destination} by {mode}?",
                                 "Distance from {origin} to {destination}, {mode}?"])),
    "set_thermostat": ("Set the thermostat temperature (celsius) in a room.",
                       {"room": P("string", "Room name"), "temperature": P("number", "Target temperature in celsius")},
                       lambda r: ({"room": r.choice(ROOMS), "temperature": r.choice([18, 19, 20, 21, 22, 23, 24, 22.5])},
                                  ["Set the {room} to {temperature} degrees.", "Make the {room} thermostat {temperature} °C."])),
    "book_table": ("Book a restaurant table.",
                   {"restaurant": P("string", "Restaurant name"), "party_size": P("integer", "Number of people"),
                    "time": P("string", "HH:MM, 24h")},
                   lambda r: ({"restaurant": r.choice(RESTAURANTS), "party_size": r.randint(1, 10), "time": _time(r)},
                              ["Book a table at {restaurant} for {party_size} at {time}.",
                               "Can you reserve {restaurant} for {party_size} people, {time}?"])),
}
PARALLEL = {  # ferramenta -> (campo que varia, template com {a} e {b} e demais campos)
    "get_weather": ("city", "What's the weather in {a} and in {b}? Use {unit} for both."),
    "get_stock_price": ("ticker", "Give me the current prices of {a} and {b}."),
}
IRRELEVANT = ["Who wrote Don Quixote?", "What is 17 times 23?", "Write a haiku about the rain.",
              "Explain in one sentence what photosynthesis is.", "What's the capital of Australia?",
              "Give me a synonym for 'happy'.", "How many legs does a spider have?",
              "Translate 'good morning' into Spanish."]


def schema(name):
    desc, props, _ = TOOLS[name]
    return {"type": "function", "function": {"name": name, "description": desc,
                                             "parameters": {"type": "object", "properties": props,
                                                            "required": list(props)}}}


def task(seed: int) -> dict:
    r = random.Random(seed)
    kind = r.choices(["single", "parallel", "irrelevance"], [0.6, 0.2, 0.2])[0]
    names = list(TOOLS)
    if kind == "single":
        gold_tool = r.choice(names)
        args, tpls = TOOLS[gold_tool][2](r)
        user = r.choice(tpls).format(**{k: _fmt(v) if isinstance(v, (int, float)) else v for k, v in args.items()})
        calls = [{"name": gold_tool, "arguments": args}]
    elif kind == "parallel":
        gold_tool = r.choice(list(PARALLEL))
        field, tpl = PARALLEL[gold_tool]
        a1, _ = TOOLS[gold_tool][2](r)
        pool = CITIES if field == "city" else TICKERS
        va, vb = r.sample(pool, 2)
        calls = [{"name": gold_tool, "arguments": {**a1, field: v}} for v in (va, vb)]
        user = tpl.format(a=va, b=vb, **{k: v for k, v in a1.items() if k != field})
    else:
        gold_tool = None
        user = r.choice(IRRELEVANT)
        calls = []
    others = [n for n in names if n != gold_tool]
    offered = ([gold_tool] if gold_tool else []) + r.sample(others, 4 if gold_tool is None else 3)
    r.shuffle(offered)
    return {"seed": seed, "kind": kind, "user": user, "tools": [schema(n) for n in offered], "gold": calls}


def split(name: str, n: int) -> list[dict]:
    base = {"dev": 1000, "test": 900000}[name]
    return [task(base + i) for i in range(n)]


CALL_RE = re.compile(r"<tool_call>\s*<function=([^>\n]+)>(.*?)</function>\s*</tool_call>", re.S)
PARAM_RE = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.S)


def parse(text: str) -> tuple[list[dict], bool]:
    """-> (chamadas, formato_ok). Ignora o raciocínio antes de </think>."""
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    calls = [{"name": m.group(1).strip(), "arguments": {k.strip(): v.strip() for k, v in PARAM_RE.findall(m.group(2))}}
             for m in CALL_RE.finditer(text)]
    ok = text.count("<tool_call>") == len(calls)
    return calls, ok


def _norm(v, typ):
    s = str(v).strip().strip("\"'").strip()
    if typ in ("number", "integer"):
        try:
            return round(float(s.replace(",", "")), 6)
        except ValueError:
            return ("bad", s)
    return s.casefold()


def _key(call, t):
    props = TOOLS[call["name"]][1] if call["name"] in TOOLS else {}
    args = tuple(sorted((k, _norm(v, props.get(k, {}).get("type", "string"))) for k, v in call["arguments"].items()))
    return call["name"], args


def score(t: dict, text: str, cut: bool) -> dict:
    calls, fmt_ok = parse(text)
    gold = t["gold"]
    exact = sorted(map(repr, (_key(c, t) for c in calls))) == sorted(map(repr, (_key(c, t) for c in gold)))
    names_ok = sorted(c["name"] for c in calls) == sorted(c["name"] for c in gold)
    return {"kind": t["kind"], "n_calls": len(calls), "format_ok": fmt_ok and not cut, "name_ok": names_ok,
            "exact": exact and fmt_ok and not cut}
