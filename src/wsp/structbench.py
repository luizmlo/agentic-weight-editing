"""Triagem do segundo defeito: saída estruturada com verificador determinístico (sem LLM juiz).

Três famílias, geradas por sementes fixas (dev: 2000+i; test: 800000+i, trancado):
  json      extrair um registro de um texto em prosa para um JSON Schema com tipos, enum, lista, objeto aninhado
            e campos ausentes (null). Mede separadamente: parse, conformidade ao schema (estrutura/tipos) e valores.
  tools     tool calling no formato nativo do Qwen3.5, mais difícil que wsp.toolbench: 6 ferramentas oferecidas,
            parâmetros inteiros/booleanos/enum/lista, parâmetros opcionais (só passar se pedido) e pedidos com duas
            ferramentas diferentes.
  ifeval    pedido curto + 1–2 restrições de formato verificáveis (minúsculas, nº exato de itens, limite de
            palavras, frase final, sem vírgulas, resposta embrulhada em JSON).
"""

from __future__ import annotations

import json
import random
import re

# ----------------------------------------------------------------------------------------------- json
FIRST = ["Ana", "Bruno", "Carla", "Diego", "Elena", "Felipe", "Grace", "Hiro", "Ines", "Jonas", "Kofi", "Lia"]
LAST = ["Silva", "Okafor", "Tanaka", "Moreau", "Novak", "Costa", "Reyes", "Larsen", "Haddad", "Kim"]
CITIES = ["Lisbon", "Recife", "Osaka", "Denver", "Nairobi", "Prague", "Lima", "Perth", "Dublin", "Hanoi"]
STREETS = ["Oak Street", "Rua das Flores", "Maple Avenue", "Harbor Road", "Calle Mayor", "Elm Lane"]
PRODUCTS = ["desk lamp", "wireless mouse", "notebook", "coffee grinder", "yoga mat", "backpack", "kettle",
            "phone case", "water bottle", "monitor stand"]
TAGS = ["urgent", "gift", "fragile", "eco", "bulk", "repeat-customer", "international", "promo"]
AMENITIES = ["balcony", "elevator", "parking", "pool", "gym", "laundry", "air conditioning", "doorman"]
SKILLS = ["Python", "SQL", "Excel", "Figma", "Kubernetes", "Spanish", "accounting", "React", "statistics"]


def _price(r):
    return round(r.randint(5, 400) + r.choice([0, 0.5, 0.99, 0.25]), 2)


def _rec_order(r):
    items = r.sample(PRODUCTS, r.randint(1, 3))
    qty = {p: r.randint(1, 5) for p in items}
    gift = r.random() < 0.5
    coupon = r.choice([None, None, "SAVE10", "WELCOME", "BLACKFRI"])
    rec = {"order_id": r.randint(10000, 99999), "customer": f"{r.choice(FIRST)} {r.choice(LAST)}",
           "status": r.choice(["pending", "shipped", "delivered", "cancelled"]),
           "items": [{"product": p, "quantity": qty[p]} for p in items], "total": _price(r), "gift": gift,
           "coupon": coupon, "shipping": {"city": r.choice(CITIES), "express": r.random() < 0.5}}
    s = [f"Order #{rec['order_id']} was placed by {rec['customer']}.",
         "It contains " + ", ".join(f"{q} x {p}" for p, q in qty.items()) + ".",
         f"The order total came to ${rec['total']}.",
         f"Right now the order is {rec['status']}.",
         ("The customer asked for it to be gift-wrapped." if gift else "No gift wrapping was requested."),
         (f"The coupon {coupon} was applied." if coupon else ""),
         f"It ships to {rec['shipping']['city']}" + (" with express delivery." if rec["shipping"]["express"] else " by standard mail."),
         r.choice(["The customer also left a five-star review last month.", "Support was contacted once about a previous order.", ""])]
    sch = {"type": "object", "properties": {
        "order_id": {"type": "integer"}, "customer": {"type": "string"},
        "status": {"type": "string", "enum": ["pending", "shipped", "delivered", "cancelled"]},
        "items": {"type": "array", "items": {"type": "object", "properties": {
            "product": {"type": "string"}, "quantity": {"type": "integer"}}, "required": ["product", "quantity"]}},
        "total": {"type": "number"}, "gift": {"type": "boolean"}, "coupon": {"type": ["string", "null"]},
        "shipping": {"type": "object", "properties": {"city": {"type": "string"}, "express": {"type": "boolean"}},
                     "required": ["city", "express"]}},
        "required": ["order_id", "customer", "status", "items", "total", "gift", "coupon", "shipping"]}
    return rec, s, sch


def _rec_listing(r):
    am = r.sample(AMENITIES, r.randint(1, 4))
    pets = r.random() < 0.5
    floor = r.choice([None, r.randint(1, 20)])
    rec = {"city": r.choice(CITIES), "address": f"{r.randint(1, 999)} {r.choice(STREETS)}",
           "bedrooms": r.randint(0, 4), "area_m2": r.choice([r.randint(25, 180), round(r.randint(25, 180) + 0.5, 1)]),
           "rent": r.randint(400, 4000), "furnished": r.random() < 0.5, "pets_allowed": pets, "floor": floor,
           "amenities": am, "type": r.choice(["apartment", "house", "studio", "loft"])}
    s = [f"For rent: {'an' if rec['type'][0] in 'aeiou' else 'a'} {rec['type']} at {rec['address']}, {rec['city']}.",
         ("It has no separate bedroom." if rec["bedrooms"] == 0 else f"It has {rec['bedrooms']} bedroom{'s' if rec['bedrooms'] > 1 else ''}."),
         f"The total area is {rec['area_m2']} square meters.",
         f"Monthly rent is {rec['rent']} dollars.",
         ("It comes fully furnished." if rec["furnished"] else "It is unfurnished."),
         ("Pets are welcome." if pets else "Sorry, no pets."),
         (f"The unit is on floor {floor}." if floor else ""),
         "Amenities include " + ", ".join(am) + ".",
         r.choice(["The landlord prefers a 12-month lease.", "Viewings are on weekends.", ""])]
    sch = {"type": "object", "properties": {
        "city": {"type": "string"}, "address": {"type": "string"}, "bedrooms": {"type": "integer"},
        "area_m2": {"type": "number"}, "rent": {"type": "integer"}, "furnished": {"type": "boolean"},
        "pets_allowed": {"type": "boolean"}, "floor": {"type": ["integer", "null"]},
        "amenities": {"type": "array", "items": {"type": "string"}},
        "type": {"type": "string", "enum": ["apartment", "house", "studio", "loft"]}},
        "required": ["city", "address", "bedrooms", "area_m2", "rent", "furnished", "pets_allowed", "floor",
                     "amenities", "type"]}
    return rec, s, sch


def _rec_job(r):
    sk = r.sample(SKILLS, r.randint(1, 4))
    remote = r.choice(["remote", "hybrid", "on-site"])
    sal = r.choice([None, r.randint(30, 200) * 1000])
    rec = {"title": r.choice(["Data Analyst", "Backend Engineer", "Product Designer", "Accountant", "SRE", "Teacher"]),
           "company": r.choice(["Acme", "Nimbus", "Orbital", "Tallow & Co", "Brightside", "Quokka Labs"]),
           "location": {"city": r.choice(CITIES), "mode": remote}, "salary_usd": sal,
           "years_experience": r.randint(0, 10), "skills": sk, "full_time": r.random() < 0.7,
           "visa_sponsorship": r.random() < 0.3}
    s = [f"{rec['company']} is hiring a {rec['title']}.",
         f"The role is based in {rec['location']['city']} and is {remote}.",
         (f"The salary is {sal} US dollars per year." if sal else "Salary was not disclosed."),
         f"Candidates need at least {rec['years_experience']} years of experience." if rec["years_experience"] else "No prior experience is required.",
         "Required skills: " + ", ".join(sk) + ".",
         ("This is a full-time position." if rec["full_time"] else "This is a part-time position."),
         ("Visa sponsorship is available." if rec["visa_sponsorship"] else "The company cannot sponsor visas."),
         r.choice(["The team has twelve people.", "Interviews happen in three rounds.", ""])]
    sch = {"type": "object", "properties": {
        "title": {"type": "string"}, "company": {"type": "string"},
        "location": {"type": "object", "properties": {"city": {"type": "string"},
                                                      "mode": {"type": "string", "enum": ["remote", "hybrid", "on-site"]}},
                     "required": ["city", "mode"]},
        "salary_usd": {"type": ["integer", "null"]}, "years_experience": {"type": "integer"},
        "skills": {"type": "array", "items": {"type": "string"}}, "full_time": {"type": "boolean"},
        "visa_sponsorship": {"type": "boolean"}},
        "required": ["title", "company", "location", "salary_usd", "years_experience", "skills", "full_time",
                     "visa_sponsorship"]}
    return rec, s, sch


RECORDS = {"order": _rec_order, "listing": _rec_listing, "job": _rec_job}


def json_task(seed: int) -> dict:
    r = random.Random(seed)
    kind = r.choice(list(RECORDS))
    rec, sents, sch = RECORDS[kind](r)
    sents = [x for x in sents if x]
    head, rest = sents[0], sents[1:]
    r.shuffle(rest)
    text = " ".join([head] + rest)
    user = ("Extract the information from the text below into a JSON object that follows this JSON Schema. "
            "Use null for anything the text does not mention. Output only the JSON object.\n\n"
            f"Schema:\n{json.dumps(sch)}\n\nText:\n{text}")
    return {"family": "json", "seed": seed, "kind": kind, "user": user, "schema": sch, "gold": rec}


def _types(t):
    return t if isinstance(t, list) else [t]


def _type_ok(v, t):
    if t == "null":
        return v is None
    if t == "integer":
        return isinstance(v, int) and not isinstance(v, bool)
    if t == "number":
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    if t == "boolean":
        return isinstance(v, bool)
    if t == "string":
        return isinstance(v, str)
    if t == "array":
        return isinstance(v, list)
    if t == "object":
        return isinstance(v, dict)
    return False


def schema_errors(v, sch, path="$") -> list[str]:
    """Erros de estrutura/tipo (não de valor). Lista vazia = conforme."""
    if not any(_type_ok(v, t) for t in _types(sch["type"])):
        return [f"{path}: tipo {type(v).__name__}, esperado {sch['type']}"]
    errs = []
    if "enum" in sch and v not in sch["enum"]:
        errs.append(f"{path}: {v!r} fora do enum")
    if isinstance(v, dict) and "properties" in sch:
        for k in sch.get("required", []):
            if k not in v:
                errs.append(f"{path}.{k}: faltando")
        for k, x in v.items():
            if k not in sch["properties"]:
                errs.append(f"{path}.{k}: chave extra")
            else:
                errs += schema_errors(x, sch["properties"][k], f"{path}.{k}")
    if isinstance(v, list) and "items" in sch:
        for i, x in enumerate(v):
            errs += schema_errors(x, sch["items"], f"{path}[{i}]")
    return errs


def _eq(a, b):
    if isinstance(b, bool) or b is None:
        return a is b or a == b and type(a) is type(b)
    if isinstance(b, (int, float)):
        return isinstance(a, (int, float)) and not isinstance(a, bool) and abs(a - b) < 1e-6
    if isinstance(b, str):
        return isinstance(a, str) and a.strip().casefold() == b.strip().casefold()
    if isinstance(b, list):
        if not isinstance(a, list) or len(a) != len(b):
            return False
        if all(isinstance(x, str) for x in b):  # listas de strings: a ordem não importa
            return sorted(map(str.casefold, map(str, a))) == sorted(map(str.casefold, b))
        return all(_eq(x, y) for x, y in zip(a, b))
    if isinstance(b, dict):
        return isinstance(a, dict) and set(a) == set(b) and all(_eq(a[k], b[k]) for k in b)
    return a == b


FENCE = re.compile(r"^\s*```(?:json)?\s*\n?(.*?)\n?```\s*$", re.S)


def parse_json(text: str):
    """-> (objeto | None, modo): 'strict' (a saída inteira é JSON, cercas ``` toleradas), 'lenient' (JSON
    achado do primeiro '{' ao último '}', com texto em volta) ou 'fail'."""
    t = text.split("</think>", 1)[-1].strip()
    m = FENCE.match(t)
    body = m.group(1) if m else t
    try:
        return json.loads(body), "strict"
    except json.JSONDecodeError:
        pass
    i, j = t.find("{"), t.rfind("}")
    if i >= 0 and j > i:
        try:
            return json.loads(t[i:j + 1]), "lenient"
        except json.JSONDecodeError:
            pass
    return None, "fail"


def score_json(t: dict, text: str, cut: bool) -> dict:
    obj, mode = parse_json(text)
    errs = schema_errors(obj, t["schema"]) if obj is not None else ["não é JSON"]
    vals = {k: obj.get(k) for k in t["gold"]} if isinstance(obj, dict) else {}
    field_ok = [_eq(vals.get(k, "<faltando>"), v) for k, v in t["gold"].items()] if vals else [False] * len(t["gold"])
    return {"family": "json", "kind": t["kind"], "parse": mode, "json_ok": obj is not None and not cut,
            "schema_ok": obj is not None and not errs and not cut, "errors": errs[:6],
            "field_acc": sum(field_ok) / len(field_ok), "exact": obj is not None and not errs and all(field_ok) and not cut}


# ---------------------------------------------------------------------------------------------- tools
def P(t, desc, enum=None, items=None):
    d = {"type": t, "description": desc}
    if enum:
        d["enum"] = enum
    if items:
        d["items"] = items
    return d


# nome -> (descrição, propriedades, obrigatórios, gerador(r) -> (args, frase))
def _g_ticket(r):
    pr = r.choice(["low", "medium", "high"])
    a = {"title": r.choice(["Login fails on mobile", "Invoice PDF is blank", "Slow dashboard", "Typo on pricing page"]),
         "priority": pr}
    s = f"Open a support ticket titled '{a['title']}' with {pr} priority"
    if r.random() < 0.5:
        a["tags"] = r.sample(["mobile", "billing", "ui", "performance", "backend"], 2)
        s += f", tagged {a['tags'][0]} and {a['tags'][1]}"
    if r.random() < 0.4:
        a["notify_team"] = r.random() < 0.5
        s += ", and " + ("notify the team" if a["notify_team"] else "don't notify the team")
    return a, s + "."


def _g_transfer(r):
    a = {"amount": r.randint(10, 5000), "to_account": f"{r.randint(1000, 9999)}-{r.randint(0, 9)}"}
    s = f"Transfer {a['amount']} to account {a['to_account']}"
    if r.random() < 0.5:
        a["currency"] = r.choice(["USD", "EUR", "BRL"])
        s = f"Transfer {a['amount']} {a['currency']} to account {a['to_account']}"
    if r.random() < 0.4:
        a["scheduled_date"] = f"2027-0{r.randint(1, 9)}-{r.randint(10, 28)}"
        s += f" on {a['scheduled_date']}"
    return a, s + "."


def _g_lights(r):
    a = {"room": r.choice(["kitchen", "office", "bedroom", "garage"]), "on": r.random() < 0.6}
    s = f"Turn {'on' if a['on'] else 'off'} the {a['room']} lights"
    if a["on"] and r.random() < 0.6:
        a["brightness"] = r.choice([10, 25, 40, 50, 75, 90, 100])
        s += f" at {a['brightness']}% brightness"
    return a, s + "."


def _g_playlist(r):
    a = {"name": r.choice(["Road Trip", "Focus", "Sunday Morning", "Gym Mix"]),
         "songs": r.sample(["Yesterday", "Imagine", "Halo", "Roxanne", "Hurt", "Jolene", "Creep"], r.randint(2, 3))}
    s = f"Create a playlist called '{a['name']}' with " + ", ".join(a["songs"][:-1]) + f" and {a['songs'][-1]}"
    if r.random() < 0.4:
        a["public"] = r.random() < 0.5
        s += ", and make it " + ("public" if a["public"] else "private")
    return a, s + "."


def _g_reminder(r):
    a = {"text": r.choice(["call mom", "pay rent", "water the plants", "renew passport"]),
         "time": f"{r.randint(6, 21):02d}:{r.choice(['00', '30'])}"}
    s = f"Remind me to {a['text']} at {a['time']}"
    if r.random() < 0.5:
        a["repeat"] = r.choice(["daily", "weekly", "monthly"])
        s += f", repeating {a['repeat']}"
    return a, s + "."


def _g_search(r):
    a = {"query": r.choice(["running shoes", "standing desk", "noise cancelling headphones", "air fryer"])}
    s = f"Search the store for {a['query']}"
    if r.random() < 0.5:
        a["max_price"] = r.choice([50, 80, 120, 200, 350])
        s += f" under {a['max_price']} dollars"
    if r.random() < 0.4:
        a["in_stock_only"] = True
        s += ", only items in stock"
    if r.random() < 0.3:
        a["sort"] = r.choice(["price_asc", "price_desc", "rating"])
        s += {"price_asc": ", cheapest first", "price_desc": ", most expensive first", "rating": ", best rated first"}[a["sort"]]
    return a, s + "."


def _g_email(r):
    a = {"to": r.sample(["ana@example.com", "li@example.org", "sam@example.net", "kofi@example.com"], r.randint(1, 2)),
         "subject": r.choice(["Q3 report", "Lunch on Friday", "Contract draft", "Welcome aboard"])}
    s = "Send an email to " + " and ".join(a["to"]) + f" with the subject '{a['subject']}'"
    if r.random() < 0.4:
        a["high_importance"] = True
        s += ", marked as high importance"
    return a, s + "."


def _g_book(r):
    a = {"hotel": r.choice(["Grand Palace", "Sea Breeze Inn", "Hotel Aurora", "City Lodge"]),
         "nights": r.randint(1, 7), "guests": r.randint(1, 4)}
    s = f"Book {a['nights']} night{'s' if a['nights'] > 1 else ''} at {a['hotel']} for {a['guests']} guest{'s' if a['guests'] > 1 else ''}"
    if r.random() < 0.4:
        a["breakfast"] = r.random() < 0.5
        s += ", " + ("with breakfast" if a["breakfast"] else "without breakfast")
    return a, s + "."


HTOOLS = {
    "create_ticket": ("Open a support ticket.", {"title": P("string", "Ticket title"),
                      "priority": P("string", "Priority", ["low", "medium", "high"]),
                      "tags": P("array", "Optional tags", items={"type": "string"}),
                      "notify_team": P("boolean", "Whether to notify the team (default true)")}, ["title", "priority"], _g_ticket),
    "transfer_money": ("Transfer money to another account.", {"amount": P("integer", "Amount"),
                       "to_account": P("string", "Destination account"),
                       "currency": P("string", "ISO currency (default USD)", ["USD", "EUR", "BRL"]),
                       "scheduled_date": P("string", "YYYY-MM-DD; omit to send now")}, ["amount", "to_account"], _g_transfer),
    "set_lights": ("Control smart lights in a room.", {"room": P("string", "Room"), "on": P("boolean", "Turn on (true) or off (false)"),
                   "brightness": P("integer", "Brightness 0-100 (optional)")}, ["room", "on"], _g_lights),
    "create_playlist": ("Create a music playlist.", {"name": P("string", "Playlist name"),
                        "songs": P("array", "Song titles", items={"type": "string"}),
                        "public": P("boolean", "Public playlist (default false)")}, ["name", "songs"], _g_playlist),
    "create_reminder": ("Create a reminder.", {"text": P("string", "What to remember"), "time": P("string", "HH:MM, 24h"),
                        "repeat": P("string", "Repetition (optional)", ["daily", "weekly", "monthly"])}, ["text", "time"], _g_reminder),
    "search_products": ("Search the online store.", {"query": P("string", "Search terms"),
                        "max_price": P("integer", "Maximum price (optional)"),
                        "in_stock_only": P("boolean", "Only items in stock (default false)"),
                        "sort": P("string", "Sort order (optional)", ["price_asc", "price_desc", "rating"])}, ["query"], _g_search),
    "send_email": ("Send an email.", {"to": P("array", "Recipients", items={"type": "string"}), "subject": P("string", "Subject"),
                   "high_importance": P("boolean", "Mark as important (default false)")}, ["to", "subject"], _g_email),
    "book_hotel": ("Book a hotel room.", {"hotel": P("string", "Hotel name"), "nights": P("integer", "Number of nights"),
                   "guests": P("integer", "Number of guests"), "breakfast": P("boolean", "Include breakfast (optional)")},
                   ["hotel", "nights", "guests"], _g_book),
}


def tool_schema(name):
    desc, props, req, _ = HTOOLS[name]
    return {"type": "function", "function": {"name": name, "description": desc,
                                             "parameters": {"type": "object", "properties": props, "required": req}}}


def tools_task(seed: int) -> dict:
    r = random.Random(seed)
    kind = r.choices(["single", "double"], [0.6, 0.4])[0]
    names = r.sample(list(HTOOLS), 2 if kind == "double" else 1)
    calls, parts = [], []
    for n in names:
        a, s = HTOOLS[n][3](r)
        calls.append({"name": n, "arguments": a})
        parts.append(s)
    user = parts[0] if kind == "single" else parts[0][:-1] + ". Also, " + parts[1][0].lower() + parts[1][1:]
    offered = names + r.sample([n for n in HTOOLS if n not in names], 6 - len(names))
    r.shuffle(offered)
    return {"family": "tools", "seed": seed, "kind": kind, "user": user, "tools": [tool_schema(n) for n in offered],
            "gold": calls}


CALL_RE = re.compile(r"<tool_call>\s*<function=([^>\n]+)>(.*?)</function>\s*</tool_call>", re.S)
PARAM_RE = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.S)


def _coerce(v: str, typ: str):
    """Valor de <parameter> (texto) -> valor tipado, como um executor de ferramentas faria; ('bad', v) se não der."""
    s = v.strip()
    try:
        if typ == "integer":
            x = json.loads(s)
            return x if isinstance(x, int) and not isinstance(x, bool) else ("bad", s)
        if typ == "boolean":
            return {"true": True, "false": False}.get(s.lower(), ("bad", s))
        if typ == "array":
            x = json.loads(s)
            return x if isinstance(x, list) else ("bad", s)
    except json.JSONDecodeError:
        return ("bad", s)
    return s.strip("\"'").strip()


def score_tools(t: dict, text: str, cut: bool) -> dict:
    body = text.split("</think>", 1)[-1]
    raw = [(m.group(1).strip(), PARAM_RE.findall(m.group(2))) for m in CALL_RE.finditer(body)]
    fmt_ok = body.count("<tool_call>") == len(raw) and not cut
    calls, type_ok, extra_opt, missing_req = [], True, 0, 0
    for name, params in raw:
        props = HTOOLS[name][1] if name in HTOOLS else {}
        args = {}
        for k, v in params:
            k = k.strip()
            val = _coerce(v, props.get(k, {}).get("type", "string"))
            type_ok &= not (isinstance(val, tuple) and val and val[0] == "bad")
            args[k] = val
        calls.append({"name": name, "arguments": args})
    gold = {c["name"]: c["arguments"] for c in t["gold"]}
    got = {c["name"]: c["arguments"] for c in calls}
    names_ok = sorted(got) == sorted(gold) and len(calls) == len(t["gold"])
    for n, a in got.items():
        if n in gold:
            req = HTOOLS[n][2]
            extra_opt += sum(k not in gold[n] and k not in req for k in a)
            missing_req += sum(k not in a for k in req)
    exact = names_ok and all(_eq(got[n], gold[n]) for n in gold) and fmt_ok
    return {"family": "tools", "kind": t["kind"], "format_ok": fmt_ok, "name_ok": names_ok, "type_ok": bool(type_ok) and bool(calls),
            "extra_optional": extra_opt, "missing_required": missing_req, "exact": exact}


# --------------------------------------------------------------------------------------------- ifeval
TOPICS = ["how to keep houseplants alive", "why sleep matters", "tips for a first job interview",
          "how to start running", "what makes a good password", "how to save money on groceries",
          "the benefits of learning a second language", "how to prepare for a long flight",
          "how to make a simple tomato sauce", "how to study for an exam"]
CONSTRAINTS = {  # nome -> (instrução, verificador(texto, parâmetro))
    "lower": (lambda p: "Your entire response must be in lowercase letters; no capital letters at all.",
              lambda t, p: t == t.lower() and any(c.isalpha() for c in t)),
    "bullets": (lambda p: f"Your answer must contain exactly {p} bullet points, using markdown bullets such as '* item'.",
                lambda t, p: len(re.findall(r"^\s*[*-] ", t, re.M)) == p),
    "max_words": (lambda p: f"Answer in fewer than {p} words.", lambda t, p: len(t.split()) < p),
    "end_phrase": (lambda p: f"Finish your response with this exact phrase: '{p}'. No other words should follow it.",
                   lambda t, p: t.strip().rstrip("'\"").endswith(p)),
    "no_commas": (lambda p: "Do not use any commas in your entire response.", lambda t, p: "," not in t),
    "json_wrap": (lambda p: "Wrap your entire output in JSON format, as an object with a single key \"answer\".",
                  lambda t, p: isinstance(parse_json(t)[0], dict) and parse_json(t)[1] == "strict"),
}


def ifeval_task(seed: int) -> dict:
    r = random.Random(seed)
    names = r.sample(list(CONSTRAINTS), r.choice([1, 2]))
    if "json_wrap" in names and "bullets" in names:
        names.remove("bullets")
    params = {"bullets": r.randint(3, 5), "max_words": r.choice([40, 60, 80]),
              "end_phrase": r.choice(["Is there anything else I can help with?", "Good luck!", "That is all."])}
    cons = [(n, params.get(n)) for n in names]
    user = f"Write a short explanation of {r.choice(TOPICS)}. " + " ".join(CONSTRAINTS[n][0](p) for n, p in cons)
    return {"family": "ifeval", "seed": seed, "user": user, "constraints": cons}


def score_ifeval(t: dict, text: str, cut: bool) -> dict:
    body = text.split("</think>", 1)[-1].strip()
    ok = {n: bool(CONSTRAINTS[n][1](body, p)) and not cut for n, p in t["constraints"]}
    return {"family": "ifeval", "per": ok, "exact": all(ok.values())}


# ---------------------------------------------------------------------------------------------- comum
FAMILIES = {"json": (json_task, score_json), "tools": (tools_task, score_tools), "ifeval": (ifeval_task, score_ifeval)}
BASE_SEED = {"dev": 2000, "test": 800000}


def split(family: str, name: str, n: int, offset: int = 0) -> list[dict]:
    return [FAMILIES[family][0](BASE_SEED[name] + offset + i) for i in range(n)]


def score(t: dict, text: str, cut: bool) -> dict:
    return FAMILIES[t["family"]][1](t, text, cut)
