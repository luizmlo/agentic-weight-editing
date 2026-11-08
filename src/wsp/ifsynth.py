"""Prompts sintéticos no estilo IFEval para o baseline de fine-tuning do segundo defeito (análogo ao E4).

Tarefa de escrita (lista própria, escrita à mão, sem nenhum prompt do IFEval) + 1–3 instruções verificáveis sorteadas
com os próprios geradores do verificador oficial (`build_description()` de cada instrução, sem conflitos, como no
IFEval). Cada prompt guarda as instruções e os parâmetros, então a resposta pode ser checada com `check(item, texto)`.
"""

from __future__ import annotations

import random

from wsp.ifeval import instructions_registry as reg

TASKS = [
    "Write a blog post about the benefits of taking a walk every morning.",
    "Write an email to a coworker asking to move our Thursday meeting to Friday.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Write a product description for a reusable stainless steel water bottle.",
    "Explain how the gears on a bicycle make it easier to ride uphill.",
    "Write a cover letter for a junior data analyst position at a hospital.",
    "Write a review of a small family-owned pizza restaurant.",
    "Describe a perfect rainy Sunday at home.",
    "Write a speech for a high school graduation about curiosity.",
    "Write a poem about the first day of autumn.",
    "Write a guide for planting tomatoes on a balcony.",
    "Explain to a beginner what a spreadsheet formula is.",
    "Write a letter to your future self ten years from now.",
    "Write an advertisement for a neighborhood book swap.",
    "Write a summary of the main causes of traffic jams in big cities.",
    "Write a dialogue between a cat and a dog arguing about the best nap spot.",
    "Write a travel itinerary for a weekend in a mountain village.",
    "Write a complaint to a phone company about a billing error.",
    "Explain why the sky looks blue during the day.",
    "Write a motivational message for someone starting a new job.",
    "Write a description of a futuristic city powered only by solar energy.",
    "Write a thank-you note to a teacher who helped you learn to read.",
    "Write a short biography of an imaginary inventor named Clara Voss.",
    "Write a list of tips for staying focused while working from home.",
    "Write a song about a road trip with old friends.",
    "Write a pitch for a mobile app that helps people drink more water.",
    "Explain the difference between weather and climate.",
    "Write a fairy tale about a dragon who is afraid of the dark.",
    "Write a news report about a town that built a giant community garden.",
    "Write an essay on why public libraries still matter.",
    "Write instructions for making a cup of pour-over coffee.",
    "Write a social media post announcing the opening of a bakery.",
    "Write a riddle and then explain its answer.",
    "Write a job posting for a part-time dog walker.",
    "Describe your ideal workspace.",
    "Write a persuasive paragraph on why cities should have more bike lanes.",
    "Write a short mystery story set on a night train.",
    "Write a welcome message for new members of a chess club.",
    "Explain how vaccines train the immune system, for a twelve-year-old.",
    "Write a memo to employees about a new recycling policy.",
    "Write a menu description for a vegetarian lasagna.",
    "Write a eulogy for a beloved old sofa.",
    "Write a limerick about a forgetful wizard.",
    "Write a proposal for a school trip to a science museum.",
    "Describe the sound of a busy market in the morning.",
    "Write a FAQ for a small online shop that sells handmade candles.",
    "Write a story about a robot learning to bake bread.",
    "Write an apology letter for missing a friend's birthday party.",
    "Explain how to change a flat tire on a car.",
    "Write a short history of an imaginary island called Velmora.",
]
# instruções usadas (todas as do registro, exceto idioma: o verificador de idioma depende do langdetect e a
# resposta pedida em outra língua não é o defeito que estudamos)
IDS = [i for i in reg.INSTRUCTION_DICT if not i.startswith("language:")]


def make(seed: int) -> dict:
    r = random.Random(seed)
    random.seed(seed)  # os build_description() do IFEval sorteiam parâmetros com o `random` global
    task = r.choice(TASKS)
    chosen = []
    for _ in range(r.choice([1, 1, 2, 2, 3])):
        cands = [i for i in IDS if i not in chosen and not any(i in reg.INSTRUCTION_CONFLICTS.get(c, ()) or
                                                                c in reg.INSTRUCTION_CONFLICTS.get(i, ()) for c in chosen)]
        if cands:
            chosen.append(r.choice(cands))
    descs, kwargs = [], []
    for iid in chosen:
        ins = reg.INSTRUCTION_DICT[iid](iid)
        if iid == "combination:repeat_prompt":
            d = ins.build_description(prompt_to_repeat=task)
        else:
            d = ins.build_description()
        descs.append(d)
        kwargs.append({k: v for k, v in (ins.get_instruction_args() or {}).items() if v is not None})
    prompt = task + " " + " ".join(descs)
    return {"seed": seed, "prompt": prompt, "ids": chosen, "kwargs": kwargs}


def check(item: dict, text: str) -> list[bool]:
    out = []
    for iid, kw in zip(item["ids"], item["kwargs"]):
        ins = reg.INSTRUCTION_DICT[iid](iid)
        ins.build_description(**kw)
        out.append(bool(text.strip()) and bool(ins.check_following(text)))
    return out
