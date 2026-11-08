"""Sondas teacher-forced para o segundo defeito (seguir instruções de formato, IFEval, sem raciocínio).

Pares casados: o MESMO texto de resposta em duas versões, uma que cumpre a instrução e outra que não cumpre
(ex.: o parágrafo em minúsculas vs. o original; com e sem a frase final pedida). A margem de um par é
    m = logp médio por token (versão que cumpre) − logp médio por token (versão que não cumpre)
e a sonda mede quanto a edição muda m em relação ao modelo base.

  constrained   prompt COM a instrução: queremos Δm > 0 (a edição passa a preferir cumprir)
  control       o mesmo par com o prompt SEM a instrução: Δm deve ficar ~0. Se subir, a edição está aplicando o
                formato sempre (ex.: escrever tudo em minúsculas), não seguindo a instrução.

Respostas escritas à mão (sem geração), 8 instruções × tópicos. Tópicos de busca e do holdout lacrado são disjuntos.
Nenhum prompt aqui vem do IFEval.
"""

from __future__ import annotations

import json

import torch

PARAS = {  # tópico -> (pedido, resposta curta)
    "plants": ("Write a short paragraph on how to keep houseplants alive.",
               "Most houseplants die from too much care, not too little. Water only when the top inch of soil is dry, "
               "and make sure the pot has a drainage hole. Place the plant where it gets bright, indirect light, and "
               "rotate it every few weeks so it grows evenly. Wipe dust off the leaves, and feed it a diluted "
               "fertilizer in spring and summer."),
    "sleep": ("Write a short paragraph explaining why sleep matters.",
              "Sleep is when the body repairs itself and the brain sorts the day's memories. Adults who sleep less than "
              "seven hours a night are more likely to get sick, gain weight, and make mistakes at work. A regular "
              "bedtime, a dark room, and less screen time in the evening are simple ways to sleep better."),
    "interview": ("Write a short paragraph with advice for a first job interview.",
                  "Research the company before you go, and prepare two or three stories that show what you can do. "
                  "Arrive ten minutes early, dress one step more formally than the team does, and bring a copy of your "
                  "resume. At the end, ask a thoughtful question about the role, and send a short thank-you note the "
                  "same day."),
    "running": ("Write a short paragraph on how to start running.",
                "Start with run-walk intervals: one minute of easy running followed by two minutes of walking, repeated "
                "for twenty minutes. Do this three times a week, and add a little more running each week. Good shoes "
                "matter more than fancy gear, and a pace at which you can still talk is fast enough."),
    "password": ("Write a short paragraph on what makes a good password.",
                 "A good password is long, unique, and hard to guess. A phrase of four or five random words is easier to "
                 "remember than a short string of symbols, and it is much harder to crack. Never reuse a password across "
                 "sites, and let a password manager remember them for you."),
    "groceries": ("Write a short paragraph on how to save money on groceries.",
                  "Plan your meals for the week, and write a list before you go to the store. Buy staples like rice, "
                  "beans, and oats in bulk, and choose fruit and vegetables that are in season. Store brands are often "
                  "just as good as name brands, and shopping after a meal helps you skip impulse buys."),
    "flight": ("Write a short paragraph on how to prepare for a long flight.",
               "Choose an aisle seat if you like to stretch, and wear loose, comfortable clothes. Drink water often, skip "
               "the alcohol, and walk the aisle every couple of hours. Download movies and podcasts before you leave, "
               "and pack a small bag with earplugs, an eye mask, and a charger."),
    "sauce": ("Write a short paragraph on how to make a simple tomato sauce.",
              "Warm some olive oil in a pan, and gently cook a chopped onion and two cloves of garlic until soft. Add a "
              "can of crushed tomatoes, a pinch of salt, and a few basil leaves. Let it simmer for twenty minutes, "
              "stirring now and then, and taste it before serving."),
    # holdout (lacrado): só o supervisor vê estas métricas
    "exam": ("Write a short paragraph on how to study for an exam.",
             "Start a week ahead, and split the material into small daily sessions. Test yourself with practice "
             "questions instead of rereading your notes, and explain the hardest ideas out loud as if you were "
             "teaching them. Sleep well the night before, and eat a light breakfast on the day."),
    "language": ("Write a short paragraph on the benefits of learning a second language.",
                 "Learning a second language opens doors to new friends, new jobs, and new books. It trains memory and "
                 "attention, and it makes travel far richer. Even a few minutes of practice every day, with an app or a "
                 "conversation partner, adds up over a year."),
    "budget": ("Write a short paragraph on how to make a monthly budget.",
               "List everything you earn and everything you spend in a month, and sort the spending into needs, wants, "
               "and savings. Set a limit for each group, and track your purchases for a few weeks to see where the money "
               "really goes. Adjust the limits as you learn, and pay yourself first."),
}
SEARCH_TOPICS = ["plants", "sleep", "interview", "running", "password", "groceries", "flight", "sauce"]
HOLDOUT_TOPICS = ["exam", "language", "budget"]


def _no_comma(t):
    return t.replace(", ", " ").replace(",", "")


# instrução -> (texto da instrução, versão que cumpre(texto), versão que não cumpre(texto))
CONSTRAINTS = {
    "lowercase": ("Your entire response should be in English, and in all lowercase letters. No capital letters are allowed.",
                  str.lower, lambda t: t),
    "uppercase": ("Your entire response should be in English, and in all capital letters.", str.upper, lambda t: t),
    "no_comma": ("In your entire response, refrain from the use of any commas.", _no_comma, lambda t: t),
    "end_phrase": ("Finish your response with this exact phrase: Is there anything else I can help with? No other words "
                   "should follow this phrase.", lambda t: t + " Is there anything else I can help with?", lambda t: t),
    "json": ("Entire output should be wrapped in JSON format.", lambda t: json.dumps({"answer": t}, indent=2), lambda t: t),
    "quotation": ("Wrap your entire response with double quotation marks.", lambda t: f'"{t}"', lambda t: t),
    "title": ("Your answer must contain a title, wrapped in double angular brackets, such as <<poem of joy>>.",
              lambda t: "<<A Few Simple Tips>>\n\n" + t, lambda t: t),
    "postscript": ("At the end of your response, please explicitly add a postscript starting with P.S.",
                   lambda t: t + "\n\nP.S. Small habits make the biggest difference.", lambda t: t),
}


def pairs(tok, topics, with_instruction: bool) -> list[dict]:
    """-> [{topic, constraint, ok: (ids, mask), bad: (ids, mask)}]; mask marca os logp que preveem a resposta."""
    out = []
    for topic in topics:
        ask, text = PARAS[topic]
        for name, (instr, good, bad) in CONSTRAINTS.items():
            user = f"{ask} {instr}" if with_instruction else ask
            prompt = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            item = {"topic": topic, "constraint": name}
            for k, f in (("ok", good), ("bad", bad)):
                r_ids = tok(f(text) + "<|im_end|>", add_special_tokens=False)["input_ids"]
                ids = torch.tensor(p_ids + r_ids)
                m = torch.zeros(len(ids) - 1, dtype=torch.bool)  # logp[t] prevê o token t+1
                m[len(p_ids) - 1:] = True
                item[k] = (ids, m)
            out.append(item)
    return out
