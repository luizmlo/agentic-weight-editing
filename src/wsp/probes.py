"""Sondas de "ponto de ramificação" (branch point) para medir edições anti-loop sem gerar.

Protocolo do Astra (exp8, pass ~16, transfer probe): mesmo prefixo (problema + início do raciocínio),
duas continuações — produtiva (avança a solução) ou repetitiva (uma frase de re-checagem repetida 4x).
Mede-se o Δlogp(editado - base) médio nos tokens da continuação. Assinatura do Astra (α=4, direção
cheia): produtiva +0.005, repetitiva -0.104 nats/token.

ASTRA_CASES: os 6 casos publicados nos logs (runs/p50c/turns/t064). OWN_CASES: 6 casos novos no estilo
GSM8K escritos para este experimento (validação fora da amostra da métrica).

Também: onset natural do loop num texto (primeiro ponto em que >90% dos próximos 256 tokens são
8-gramas repetidos).
"""

from __future__ import annotations

import torch

from wsp import tf

PREFIX = "<|im_start|>user\n"
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>assistant\n<think>\n"

ASTRA_CASES = [
    ("Find the area of a right triangle with legs 6 and 8.",
     "The two perpendicular legs are 6 and 8. The hypotenuse is sqrt(36+64)=10. ",
     "The area uses the perpendicular legs, not the hypotenuse. Thus A=6*8/2=24. Checking with a rectangle of sides 6 and 8, its area is 48 and the diagonal divides it into two equal triangles.",
     "But is the hypotenuse 10? Yes, sqrt(36+64)=10. "),
    ("A bag has 3 red and 2 blue balls. Two balls are drawn without replacement. What is the probability both are red?",
     "There are 5 balls. The probability the first is red is 3/5. ",
     "After a red ball is removed, 2 red balls remain out of 4. The probability is (3/5)*(2/4)=3/10. Equivalently, there are 3 red pairs among 10 possible pairs, giving the same result.",
     "But is there replacement? No, the problem says without replacement. "),
    ("Find the sum of the first 20 positive odd integers.",
     "The first term is 1 and the twentieth term is 39. There are 20 terms. ",
     "Pair the first and last terms: 1+39=40. Each pair has sum 40, and there are 10 pairs. The total is 400. This also follows from the identity that the first n positive odd integers sum to n squared.",
     "But are there 20 terms? Yes, the problem asks for the first 20 odd integers. "),
    ("Find the remainder when 7^20 is divided by 5.",
     "Since 7 has remainder 2 modulo 5, we can instead compute 2^20 modulo 5. ",
     "Squaring gives 2^2=4, and squaring again gives 2^4=16, which is 1 modulo 5. Therefore 2^20=(2^4)^5 has remainder 1. The exponent 20 is divisible by the cycle length 4.",
     "But is the remainder of 7 equal to 2? Yes, 7=5+2. "),
    ("Solve x^2-5x+6=0 over the real numbers.",
     "The constant term is 6 and the coefficient of x is -5. ",
     "The numbers -2 and -3 have product 6 and sum -5. Thus the polynomial factors as (x-2)(x-3). A product is zero when either factor is zero, so x=2 or x=3. Substitution confirms both roots.",
     "But is the constant 6? Yes, the constant term is 6. "),
    ("For f(x)=x^3-4x, compute f'(2).",
     "The derivative of x^3 is 3x^2 and the derivative of -4x is -4. ",
     "Consequently f'(x)=3x^2-4. Substituting x=2 gives 3*4-4=8. We need the derivative at 2, not the function value f(2)=0; these are different quantities.",
     "But do we need f(2)? No, the question asks for the derivative at 2. "),
]

OWN_CASES = [
    ("Tom has 3 boxes with 12 pencils each. He gives away 8 pencils. How many pencils does he have left?",
     "Tom starts with 3 boxes of 12 pencils, so he has 3*12=36 pencils. ",
     "He gives away 8 of them, so 36-8=28 pencils remain. Checking: 28+8=36, which matches the starting total, so the answer is 28.",
     "Wait, does each box have 12 pencils? Yes, each box has 12 pencils. "),
    ("A shirt costs $25 and is on sale for 20% off. What is the sale price?",
     "The discount is 20% of 25, which is 0.2*25=5 dollars. ",
     "Subtracting the discount from the original price gives 25-5=20 dollars. As a check, 20 is 80% of 25, consistent with paying 80% of the price.",
     "Wait, is the discount 20%? Yes, the problem says 20% off. "),
    ("A train travels 60 miles per hour for 3 hours and then 40 miles per hour for 2 hours. How far does it travel in total?",
     "In the first part, the train covers 60*3=180 miles. ",
     "In the second part it covers 40*2=80 miles. The total distance is 180+80=260 miles. Both legs use distance equals speed times time, so the total is consistent.",
     "Wait, is the first speed 60? Yes, the train goes 60 miles per hour. "),
    ("Maria reads 15 pages a day. How many days does she need to finish a 180-page book?",
     "Maria reads 15 pages each day and the book has 180 pages. ",
     "Dividing gives 180/15=12 days. Checking, 12*15=180, so after 12 days she has read exactly the whole book.",
     "Wait, does she read 15 pages a day? Yes, she reads 15 pages a day. "),
    ("A farmer has 48 eggs and packs them into cartons of 6. He sells each carton for $3. How much money does he make?",
     "With 48 eggs and 6 eggs per carton, he fills 48/6=8 cartons. ",
     "Each carton sells for 3 dollars, so he earns 8*3=24 dollars. The number of cartons is a whole number, so no eggs are left over.",
     "Wait, are there 6 eggs per carton? Yes, each carton holds 6 eggs. "),
    ("Sam saves $7 every week. After how many weeks will he have saved $91?",
     "Sam adds 7 dollars to his savings each week, and he wants 91 dollars. ",
     "The number of weeks is 91/7=13. After 13 weeks he has 13*7=91 dollars, which is exactly the goal.",
     "Wait, does he save 7 dollars each week? Yes, he saves 7 dollars per week. "),
]


def branch_items(tok, cases, system: str | None = None, conds=("productive", "repetitive")):
    """-> lista de {case, cond, ids, mask}; mask marca as posições de logp que preveem a continuação.
    cond 'close' = fechar o raciocínio logo após o prefixo comum ("</think>")."""
    out = []
    sys_head = f"<|im_start|>system\n{system}<|im_end|>\n" if system else ""
    for i, (p, common, good, loop) in enumerate(cases):
        for cond, cont in (("productive", good), ("repetitive", (loop * 4).strip()), ("close", "</think>")):
            if cond not in conds:
                continue
            head = sys_head + PREFIX + p + SUFFIX + common
            text = head + cont
            enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
            ids = torch.tensor(enc["input_ids"])
            b = next(j for j, (_, e) in enumerate(enc["offset_mapping"]) if e > len(head))
            m = torch.zeros(len(ids) - 1, dtype=torch.bool)  # logp[t] prevê o token t+1
            m[b - 1:] = True
            out.append({"case": i, "cond": cond, "ids": ids, "mask": m})
    return out


def onset(ids: list[int], win=256, thr=0.9):
    """Início do loop duro: primeiro t com >thr dos próximos `win` tokens repetidos (8-grama); devolve t-7."""
    m = tf.dup_ngram_mask(ids).float()
    c = torch.cat([torch.zeros(1), m.cumsum(0)])
    for t in range(len(ids) - win):
        if m[t] and (c[t + win] - c[t]) / win > thr:
            return max(t - 7, 0)
    return None
