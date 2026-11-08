"""M→A, hipótese de dose: a direção achada pelo r1 ajuda quando aplicada em dose pequena
(branch_rep ~ -0,1 a -0,2, como o Astra) em vez de proibir a repetição literal (-2 nats)?

Varre α para duas direções do r1 (L12 201939_974 e L14 185027_886, mesmas colunas), escolhe para
cada uma o α com branch_rep mais próximo de TARGET e salva runs/dose/done/<id>.pt; escreve
runs/dose/pick.txt com os ids escolhidos (lido por queue_dose.sh).
"""
import torch

from wsp import m2a

TARGET = -0.15
SRC = {"L12": ("r1", "201939_974", [2, 3, 4, 5, 6]), "L14": ("r1", "185027_886", [2, 3, 4, 5])}
picks = []
for name, (run, cid, alphas) in SRC.items():
    base = torch.load(m2a.RUNS / run / "done" / f"{cid}.pt")["edits"]
    res = []
    for a in alphas:
        edits = [{**e, "alpha": float(a)} for e in base]
        r = m2a.submit("dose", edits, note=f"dose {name} alpha={a} (from {run}/{cid})")
        print(name, a, {k: round(r[k], 4) for k in ("branch_rep", "branch_prod", "close", "neutral")}, flush=True)
        res.append((abs(r["branch_rep"] - TARGET), a, r["id"]))
    best = min(res)
    print("pick", name, "alpha", best[1], best[2], flush=True)
    picks.append(f"{name}:{best[1]}:{best[2]}")
(m2a.RUNS / "dose" / "pick.txt").write_text("\n".join(picks) + "\n")
