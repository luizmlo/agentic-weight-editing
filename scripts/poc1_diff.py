"""PoC-1: engenharia reversa da edição publicada (exp. 8 do paper).

Compara tensor a tensor o Qwen3.5-0.8B base com o editado pelo Astra e, para cada tensor
alterado, verifica se ΔW é rank-1 e recupera a direção u (vetor singular à esquerda).
Segundo o relatório do Astra: W = W0 - 6 * outer(u, mask64 * (u @ W0)).

Saída: data/poc1_diff.json e data/edit_dirs.pt (u por camada, máscara, alpha estimado).
"""

import json

import torch
from safetensors import safe_open

from wsp.paths import BASE_REPO, BASE_REV, DATA, EDITED_REPO, safetensors_file

base_f = safetensors_file(BASE_REPO, BASE_REV)
edit_f = safetensors_file(EDITED_REPO)

report, dirs = {"changed": []}, {}
with safe_open(base_f, "pt") as b, safe_open(edit_f, "pt") as e:
    names = sorted(b.keys())
    assert names == sorted(e.keys()), "conjuntos de tensores diferentes"
    for n in names:
        w0, w1 = b.get_tensor(n), e.get_tensor(n)
        if torch.equal(w0, w1):
            continue
        w0, w1 = w0.float(), w1.float()
        d = w1 - w0
        cols = (d.abs().sum(0) > 0).nonzero().squeeze(1)
        s = torch.linalg.svdvals(d)
        U, S, Vh = torch.linalg.svd(d, full_matrices=False)
        u = U[:, 0]
        # alpha: d[:, cols] = -alpha * u (u @ W0[:, cols])  ->  alpha = -(u @ d[:,cols]) / (u @ W0[:,cols])
        proj0 = u @ w0[:, cols]
        alpha = (-(u @ d[:, cols]) / proj0).median().item()
        recon = -alpha * torch.outer(u, proj0)
        rel_err = ((d[:, cols] - recon).norm() / d.norm()).item()
        info = {
            "tensor": n,
            "shape": list(w0.shape),
            "changed_cols": len(cols),
            "sv_top5": [round(x, 4) for x in s[:5].tolist()],
            "rank1_energy": round((s[0] ** 2 / (s**2).sum()).item(), 6),
            "alpha_est": round(alpha, 3),
            "recon_rel_err_bf16": round(rel_err, 4),
            "rel_change": round((d.norm() / w0.norm()).item(), 4),
        }
        report["changed"].append(info)
        layer = int(n.split("layers.")[1].split(".")[0])
        dirs[layer] = {"u": u, "cols": cols, "alpha": alpha, "tensor": n}
        print(json.dumps(info))

ls = sorted(dirs)
if len(ls) == 2:
    c = torch.dot(dirs[ls[0]]["u"], dirs[ls[1]]["u"]).abs().item()
    ov = len(set(dirs[ls[0]]["cols"].tolist()) & set(dirs[ls[1]]["cols"].tolist()))
    report["cos_u_between_layers"] = round(c, 4)
    report["shared_neuron_idx"] = ov
    print(f"|cos(u{ls[0]}, u{ls[1]})| = {c:.4f}; neurônios com mesmo índice: {ov}")

DATA.mkdir(parents=True, exist_ok=True)
torch.save(dirs, DATA / "edit_dirs.pt")
(DATA / "poc1_diff.json").write_text(json.dumps(report, indent=1))
