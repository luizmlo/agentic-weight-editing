"""E4 (baseline de fine-tuning): peças comuns ao teste de viabilidade e ao treino.

- load_lora(): base fp16 na GPU + LoRA (peft, adaptador em fp32) + gradient checkpointing
- chunked_loss(): CE só nos tokens da resposta, calculada em blocos com checkpoint, sem materializar
  os logits inteiros (vocabulário de 248k: 4k tokens x 248k x 4 B ≈ 4 GB)
"""

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer

from wsp.paths import BASE_REPO, BASE_REV, snapshot

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
           "in_proj_qkv", "in_proj_z", "out_proj"]


def load_lora(r=16, alpha=32, dropout=0.0):
    path = snapshot(BASE_REPO, BASE_REV)
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16, device_map="cuda")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=TARGETS, bias="none",
                     task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg, autocast_adapter_dtype=True)
    for n, p in model.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.train()  # o gradient checkpointing do transformers só age com self.training=True (dropout é 0)
    return model, tok


def deltanet_path(model) -> str:
    """Qual implementação do Gated DeltaNet está em uso (kernel fla/Triton ou torch puro)."""
    try:
        import fla  # noqa: F401  # o transformers usa os kernels Triton do fla quando o pacote está instalado
        return "fla (Triton)"
    except ImportError:
        return "torch (fallback)"


def chunked_loss(model, ids: torch.Tensor, n_prompt: int, chunk: int = 256):
    """ids: [T] na GPU. Perda média por token da resposta (posições >= n_prompt)."""
    inner = model.get_base_model()
    h = inner.model(input_ids=ids[None]).last_hidden_state[0]  # [T, d]
    head = inner.lm_head
    tgt = ids[1:]
    h = h[:-1]
    start = max(n_prompt - 1, 0)
    total = torch.zeros((), device=ids.device, dtype=torch.float32)
    n = 0

    def piece(hs, ts):
        return F.cross_entropy(head(hs).float(), ts, reduction="sum")

    for s in range(start, h.shape[0], chunk):
        e = min(s + chunk, h.shape[0])
        total = total + checkpoint(piece, h[s:e], tgt[s:e], use_reentrant=False)
        n += e - s
    return total / max(n, 1), n
