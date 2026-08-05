# Comparisons

Every baseline uses the same Qwen2.5-VL backbone and the same LoRA rank as
SmartRes, fine-tuned with the same schedule (3 epochs, lr 1e-4, cosine).

| Method | Ratios | Where |
|---|---|---|
| Down-scaling | 10%, 32%, 50% | `comparisons/downscale.py` |
| FastV | 50%, 70% | `comparisons/fastv/` |
| ToMe | 50%, 70% | [upstream](https://github.com/facebookresearch/ToMe) |
| VisionZip | 50%, 70% | [upstream](https://github.com/JIA-Lab-research/VisionZip) |
| SparseVLM | 50% | [upstream](https://github.com/Gumpest/SparseVLMs) |
| Dyn-LLaVA | 50%, 70% | [upstream](https://github.com/Osilly/dynamic_llava) |
| VScan | 50% | [upstream](https://github.com/Tencent/SelfEvolvingAgent/tree/main/VScan) |

## Down-scaling

Resize every image to a fixed token budget; boxes stay in the resized frame.

```bash
python comparisons/downscale.py --input data/egointention_context_test.json \
    --output data/egointention_context_test_32pct.json \
    --ratio 0.32 --write-images /path/to/frames_32pct
python tools/run_eval.py configs/qwen2_5vl_3b_lora_predict_egoint_lite.yaml \
    --eval-dataset egointention_context_test_32pct --output-dir outputs/downscale_32
```

## FastV

Prunes image tokens at decoder layer `k` using layer `k-1` attention. Needs
`attn_implementation="eager"` — it ranks by attention weights, which fused kernels do not
return.

```python
from comparisons.fastv import install_fastv
state = install_fastv(model, k=2, keep_ratio=0.5)   # paper: k=2, 50% and 70%
```

## Settings for the other baselines

Run these from their own repositories against the same backbone and schedule. These are
the values we used.

**ToMe**

```yaml
tome_r: 0.50            # discard ratio; 0.30 for the 70%-token row
tome_prop_attn: true    # proportional attention
tome_trace_source: false
```

**SparseVLM**

```yaml
sparse_retention_ratio: 0.5
sparse_version: "2_0"
```

**VScan**

```yaml
vscan_k: 2              # prune at LLM layer 2, aligned with FastV
vscan_r: 0.50           # keep 50% of image tokens
vscan_local_ratio: 0.5  # half the kept budget windowed-local, half global
vscan_window_size: 4
```

**VisionZip** and **Dyn-LLaVA** — 50% and 70% token retention, defaults otherwise.

All decode with `per_device_eval_batch_size: 1`, `max_new_tokens: 5120`,
`do_sample: false`.
