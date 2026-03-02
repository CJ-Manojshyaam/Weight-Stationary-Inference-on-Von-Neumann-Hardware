# Weight-Stationary-Inference-on-Von-Neumann-Hardware
Can we make existing CPUs behave more like neuromorphic chips — just in software?
## What This Is
A demonstration of **weight-stationary dataflow** on standard CPUs.
By loading model weights once into L3 cache and reusing them across all samples,
we reduce DRAM fetch cycles — the dominant energy cost in neural network inference.

## Key Insight
| Model Size | Fits in L3? | WS FP32 Saving | WS INT8 Saving |
|---|---|---|---|
| < 16 MB (Small CNN) | ✅ Yes | ~11× energy | ~48× energy |
| > 16 MB (ResNet-50+) | ❌ No | 1× (no gain) | 4× (always) |

**Quantization (INT8) always helps** — 4× fewer bytes transferred from DRAM regardless of model size.

## Run
```bash
pip install -r requirements.txt
jupyter notebook weight_stationary_inference.ipynb
