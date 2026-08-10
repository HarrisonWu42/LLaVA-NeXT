# Install in an Existing ms-swift Environment

This installation path is for using LLaVA-OneVision source code inside an
existing, shared ms-swift environment. It is designed to preserve the current
core stack instead of replacing it with the historical versions used by the
legacy LLaVA-NeXT training pipeline.

The validated profile is:

| Component | Validated version |
| --- | --- |
| Python | 3.10 |
| PyTorch / torchvision | 2.8 / 0.23 |
| Transformers / Tokenizers | 4.57 / 0.22 |
| Accelerate | 1.14 |
| PEFT | 0.19 |
| TRL | 0.19 |
| ms-swift | 4.4 |

## Shared-environment install

LLaVA-NeXT supplies the original checkpoint architecture, but it does not by
itself make the checkpoint trainable through ms-swift. First ensure that the
active environment uses the companion ms-swift 4.4 checkout containing the
`llava_onevision_qwen` loader. For a local checkout, preserve the environment's
dependencies while registering that source tree:

```bash
python -m pip install -e <ms-swift-path> --no-deps
```

Then run these commands from the LLaVA-NeXT repository root:

```bash
# Add only the small packages missing from the validated shared environment.
# --no-deps prevents these packages from changing Torch, Transformers, or PEFT.
python -m pip install --no-deps "shortuuid>=1,<2" "einops-exts>=0.0.4,<0.1" "ftfy>=6,<7"

# Register this source tree without asking pip to resolve the legacy stack.
python -m pip install -e . --no-deps

# This imports the real Qwen2/OneVision classes but downloads no model files.
python scripts/check_ms_swift_environment.py
```

The smoke script makes architecture import failures fatal, prints the active
Python/Torch/Transformers/PEFT versions, explicitly imports
`LlavaQwenConfig` and `LlavaQwenForCausalLM`, verifies the `qwen_1_5`
conversation template, checks the Q-Former compatibility imports, and verifies
that ms-swift registered `llava_onevision_qwen`. It does not instantiate a
model or load weights.

`python -m pip check` can be useful as a separate diagnostic, but it is not the
acceptance gate for this environment. In the validated server environment,
decord 0.6.0 may emit `is not supported on this platform` because of its wheel
metadata even when its runtime import works. Use the smoke script above as the
compatibility gate and evaluate any `pip check` findings individually.

The `ms-swift` optional extra records the complete validated version ranges.
To let pip fill all compatible runtime dependencies, inspect its plan first:

```bash
python -m pip install --dry-run -e ".[ms-swift]"
python -m pip install -e ".[ms-swift]"
```

Do not proceed if the dry-run proposes replacing the environment's core
packages. The explicit `--no-deps` workflow above is safer for a provisioned
shared environment.

## Train an original LLaVA-OneVision checkpoint

Pass both the companion model type and this source checkout explicitly:

```bash
swift sft \
  --model <original-LLaVA-OneVision-checkpoint> \
  --model_type llava_onevision_qwen \
  --local_repo_path <LLaVA-NeXT-root> \
  <other-ms-swift-arguments>
```

The exact 0.5B/7B checkpoint basename can be auto-matched, but specifying
`--model_type llava_onevision_qwen` is more robust for renamed local
directories. `--local_repo_path` is required so the loader uses this reviewed
source revision instead of silently obtaining another copy.

This original-checkpoint route currently requires exactly one image per
training sample. Text-only, multi-image, and video samples are rejected with an
explicit error. Use the Transformers-native `llava_onevision_hf` route and an
`*-ov-hf` checkpoint when broader media support is required.

Training in the current environment is owned by ms-swift. Do not use the
legacy `llava/train` Trainer entrypoints with Transformers 4.57: they still
depend on Trainer internals that moved after the upstream frozen environment.

## PEFT and quantization

Keep PEFT installed at the version selected by ms-swift. PEFT is required for
LoRA/QLoRA even though full-parameter training does not use adapter layers.
The validated `peft==0.19.1` satisfies the `peft>=0.11,<0.20` range used by
ms-swift 4.4; do not replace it with the legacy LLaVA pin (`peft==0.4.0`).

`bitsandbytes` is intentionally not part of the `ms-swift` extra. Install a
CUDA-compatible bitsandbytes build only when using 4-bit/8-bit or QLoRA paths.
Likewise, keep the environment's existing flash-attn/DeepSpeed builds instead
of reinstalling them through this source package.

## Legacy reproduction

The `train` extra and `requirements.txt` are retained solely to reproduce the
historical upstream training environment. They pin old versions including
Torch 2.1.2, torchvision 0.16.2, PEFT 0.4.0, and an old Transformers commit.
The vendored legacy `trl` source remains in the repository but is deliberately
excluded from package discovery, so current installs do not expose it as an
installed package. Avoid launching Python with the LLaVA-NeXT repository root
on `PYTHONPATH`, because an explicit source-root path could still shadow the
newer TRL required by ms-swift. Legacy DPO reproduction must manage its
compatible TRL source in the isolated legacy environment.
Install them only in a separate environment:

```bash
conda create -n llava-legacy python=3.10 -y
conda activate llava-legacy
python -m pip install -e ".[train]"
```

Never run that command in the shared ms-swift environment.
