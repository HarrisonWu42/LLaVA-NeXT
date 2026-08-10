"""Offline import smoke test for the current LLaVA-OneVision stack.

This script does not download a config, tokenizer, checkpoint, or model weight.
Run it after installing this repository into the existing ms-swift environment.
"""

import os
import sys
from pathlib import Path

from packaging.version import Version


# llava.model historically prints and suppresses architecture import errors.
# Make every such error fatal for this smoke test before importing llava.
os.environ["LLAVA_STRICT_MODEL_IMPORTS"] = "1"


def _print_version(module_name, module):
    version = getattr(module, "__version__", "unknown")
    print(f"{module_name}: {version}")


def main():
    print(f"Python: {sys.version.split()[0]}")

    import accelerate
    import peft
    import tokenizers
    import torch
    import torchvision
    import transformers
    import trl

    _print_version("torch", torch)
    _print_version("torchvision", torchvision)
    _print_version("transformers", transformers)
    _print_version("tokenizers", tokenizers)
    _print_version("accelerate", accelerate)
    _print_version("peft", peft)

    import llava
    from llava.conversation import conv_templates
    from llava.model.language_model.llava_qwen import LlavaQwenConfig, LlavaQwenForCausalLM
    from llava.model.multimodal_resampler.qformer import (
        apply_chunking_to_forward,
        find_pruneable_heads_and_indices,
        prune_linear_layer,
    )

    repository_root = Path(__file__).resolve().parents[1]
    imported_llava = Path(llava.__file__).resolve()
    if repository_root not in imported_llava.parents:
        raise RuntimeError(f"Imported llava from {imported_llava}, expected source under {repository_root}")
    imported_trl = Path(trl.__file__).resolve()
    if repository_root in imported_trl.parents:
        raise RuntimeError(
            f"Imported the vendored legacy TRL from {imported_trl}. Remove the LLaVA-NeXT root from PYTHONPATH "
            "and use the TRL version installed for ms-swift."
        )
    trl_version = Version(trl.__version__)
    if not Version("0.15") <= trl_version < Version("1.0"):
        raise RuntimeError(f"ms-swift 4.4 requires TRL >=0.15,<1.0, but imported {trl.__version__} from {imported_trl}")

    assert LlavaQwenConfig.model_type == "llava_qwen"
    assert LlavaQwenForCausalLM.config_class is LlavaQwenConfig
    assert "qwen_1_5" in conv_templates
    assert all(
        callable(helper)
        for helper in (
            apply_chunking_to_forward,
            find_pruneable_heads_and_indices,
            prune_linear_layer,
        )
    )

    try:
        import swift
    except ModuleNotFoundError as exc:
        if exc.name == "swift":
            raise RuntimeError(
                "ms-swift is not installed. Install the companion checkout that registers "
                "the llava_onevision_qwen model type."
            ) from exc
        raise

    from swift.model import MLLMModelType, MODEL_MAPPING

    try:
        model_type = MLLMModelType.llava_onevision_qwen
    except AttributeError as exc:
        raise RuntimeError("This ms-swift checkout does not define llava_onevision_qwen") from exc
    if model_type not in MODEL_MAPPING:
        raise RuntimeError("This ms-swift checkout does not register llava_onevision_qwen")

    print(f"llava source: {imported_llava}")
    _print_version("ms-swift", swift)
    print(f"ms-swift source: {Path(swift.__file__).resolve()}")
    _print_version("trl", trl)
    print(f"trl source: {imported_trl}")
    print("LlavaQwenConfig/LlavaQwenForCausalLM: OK")
    print("qwen_1_5 conversation template: OK")
    print("Q-Former Transformers helpers: OK")
    print("ms-swift llava_onevision_qwen registration: OK")
    print("Environment smoke test: PASS")


if __name__ == "__main__":
    main()
