import os


def resolve_vision_tower(vision_tower):
    """Return a normalized tower path/name and its implementation kind.

    Explicit tower families must be resolved before the generic local-path
    fallback. In particular, OneVision checkpoints commonly point at a local
    SigLIP directory, which is a valid path but must not be treated as CLIP.
    """
    if vision_tower is None:
        raise ValueError("A vision tower path or model name is required")

    try:
        vision_tower = os.fsdecode(os.fspath(vision_tower))
    except TypeError as exc:
        raise TypeError("vision_tower must be a string or path-like object") from exc

    if not vision_tower:
        raise ValueError("A vision tower path or model name is required")

    normalized = vision_tower.lower()

    # Keep explicit model families ahead of the local-path fallback below.
    if "siglip" in normalized:
        return vision_tower, "siglip"
    if normalized.startswith("hf:"):
        return vision_tower, "hf"
    if normalized == "imagebind_huge":
        return vision_tower, "imagebind"
    if normalized.startswith("open_clip_hub"):
        return vision_tower, "open_clip"
    if "mlcd-vit-bigg-patch14" in normalized:
        return vision_tower, "mlcd"

    if (
        os.path.exists(vision_tower)
        or normalized.startswith("openai")
        or normalized.startswith("laion")
        or "sharegpt4v" in normalized
    ):
        return vision_tower, "clip"

    raise ValueError(f"Unknown vision tower: {vision_tower}")
