from .vision_tower_utils import resolve_vision_tower
from .imagebind import ImageBindWrapper
from .open_clip_encoder import OpenCLIPVisionTower
from .hf_vision import HFVisionTower
from .siglip_encoder import SigLipVisionTower
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2
from .mlcd_encoder import MLCDVisionTower, MLCDVisionTowerS2
# from .eva_clip.eva_clip_encoder import EvaClipVisionTower
# from .dev_eva_clip.eva_vit import EvaViTWrapper


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, "mm_vision_tower", getattr(vision_tower_cfg, "vision_tower", None))
    vision_tower, tower_type = resolve_vision_tower(vision_tower)
    use_s2 = getattr(vision_tower_cfg, "s2", False)

    if tower_type == "siglip":
        return SigLipVisionTower(vision_tower, vision_tower_cfg=vision_tower_cfg, **kwargs)
    elif tower_type == "hf":
        return HFVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif tower_type == "imagebind":
        return ImageBindWrapper(vision_tower, args=vision_tower_cfg, **kwargs)
    elif tower_type == "open_clip":
        return OpenCLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif tower_type == "mlcd":
        if use_s2:
            return MLCDVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            return MLCDVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif tower_type == "clip":
        if use_s2:
            return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    # elif "internal-eva" in vision_tower.lower() or "eva02" in vision_tower.lower():
    #     return EvaClipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    # elif vision_tower in ["EVA-CLIP-8B", "EVA-CLIP-8B-plus"]:
    #     return EvaViTWrapper(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f"Unknown vision tower type {tower_type!r}: {vision_tower}")
