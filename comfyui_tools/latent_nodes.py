"""Save and load LATENTs anywhere on disk, including MiniMax H3 AV latents.

ComfyUI's own Save Latent writes to output/latents while Load Latent only
lists the input folder, neither takes a path, and both store just a plain
``samples`` tensor. A MiniMax H3 AV latent carries ``comfy.nested_tensor``
NestedTensors (video + audio) in ``samples`` and ``noise_mask``, which that
format cannot hold. These nodes store every entry of the LATENT dict in one
safetensors file (``.latent``) and rebuild it exactly on load.
"""

import json
import logging
import os

import torch

from .utils import CATEGORY

try:  # ComfyUI dependencies
    import safetensors.torch
except Exception:  # pragma: no cover
    safetensors = None

try:
    from comfy.nested_tensor import NestedTensor
except Exception:  # pragma: no cover - outside ComfyUI
    NestedTensor = None

FORMAT = "comfyui_tools.latent.v1"


def is_nested(value):
    return getattr(value, "is_nested", False) is True and hasattr(value, "tensors")


def default_directory():
    try:
        import folder_paths

        return os.path.join(folder_paths.get_output_directory(), "latents")
    except Exception:
        return os.path.abspath("latents")


def resolve_file(path):
    """A saved latent: absolute, ~, or relative to ComfyUI's output or input folder."""
    path = os.path.expanduser(str(path).strip().strip('"'))
    if os.path.isabs(path):
        return path
    try:
        import folder_paths

        for base in (folder_paths.get_output_directory(), folder_paths.get_input_directory()):
            candidate = os.path.join(base, path)
            if os.path.isfile(candidate):
                return candidate
    except Exception:
        pass
    return os.path.abspath(path)


def pack(latent):
    """LATENT dict -> (tensors, metadata) for safetensors."""
    tensors = {}
    layout = {}
    for key, value in latent.items():
        if is_nested(value):
            for index, tensor in enumerate(value.tensors):
                tensors[f"{key}.{index}"] = tensor.detach().cpu().contiguous()
            layout[key] = {"kind": "nested", "count": len(value.tensors)}
        elif isinstance(value, torch.Tensor):
            tensors[key] = value.detach().cpu().contiguous()
            layout[key] = {"kind": "tensor"}
        else:
            try:
                layout[key] = {"kind": "json", "value": json.loads(json.dumps(value))}
            except (TypeError, ValueError):
                logging.warning("[comfyui_tools] latent entry %r is not saveable and was left out", key)
    return tensors, {"format": FORMAT, "layout": json.dumps(layout)}


def unpack(tensors, metadata):
    """(tensors, metadata) -> LATENT dict. Also reads ComfyUI's own .latent files."""
    if metadata.get("format") != FORMAT:
        if "latent_tensor" not in tensors:
            raise ValueError("not a latent file saved by ComfyUI or by these nodes")
        scale = 1.0 if "latent_format_version_0" in tensors else 1.0 / 0.18215
        return {"samples": tensors["latent_tensor"].float() * scale}

    latent = {}
    for key, entry in json.loads(metadata["layout"]).items():
        if entry["kind"] == "tensor":
            latent[key] = tensors[key]
        elif entry["kind"] == "nested":
            if NestedTensor is None:
                raise RuntimeError("this latent holds a NestedTensor; load it inside ComfyUI")
            latent[key] = NestedTensor([tensors[f"{key}.{i}"] for i in range(entry["count"])])
        else:
            latent[key] = entry["value"]
    return latent


def next_free_path(directory, prefix):
    counter = 1
    while True:
        path = os.path.join(directory, f"{prefix}_{counter:05d}.latent")
        if not os.path.exists(path):
            return path
        counter += 1


class SaveLatentTo:
    """Save a LATENT (plain or MiniMax H3 AV) to a folder of your choice."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "directory": ("STRING", {
                    "default": "",
                    "tooltip": "Folder to save into; created if missing. Empty = ComfyUI/output/latents.",
                }),
                "filename_prefix": ("STRING", {"default": "latent"}),
            },
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("samples", "path")
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = f"{CATEGORY}/latent"
    DESCRIPTION = "Save a latent -- including MiniMax H3 video+audio latents and their masks -- to any folder."

    def save(self, samples, directory, filename_prefix):
        folder = os.path.expanduser(str(directory).strip().strip('"')) or default_directory()
        os.makedirs(folder, exist_ok=True)
        path = next_free_path(folder, filename_prefix.strip() or "latent")

        tensors, metadata = pack(samples)
        # safetensors' save_file creates the file 0600 whatever the umask, so
        # other users and tools on Linux cannot read it; writing the bytes
        # with open() gives the usual permissions, like every other output
        with open(path, "wb") as handle:
            handle.write(safetensors.torch.save(tensors, metadata=metadata))
        return {"ui": {"text": [path]}, "result": (samples, path)}


class LoadLatentFrom:
    """Load a latent saved by Save Latent To (or by ComfyUI's Save Latent) from a path."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {
                    "default": "",
                    "tooltip": "The .latent file: absolute, or relative to ComfyUI's output or input folder.",
                }),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "load"
    CATEGORY = f"{CATEGORY}/latent"
    DESCRIPTION = "Load a .latent file from any path, rebuilding MiniMax H3 video+audio latents."

    @classmethod
    def IS_CHANGED(cls, path):
        file = resolve_file(path)
        if not os.path.isfile(file):
            return f"missing:{file}"
        return f"{file}:{os.path.getmtime(file)}:{os.path.getsize(file)}"

    def load(self, path):
        file = resolve_file(path)
        if not os.path.isfile(file):
            raise ValueError(f"latent file not found: {file}")
        with safetensors.safe_open(file, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            tensors = {key: handle.get_tensor(key) for key in handle.keys()}
        return (unpack(tensors, metadata),)


class EncodeStillH3:
    """Put an encoded image into a MiniMax H3 still latent, for latent img2img tests.

    H3 samples a NestedTensor pair (video, audio), batch 1. Connect the still
    latent you would sample from (e.g. Fizgig H3 Still Latent): this node
    encodes the image with the H3 video VAE, resized to that latent's canvas,
    and swaps it in as the video part. The audio part and every other key stay
    exactly as the still latent made them. Sample with denoise below 1.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE", {"tooltip": "The MiniMax H3 video VAE."}),
                "images": ("IMAGE", {"tooltip": "The still to start from; resized to the latent's size."}),
                "latent": ("LATENT", {"tooltip": "The H3 still latent to fill, e.g. Fizgig H3 Still Latent."}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "encode"
    CATEGORY = f"{CATEGORY}/latent"
    DESCRIPTION = "Swap an encoded image into a MiniMax H3 still latent (keeps its audio part) for img2img tests."

    def encode(self, vae, images, latent):
        from .utils import resize_image

        samples = latent["samples"]
        if not is_nested(samples) or len(samples.tensors) < 2:
            raise ValueError(
                "connect a MiniMax H3 still latent (video + audio), such as Fizgig H3 Still Latent; "
                "H3 cannot sample a video-only latent"
            )
        video, *rest = samples.tensors
        if video.shape[0] != 1 or images.shape[0] != 1:
            raise ValueError("MiniMax H3 samples one still at a time: use batch size 1 and a single image")
        if video.ndim != 5 or video.shape[2] != 1:
            raise ValueError(f"the latent's video part is {tuple(video.shape)}, not a single frame [1, 24, 1, H, W]")

        height, width = video.shape[3] * 16, video.shape[4] * 16
        image = resize_image(images[:1, :, :, :3], width, height, "lanczos")
        encoded = vae.encode(image)
        if tuple(encoded.shape) != tuple(video.shape):
            raise ValueError(
                f"the VAE returned {tuple(encoded.shape)} but the latent needs {tuple(video.shape)}; "
                "use the MiniMax H3 video VAE"
            )

        out = dict(latent)
        out["samples"] = type(samples)([encoded.to(device=video.device, dtype=video.dtype), *rest])
        return (out,)

try:  # frames per latent token in MiniMax H3: 1, 4, 4, 4, 4, then repeats
    from comfy.ldm.minimax.model import FRAME_PER_TOKEN
except Exception:  # pragma: no cover - outside ComfyUI or older builds
    FRAME_PER_TOKEN = (1, 4, 4, 4, 4)


def frames_for_tokens(n):
    return sum(FRAME_PER_TOKEN[i % len(FRAME_PER_TOKEN)] for i in range(n))


def mask_to_h3_latent(mask, tokens, height, width, reduce="min"):
    """A pixel MASK ([F, H, W] or [H, W]) as a [1, 1, T, h, w] latent mask.

    Each frame is area-downscaled to the latent grid, then the frames of every
    latent token are reduced to one: min keeps a cell 0 if it is 0 in any of
    the token's frames, max keeps it 1 if it is 1 in any, mean averages them.
    """
    import torch.nn.functional as F

    m = mask.float()
    if m.ndim == 2:
        m = m[None]
    m = F.interpolate(m[:, None], size=(height, width), mode="area")[:, 0]
    frames = m.shape[0]
    out = torch.empty((tokens, height, width), dtype=torch.float32)
    for k in range(tokens):
        if frames == 1:
            out[k] = m[0]
            continue
        f0 = min(frames_for_tokens(k), frames - 1)
        f1 = max(f0 + 1, min(frames_for_tokens(k + 1), frames))
        span = m[f0:f1]
        out[k] = span.amin(0) if reduce == "min" else span.amax(0) if reduce == "max" else span.mean(0)
    return out.clamp(0.0, 1.0)[None, None]


class H3LatentMaskMerge:
    """Merge two MiniMax H3 latents by a mask: black keeps latent_a, white takes latent_b.

    Typical use: latent_a is the plate encoded with the H3 VAE, latent_b the
    generated or upscaled shot; the face and anything that must match the
    plate stay black. Both latents need the same video size. The mask may be a
    single frame for the whole clip or one per frame; it is mapped onto H3's
    latent tokens (1, 4, 4, 4, 4 frames) and blended in latent space, so give
    it a soft edge of 32 px or more. set_noise_mask also attaches the mask, so
    a sampler after this node redraws only the white area.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent_a": ("LATENT", {"tooltip": "Kept where the mask is black (e.g. the plate)."}),
                "latent_b": ("LATENT", {"tooltip": "Taken where the mask is white (e.g. the generation)."}),
                "mask": ("MASK", {"tooltip": "White = latent_b, black = latent_a. One frame or one per frame."}),
                "temporal": (["min", "mean", "max"], {
                    "default": "min",
                    "tooltip": "How a latent token (4 frames) combines its mask frames. min: black in any "
                               "frame keeps latent_a for the whole token; max: white in any frame takes latent_b.",
                }),
                "audio": (["latent_a", "latent_b"], {
                    "default": "latent_a",
                    "tooltip": "Which latent's audio part the result carries.",
                }),
                "set_noise_mask": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Attach the mask as noise_mask, so the next sampler redraws only the white area.",
                }),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "merge"
    CATEGORY = f"{CATEGORY}/latent"
    DESCRIPTION = "Merge two MiniMax H3 latents by a mask: black keeps latent_a, white takes latent_b."

    def merge(self, latent_a, latent_b, mask, temporal="min", audio="latent_a", set_noise_mask=False):
        sa, sb = latent_a["samples"], latent_b["samples"]
        nested = is_nested(sa)
        if nested != is_nested(sb):
            raise ValueError("one latent is an H3 video+audio latent and the other is not")
        va, vb = (sa.tensors[0], sb.tensors[0]) if nested else (sa, sb)
        if va.ndim != 5:
            raise ValueError(f"expected a video latent [B, C, T, H, W], got {tuple(va.shape)}")
        if tuple(va.shape) != tuple(vb.shape):
            raise ValueError(
                f"the video parts differ: latent_a {tuple(va.shape)}, latent_b {tuple(vb.shape)}; "
                "encode both at the same size and frame count"
            )

        tokens, height, width = va.shape[2], va.shape[3], va.shape[4]
        frames = frames_for_tokens(tokens)
        if mask.ndim == 3 and mask.shape[0] not in (1, frames):
            logging.warning("[H3 Latent Mask Merge] the mask has %d frames, the latent covers %d; "
                            "tokens past the mask's end reuse its last frame", mask.shape[0], frames)
        weight = mask_to_h3_latent(mask, tokens, height, width, temporal).to(va.device)
        logging.info("[H3 Latent Mask Merge] mask %s -> latent %s, latent_a kept on %.1f%%",
                     tuple(mask.shape), tuple(weight.shape[2:]), (weight < 0.5).float().mean().item() * 100)

        a32 = va.float()
        video = (a32 + (vb.to(va.device).float() - a32) * weight).to(va.dtype)

        out = dict(latent_b if audio == "latent_b" else latent_a)
        out.pop("noise_mask", None)
        if nested:
            rest = (sb if audio == "latent_b" else sa).tensors[1:]
            out["samples"] = type(sa)([video, *rest])
            if set_noise_mask:
                out["noise_mask"] = type(sa)([weight, *[torch.zeros_like(r, dtype=torch.float32) for r in rest]])
        else:
            out["samples"] = video
            if set_noise_mask:
                out["noise_mask"] = weight
        return (out,)


NODE_CLASS_MAPPINGS = {
    "ToolsSaveLatentTo": SaveLatentTo,
    "ToolsLoadLatentFrom": LoadLatentFrom,
    "ToolsEncodeStillH3": EncodeStillH3,
    "ToolsH3LatentMaskMerge": H3LatentMaskMerge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ToolsSaveLatentTo": "Save Latent To (tools)",
    "ToolsLoadLatentFrom": "Load Latent From (tools)",
    "ToolsEncodeStillH3": "H3 Encode Still (tools)",
    "ToolsH3LatentMaskMerge": "H3 Latent Mask Merge (tools)",
}
