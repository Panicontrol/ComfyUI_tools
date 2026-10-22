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


def mask_to_h3_latent(mask, tokens, height, width, reduce="min", grow=0, feather=0):
    """A pixel MASK ([F, H, W] or [H, W]) as a [1, 1, T, h, w] latent mask.

    Each frame is grown or shrunk by ``grow`` px and blurred by ``feather`` px
    (as Mask Grow / Feather), then area-downscaled to the latent grid. Both
    happen on a 4 px grid -- a latent cell is 16 px, so finer detail would be
    averaged away anyway -- with the radii scaled to match, and frames go
    through a few at a time, so a long full-HD mask stays fast and small. The
    frames of every latent token are then reduced to one: min keeps a cell 0 if
    it is 0 in any of the token's frames, max keeps it 1 if it is 1 in any,
    mean averages them.
    """
    import torch.nn.functional as F

    from .mask_nodes import MaskGrowFeather

    m = mask if mask.ndim == 3 else mask[None]
    work = (min(m.shape[1], height * 4), min(m.shape[2], width * 4))
    px = m.shape[2] / work[1]  # mask pixels per work pixel
    grow = int(round(grow / px))
    feather = 0 if feather <= 0 else max(1, int(round(feather / px)))
    feather = min(feather, min(work) - 1)  # reflect padding needs radius < side
    small = []
    for start in range(0, m.shape[0], 8):
        part = m[start:start + 8].float()
        if part.shape[1:] != work:
            part = F.interpolate(part[:, None], size=work, mode="area")[:, 0]
        if grow or feather > 0:
            part = MaskGrowFeather().process(part, grow, feather, False)[0]
        small.append(F.interpolate(part[:, None], size=(height, width), mode="area")[:, 0])
    m = torch.cat(small)
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
    latent tokens (1, 4, 4, 4, 4 frames) and blended in latent space; grow and
    feather shape its edge in pixels first (32+ px hides the 16 px grid). set_noise_mask also attaches the mask, so
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
            # after the original widgets, so saved workflows keep their values in place
            "optional": {
                "grow": ("INT", {
                    "default": 0, "min": -256, "max": 256, "step": 1,
                    "tooltip": "Pixels: positive widens the white (latent_b) area, negative widens the "
                               "black (latent_a) area. Applied before feather.",
                }),
                "feather": ("INT", {
                    "default": 32, "min": 0, "max": 256, "step": 1,
                    "tooltip": "Gaussian blur of the mask edge in pixels, at full resolution. One latent "
                               "cell is 16 px, so 32+ hides the grid; 0 if the mask is already soft.",
                }),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "merge"
    CATEGORY = f"{CATEGORY}/latent"
    DESCRIPTION = "Merge two MiniMax H3 latents by a mask: black keeps latent_a, white takes latent_b."

    def merge(self, latent_a, latent_b, mask, temporal="min", audio="latent_a", set_noise_mask=False,
              grow=0, feather=32):
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
        weight = mask_to_h3_latent(mask, tokens, height, width, temporal, grow, feather).to(va.device)
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


H3_CLIP = 17  # an H3 clip is 17k + 5 frames; every 17 frames the VAE starts a new 1-frame token


def h3_av_parts(latent, what):
    samples = latent["samples"]
    if not is_nested(samples) or len(samples.tensors) < 2 or samples.tensors[0].ndim != 5:
        raise ValueError(f"{what} must be a MiniMax H3 video+audio latent")
    video, audio = samples.tensors[0], samples.tensors[1]
    if video.shape[0] != 1:
        raise ValueError("MiniMax H3 latents are batch 1")
    return samples, video, audio


def audio_index(frame):
    return round(frame * 5 / 3)  # H3 audio latent runs at 5/3 steps per video frame


def mask_bbox(mask, threshold=0.5):
    """(x, y, w, h) of the area above threshold in any frame, or None if empty."""
    m = mask if mask.ndim == 3 else mask[None]
    hit = (m > threshold).any(0)
    rows, cols = hit.any(1).nonzero(), hit.any(0).nonzero()
    if len(rows) == 0:
        return None
    y0, y1, x0, x1 = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
    return x0, y0, x1 - x0, y1 - y0


class H3LatentCrop:
    """Crop a MiniMax H3 latent in space and time, keeping it a valid H3 latent.

    H3 needs sides that are multiples of 32 px (2x2 latent cells) and clips of
    17k + 5 frames that start where the VAE starts a 17-frame group, so the box
    is widened outward to the 32 px grid and the frames snap to that grid. The
    audio part is cut to the same frames. Connect a mask to crop around its
    white area (plus padding) instead of x/y/width/height. The crop output
    carries where the piece came from, for H3 Latent Paste.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "x": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 32}),
                "y": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 32}),
                "width": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 32,
                                  "tooltip": "0 = to the right edge."}),
                "height": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 32,
                                   "tooltip": "0 = to the bottom edge."}),
                "start_frame": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 17,
                                        "tooltip": "Snapped down to a multiple of 17."}),
                "length": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 17,
                                   "tooltip": "Frames, snapped up to 17k + 5; 0 = to the end."}),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Crop around the white area of any frame instead of x/y/width/height."}),
                "padding": ("INT", {"default": 64, "min": 0, "max": 4096, "step": 8,
                                    "tooltip": "Pixels added around the mask's white area."}),
            },
        }

    RETURN_TYPES = ("LATENT", "H3_CROP", "MASK", "STRING")
    RETURN_NAMES = ("latent", "crop", "mask", "info")
    FUNCTION = "crop"
    CATEGORY = f"{CATEGORY}/latent"
    DESCRIPTION = "Crop a MiniMax H3 latent to a 32 px box and an H3 frame range; paste it back with H3 Latent Paste."

    def crop(self, latent, x, y, width, height, start_frame, length, mask=None, padding=64):
        samples, video, audio = h3_av_parts(latent, "latent")
        tokens, cells_h, cells_w = video.shape[2], video.shape[3], video.shape[4]
        full_w, full_h, frames = cells_w * 16, cells_h * 16, frames_for_tokens(tokens)

        if mask is not None:
            box = mask_bbox(mask)
            if box is None:
                raise ValueError("the mask is empty: nothing to crop around")
            sx, sy = full_w / mask.shape[-1], full_h / mask.shape[-2]  # mask may be another size
            bx, by, bw, bh = box
            x, y = int((bx - padding) * sx), int((by - padding) * sy)
            width, height = int((bw + 2 * padding) * sx + 0.999), int((bh + 2 * padding) * sy + 0.999)
        x, y = max(0, x), max(0, y)
        x1 = full_w if width <= 0 else min(full_w, x + width)
        y1 = full_h if height <= 0 else min(full_h, y + height)
        # widen outward to the 32 px grid H3 patches on
        c0, r0 = x // 32 * 2, y // 32 * 2
        c1, r1 = min(cells_w, -(-x1 // 32) * 2), min(cells_h, -(-y1 // 32) * 2)
        if c1 <= c0 or r1 <= r0:
            raise ValueError(f"the box is outside the {full_w}x{full_h} latent")

        f0 = min(start_frame // H3_CLIP * H3_CLIP, max(0, frames - 5) // H3_CLIP * H3_CLIP)
        k0 = f0 // H3_CLIP * 5
        available = tokens - k0
        if length > 0:
            n = (max(length, 5) - 5 + H3_CLIP - 1) // H3_CLIP * 5 + 2
            n = min(n, available)
        else:
            n = available
        k1 = k0 + n
        f1 = frames_for_tokens(k1)

        cut_video = video[:, :, k0:k1, r0:r1, c0:c1].contiguous()
        cut_audio = audio[..., audio_index(f0):min(audio.shape[-1], audio_index(f1))].contiguous()
        out = dict(latent)
        out["samples"] = type(samples)([cut_video, cut_audio, *samples.tensors[2:]])
        old = latent.get("noise_mask")
        out.pop("noise_mask", None)
        if is_nested(old) and old.tensors[0].ndim == 5 and old.tensors[0].shape[2:] == video.shape[2:]:
            out["noise_mask"] = type(old)([old.tensors[0][:, :, k0:k1, r0:r1, c0:c1],
                                           torch.zeros_like(cut_audio, dtype=torch.float32)])

        crop = {"x": c0 * 16, "y": r0 * 16, "width": (c1 - c0) * 16, "height": (r1 - r0) * 16,
                "start_frame": f0, "frames": f1 - f0, "token0": k0, "tokens": n,
                "full_width": full_w, "full_height": full_h, "full_frames": frames}
        if mask is not None:
            m = mask if mask.ndim == 3 else mask[None]
            if tuple(m.shape[1:]) != (full_h, full_w):
                from .utils import resize_mask

                m = resize_mask(m, full_w, full_h)
            m = m[min(f0, m.shape[0] - 1):f1] if m.shape[0] > 1 else m
            cut_mask = m[:, r0 * 16:r1 * 16, c0 * 16:c1 * 16]
        else:
            cut_mask = torch.ones((1, (r1 - r0) * 16, (c1 - c0) * 16))
        info = (f"{crop['width']}x{crop['height']} at {crop['x']},{crop['y']} of {full_w}x{full_h}; "
                f"frames {f0}-{f1 - 1} ({f1 - f0}) of {frames}")
        logging.info("[H3 Latent Crop] %s", info)
        return (out, crop, cut_mask, info)


class H3LatentPaste:
    """Put a piece cut by H3 Latent Crop back into the full MiniMax H3 latent.

    The piece must still have the crop's size and frame count (sample it, but
    do not resize it). Without a mask the whole box is pasted; with one, only
    its white area -- full-frame or crop-sized, as H3 Latent Crop outputs it.
    feather fades the edge inside the pasted area, so nothing outside the box
    is touched and no 16 px step shows.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "The full latent to paste into."}),
                "piece": ("LATENT", {"tooltip": "The cropped (and sampled) latent."}),
                "crop": ("H3_CROP",),
                "feather": ("INT", {"default": 32, "min": 0, "max": 256, "step": 1,
                                    "tooltip": "Soft edge in pixels, drawn inside the pasted area."}),
                "audio": (["latent", "piece"], {
                    "default": "latent",
                    "tooltip": "piece writes the piece's audio over its frames.",
                }),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Paste only the white area: full-frame or crop-sized."}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "paste"
    CATEGORY = f"{CATEGORY}/latent"
    DESCRIPTION = "Paste a piece from H3 Latent Crop back into the full MiniMax H3 latent, with a soft edge."

    def paste(self, latent, piece, crop, feather=32, audio="latent", mask=None):
        samples, video, base_audio = h3_av_parts(latent, "latent")
        _, part, part_audio = h3_av_parts(piece, "piece")
        k0, n = crop["token0"], crop["tokens"]
        r0, c0 = crop["y"] // 16, crop["x"] // 16
        rows, cols = crop["height"] // 16, crop["width"] // 16
        if (video.shape[3] * 16, video.shape[4] * 16) != (crop["full_height"], crop["full_width"]) \
                or video.shape[2] < k0 + n:
            raise ValueError("latent is not the one this crop was cut from")
        if tuple(part.shape[2:]) != (n, rows, cols):
            raise ValueError(f"the piece is {tuple(part.shape[2:])} (tokens, rows, cols) but the crop is "
                             f"{(n, rows, cols)}; paste it at the size it was cropped")

        # the paste mask on a 4 px grid (what mask_to_h3_latent works on), shrunk by
        # feather so the fade ends at the box edge; frame borders stay put
        grid = torch.zeros((1, rows * 4, cols * 4))
        if mask is not None:
            from .utils import resize_mask

            m = mask if mask.ndim == 3 else mask[None]
            if tuple(m.shape[1:]) == (crop["full_height"], crop["full_width"]):
                f0 = crop["start_frame"]
                m = m[min(f0, m.shape[0] - 1):f0 + crop["frames"]] if m.shape[0] > 1 else m
                m = m[:, crop["y"]:crop["y"] + crop["height"], crop["x"]:crop["x"] + crop["width"]]
            grid = resize_mask(m.float(), cols * 4, rows * 4)
        else:
            grid = grid + 1.0
        pad = (c0 * 4, (video.shape[4] - c0 - cols) * 4, r0 * 4, (video.shape[3] - r0 - rows) * 4)
        import torch.nn.functional as F

        full = F.pad(grid, pad)
        limit = min(crop["width"], crop["height"]) // 4  # leave the middle of a small box at full weight
        if feather > limit:
            logging.info("[H3 Latent Paste] feather %d px -> %d px: the box is %dx%d",
                         feather, limit, crop["width"], crop["height"])
        f = int(round(min(feather, limit) / 4))
        weight = mask_to_h3_latent(full, n, video.shape[3], video.shape[4], "min", grow=-f, feather=f)
        weight = weight[:, :, :, r0:r0 + rows, c0:c0 + cols].to(video.device)

        merged = video.clone()
        region = merged[:, :, k0:k0 + n, r0:r0 + rows, c0:c0 + cols].float()
        merged[:, :, k0:k0 + n, r0:r0 + rows, c0:c0 + cols] = (
            region + (part.to(video.device).float() - region) * weight).to(video.dtype)

        new_audio = base_audio
        if audio == "piece":
            new_audio = base_audio.clone()
            a0 = audio_index(crop["start_frame"])
            a1 = min(new_audio.shape[-1], a0 + part_audio.shape[-1])
            new_audio[..., a0:a1] = part_audio[..., :a1 - a0].to(new_audio.device, new_audio.dtype)
        out = dict(latent)
        out.pop("noise_mask", None)
        out["samples"] = type(samples)([merged, new_audio, *samples.tensors[2:]])
        return (out,)


def otsu_threshold(values, bins=256):
    """The threshold that best splits values into two groups (Otsu)."""
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-8:
        return hi
    hist = torch.histc(values.float(), bins=bins, min=lo, max=hi)
    centers = torch.linspace(lo, hi, bins)
    weight_lo = hist.cumsum(0)
    weight_hi = hist.sum() - weight_lo
    mean_lo = (hist * centers).cumsum(0) / weight_lo.clamp(min=1e-8)
    mean_hi = ((hist * centers).sum() - (hist * centers).cumsum(0)) / weight_hi.clamp(min=1e-8)
    between = weight_lo * weight_hi * (mean_lo - mean_hi) ** 2
    return float(centers[int(between.argmax())])


class H3LatentDiffMask:
    """Find where a prompt edit changed a MiniMax H3 shot, by comparing two latents.

    latent_a is the source, latent_b the edited shot, both encoded with the H3
    VAE at the same size and frame count. Regeneration shifts colour and grain
    over the whole frame, so the per-channel median shift is removed first and
    each channel is scaled by the source's spread; what is left is scored per
    16 px latent cell. Cells above the threshold (Otsu's automatic split by
    default) become the mask, cleaned of stray cells, smoothed over time, then
    grown and feathered. White = changed, ready for H3 Latent Mask Merge to
    keep the source everywhere else.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent_a": ("LATENT", {"tooltip": "The source shot."}),
                "latent_b": ("LATENT", {"tooltip": "The edited shot, same size and frame count."}),
                "threshold": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 5.0, "step": 0.01,
                    "tooltip": "Change score a cell needs to count as edited; 0 = automatic (Otsu). "
                               "The score is about 0 for untouched cells and 1.4 for unrelated content.",
                }),
                "clean": ("INT", {
                    "default": 1, "min": 0, "max": 8, "step": 1,
                    "tooltip": "Drop specks and fill pinholes smaller than this many latent cells (16 px).",
                }),
                "temporal_smooth": ("INT", {
                    "default": 1, "min": 0, "max": 8, "step": 1,
                    "tooltip": "Average the score over this many latent tokens either side (a token is 4 frames).",
                }),
                "grow": ("INT", {"default": 16, "min": -256, "max": 256, "step": 1,
                                 "tooltip": "Pixels to widen (+) or shrink (-) the changed area."}),
                "feather": ("INT", {"default": 32, "min": 0, "max": 256, "step": 1,
                                    "tooltip": "Soft edge in pixels."}),
            },
        }

    RETURN_TYPES = ("MASK", "MASK", "STRING")
    RETURN_NAMES = ("mask", "heatmap", "info")
    FUNCTION = "diff"
    CATEGORY = f"{CATEGORY}/latent"
    DESCRIPTION = "Mask of what a prompt edit changed, from the source and edited MiniMax H3 latents."

    def diff(self, latent_a, latent_b, threshold=0.0, clean=1, temporal_smooth=1, grow=16, feather=32):
        import torch.nn.functional as F

        from .mask_nodes import MaskGrowFeather

        _, va, _ = h3_av_parts(latent_a, "latent_a")
        _, vb, _ = h3_av_parts(latent_b, "latent_b")
        if tuple(va.shape) != tuple(vb.shape):
            raise ValueError(
                f"the video parts differ: latent_a {tuple(va.shape)}, latent_b {tuple(vb.shape)}; "
                "encode both at the same size and frame count"
            )
        a, b = va[0].float(), vb.to(va.device)[0].float()  # [C, T, h, w]
        d = b - a
        d = d - d.flatten(1).median(dim=1).values.view(-1, 1, 1, 1)  # overall colour / exposure drift
        spread = a.flatten(1).std(dim=1).clamp(min=1e-4).view(-1, 1, 1, 1)
        score = (d / spread).pow(2).mean(0).sqrt()  # [T, h, w]

        tokens, rows, cols = score.shape
        if temporal_smooth > 0 and tokens > 1:
            k = 2 * temporal_smooth + 1
            score = F.avg_pool1d(score.permute(1, 2, 0).reshape(-1, 1, tokens), k, 1, temporal_smooth,
                                 count_include_pad=False).reshape(rows, cols, tokens).permute(2, 0, 1)
        cut = threshold if threshold > 0 else otsu_threshold(score.flatten())
        cells = (score > cut).float()[:, None]  # [T, 1, h, w]
        if clean > 0:
            k = 2 * clean + 1
            cells = -F.max_pool2d(-F.max_pool2d(cells, k, 1, clean), k, 1, clean)  # close: fill holes
            cells = F.max_pool2d(-F.max_pool2d(-cells, k, 1, clean), k, 1, clean)   # open: drop specks
        share = float(cells.mean())

        # tokens -> frames (1, 4, 4, 4, 4 per group), then grow / feather on a 4 px grid
        index = torch.cat([torch.full((FRAME_PER_TOKEN[t % len(FRAME_PER_TOKEN)],), t) for t in range(tokens)])
        work = F.interpolate(cells, size=(rows * 4, cols * 4), mode="nearest")[:, 0]
        if grow or feather > 0:
            work = MaskGrowFeather().process(work, int(round(grow / 4)),
                                             min(int(round(feather / 4)), min(work.shape[1:]) - 1), False)[0]
        mask = F.interpolate(work[:, None], size=(rows * 16, cols * 16), mode="bilinear",
                             align_corners=False)[:, 0].clamp(0.0, 1.0)[index]
        heat = (score / max(cut * 2.0, 1e-6)).clamp(0.0, 1.0)[:, None]
        heat = F.interpolate(heat, size=(rows * 4, cols * 4), mode="bilinear", align_corners=False)[:, 0][index]

        info = (f"changed {share:.1%} of the shot, threshold {cut:.3f}"
                f"{' (auto)' if threshold <= 0 else ''}; score median {float(score.median()):.3f}, "
                f"max {float(score.max()):.3f}; {len(index)} frames")
        logging.info("[H3 Latent Diff Mask] %s", info)
        return (mask, heat, info)


NODE_CLASS_MAPPINGS = {
    "ToolsSaveLatentTo": SaveLatentTo,
    "ToolsLoadLatentFrom": LoadLatentFrom,
    "ToolsEncodeStillH3": EncodeStillH3,
    "ToolsH3LatentMaskMerge": H3LatentMaskMerge,
    "ToolsH3LatentCrop": H3LatentCrop,
    "ToolsH3LatentPaste": H3LatentPaste,
    "ToolsH3LatentDiffMask": H3LatentDiffMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ToolsSaveLatentTo": "Save Latent To (tools)",
    "ToolsLoadLatentFrom": "Load Latent From (tools)",
    "ToolsEncodeStillH3": "H3 Encode Still (tools)",
    "ToolsH3LatentMaskMerge": "H3 Latent Mask Merge (tools)",
    "ToolsH3LatentCrop": "H3 Latent Crop (tools)",
    "ToolsH3LatentPaste": "H3 Latent Paste (tools)",
    "ToolsH3LatentDiffMask": "H3 Latent Diff Mask (tools)",
}
