"""H3 Masked Split Upscale: tiled MiniMax H3 latent refine that redraws only a mask.

Based on MMH3 Split Upscale from Comfyui_Minimax_h3_latent_Upscaler by LBH-123-AI
(https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler, commit 40316cf),
used under the MIT License below. Kept here, under its own node ids, so a mask
input and tile skipping do not depend on that pack's updates. Changes:

  * mask input: white = redraw, black = keep the input latent. The mask is
    area-downscaled to the latent grid, each token takes the minimum over its
    frames, it multiplies every tile's fade mask (the sampler re-injects the
    frozen area at every step), and the frozen area is restored after the
    colour match and seam polish passes too
  * skip_frozen_tiles: tiles (and polish seams) with no white in the mask are
    not sampled at all -- the result is the same, only faster
  * node ids Tools*, category tools/latent; the H3_*_PARAM_V10 types are kept,
    so that pack's split param nodes plug in as well

Original notice:

    MIT License

    Copyright (c) 2026 LBH-123-AI

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:/n/nThe above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

Original module docstring (frame splitting, anchors, colour match, seam polish):
MMH3 Split Upscale - 完整最终版 (含 seam_denoise 缝邻域降噪上限)
预防: 冻结预填重叠带 + 三重时间锚 + cross-fade
校正: 空间/时间两级颜色匹配 + 首块源参考 + 每 chunk 全局钉源
抗分叉: seam_denoise 缝邻域降噪上限 (高降噪+快运动防切割残影)
修复: 每瓦片独立 Guider + 裁剪 keyframes; 修复版 probe 门控 polish
简化: overlap/fade 百分比参数。
"""

import math
import torch
import torch.nn.functional as TF
import comfy.model_management
import comfy.sample
import comfy.samplers
import comfy.utils
import comfy.nested_tensor
import latent_preview
from comfy_api.latest import io, ui

from .tile_prompts import cell_at, draw_tile_map, fill_template, parse_tile_prompts, tile_name, with_tile_text

try:
    from comfy.ldm.minimax.model import FRAME_PER_TOKEN, FRAME_RESCALE
except Exception:
    FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
    FRAME_RESCALE = 5.0 / 3.0

H3_TEMPORAL_PARAM = io.Custom("H3_TEMPORAL_PARAM_V10")
H3_SPATIAL_PARAM = io.Custom("H3_SPATIAL_PARAM_V10")
H3_TILE_PROMPTS = io.Custom("H3_TILE_PROMPTS")

VAE_DOWNSAMPLE = 16
ALIGN = 2
MAX_IDENTITY_ANCHORS = 6
POLISH_HALO = 16
DC_MATCH_CLAMP = 0.05
SEAM_CORR_GATE, SEAM_DC_GATE = 0.85, 0.6
COLOR_CLAMP = 0.05
FROZEN_EPS = 1.0 / 255.0  # mask values at or below this count as black

# ---------------------------------------------------------------------------
# 帧/token 映射
# ---------------------------------------------------------------------------
def frames_for_tokens(n): return sum(FRAME_PER_TOKEN[i % 5] for i in range(n))

def tokens_for_frames(f):
    n, acc = 0, 0
    while acc < f:
        acc += FRAME_PER_TOKEN[n % 5]; n += 1
    return n

def audio_range(f0, f1): return round(f0 * FRAME_RESCALE), round(f1 * FRAME_RESCALE)

def clip_tokens(n): return (n - 5) // 17 * 5 + 2 if n >= 5 else 1

def snap_clip_frames(v): return 5 + 17 * max(1, round((v - 5) / 17)) if v >= 5 else max(1, int(v))

def snap_overlap_frames(v): return 0 if v <= 0 else 5 + 17 * max(0, round((v - 5) / 17))

def build_user_mask(mask, T, H, W, device):
    """[MASK] MASK (F,Hp,Wp) or (Hp,Wp) -> latent mask (1,1,T,H,W).
    white(1) = regenerate, black(0) = keep input latent. Per latent token the
    minimum over its frames is taken, so a region frozen in any frame stays frozen."""
    m = mask.float()
    if m.dim() == 2:
        m = m[None]
    Fm = m.shape[0]
    m = TF.interpolate(m[:, None], size=(H, W), mode="area")[:, 0]
    out = torch.empty((T, H, W), dtype=torch.float32)
    for k in range(T):
        if Fm == 1:
            out[k] = m[0]
            continue
        f0 = min(frames_for_tokens(k), Fm - 1)
        f1 = max(f0 + 1, min(frames_for_tokens(k + 1), Fm))
        out[k] = m[f0:f1].amin(0)
    return out.clamp(0.0, 1.0)[None, None].to(device)

def token_start_at_or_before(f):
    k = 0
    while frames_for_tokens(k + 1) <= f:
        k += 1
    return k

def steps_for_frames(n):
    k, covered = 0, 0
    while covered < n:
        covered += FRAME_PER_TOKEN[k % 5]; k += 1
    return k if covered == n else None

def compute_h3_segments_adaptive(tv, chunk_frames, overlap_frames):
    tc = clip_tokens(chunk_frames)
    to = clip_tokens(overlap_frames) if overlap_frames > 0 else 0
    if to >= tc:
        to = max(0, tc - 1)
    if tc >= tv:
        return [(0, 0, tv, frames_for_tokens(tv))], frames_for_tokens(tv)
    hop = max(1, tc - to)
    bounds, prev_k0, i = [], -1, 0
    while True:
        k0 = i * hop
        if k0 + tc >= tv:
            k1 = tv
            k0 = max(k0, tv - tc)
            if prev_k0 >= 0 and k0 <= prev_k0:
                k0 = prev_k0 + 1
            if k0 >= tv:
                break
            bounds.append((k0, frames_for_tokens(k0), k1, frames_for_tokens(k1)))
            break
        bounds.append((k0, frames_for_tokens(k0), k0 + tc, frames_for_tokens(k0 + tc)))
        prev_k0 = k0
        i += 1
    return bounds, frames_for_tokens(tv)

def is_h3_av_latent(samples):
    return (samples is not None and samples.is_nested and len(samples.tensors) == 2
            and samples.tensors[0].ndim == 5 and samples.tensors[0].shape[1] == 24
            and samples.tensors[1].ndim == 4 and samples.tensors[1].shape[1] == 32)

def px_to_lat(px):
    return max(ALIGN, (round(px / VAE_DOWNSAMPLE) // ALIGN) * ALIGN)

def snap_align(v):
    return max(0, int(round(v / ALIGN)) * ALIGN)

# ---------------------------------------------------------------------------
# keyframe 链路
# ---------------------------------------------------------------------------
def trim_keyframe(kf, f0, f1):
    idx = kf["resolved_frame_index"]
    latent, audio_latent = kf.get("latent"), kf.get("audio_latent")
    if latent is None and audio_latent is None:
        return None if (idx < f0 or idx >= f1) else {"resolved_frame_index": idx - f0}
    out = {}
    if latent is not None:
        t_start = t_end = None
        pos = idx
        for k in range(latent.shape[2]):
            span = FRAME_PER_TOKEN[k % 5]
            if f0 <= pos and pos + span <= f1:
                if t_start is None: t_start = k
                t_end = k + 1
            pos += span
        if t_start is None: return None
        out["latent"] = latent[:, :, t_start:t_end].contiguous()
        out["resolved_frame_index"] = idx + frames_for_tokens(t_start) - f0
    if audio_latent is not None:
        rt = audio_latent.shape[-1]
        a_start = max(0, math.ceil((f0 - idx) * FRAME_RESCALE))
        a_end = min(rt, math.floor((f1 - idx) / FRAME_RESCALE))
        if a_end > a_start:
            out["audio_latent"] = audio_latent[..., a_start:a_end].contiguous()
            if "resolved_frame_index" not in out:
                out["resolved_frame_index"] = max(0, idx - f0)
    return out if ("latent" in out or "audio_latent" in out) else None

def reanchor_conditioning(cond, f0, f1, spatial):
    out = []
    for tensor, d in cond:
        nd = dict(d)
        kfs = nd.get("minimax_keyframes")
        if kfs:
            trimmed = [kf for kf in (trim_keyframe(kf, f0, f1) for kf in kfs) if kf is not None]
            if trimmed:
                if spatial is not None:
                    for kf in trimmed:
                        lt = kf.get("latent")
                        if lt is not None and (lt.shape[3] != spatial[0] or lt.shape[4] != spatial[1]):
                            B, C, T, H, W = lt.shape
                            kf["latent"] = torch.nn.functional.interpolate(
                                lt.view(B * T, C, H, W), size=spatial, mode="bilinear",
                                align_corners=False).view(B, C, T, spatial[0], spatial[1])
                nd["minimax_keyframes"] = trimmed
            else:
                nd.pop("minimax_keyframes", None)
        out.append([tensor, nd])
    return out

def prepend_keyframes(cond, kfs):
    if not kfs:
        return cond
    out = []
    for tensor, d in cond:
        nd = dict(d)
        nd["minimax_keyframes"] = kfs + (nd.get("minimax_keyframes") or [])
        out.append([tensor, nd])
    return out

def motion_keyframes(prev_video, prev_tokens, f0, n_frames):
    n = next((g for g in (56, 39, 22, 5) if g <= n_frames), 0)
    steps = steps_for_frames(n) if n else None
    if not steps or steps > prev_tokens or (prev_tokens - steps) % 5 != 0:
        return []
    start = prev_tokens - steps
    return [{"resolved_frame_index": frames_for_tokens(start + k) - f0,
             "latent": prev_video[:, :, start + k:start + k + 1].contiguous()}
            for k in range(steps)]

def identity_keyframes(source, f0, f1, spacing):
    kfs, p = [], f0 + spacing
    while p < f1:
        k = token_start_at_or_before(p)
        kfs.append({"resolved_frame_index": frames_for_tokens(k) - f0,
                    "latent": source[:, :, k:k + 1].contiguous()})
        p += spacing
    if len(kfs) > MAX_IDENTITY_ANCHORS:
        kfs = [kfs[i * len(kfs) // MAX_IDENTITY_ANCHORS] for i in range(MAX_IDENTITY_ANCHORS)]
    return kfs

def anchor_conditioning(cond, prev_video, f0, strength):
    t = tokens_for_frames(f0)
    if t >= prev_video.shape[2]:
        raise ValueError("previous result does not reach the current segment start")
    anchor_kf = {"resolved_frame_index": 0, "latent": prev_video[:, :, t:t + 1].contiguous()}
    aug = max(0.0, min(1.0, float(strength)))
    out = []
    for tensor, d in cond:
        nd = dict(d)
        kfs = nd.get("minimax_keyframes")
        if kfs:
            kept = [kf for kf in kfs if kf.get("resolved_frame_index") != 0 or "latent" not in kf]
            nd["minimax_keyframes"] = [anchor_kf] + kept
        else:
            nd["minimax_keyframes"] = [anchor_kf]
        nd["minimax_visual_cond_noise_aug"] = aug
        out.append([tensor, nd])
    return out

def crop_keyframes_to_tile(cond, src_h, src_w, r0, c0, tr, tc):
    out = []
    for tensor, d in cond:
        nd = dict(d)
        kfs = nd.get("minimax_keyframes")
        if kfs:
            cropped = []
            for kf in kfs:
                nkf = dict(kf)
                lt = kf.get("latent")
                if lt is not None:
                    if lt.shape[3] == src_h and lt.shape[4] == src_w:
                        nkf["latent"] = lt[:, :, :, r0:r0 + tr, c0:c0 + tc].contiguous()
                    else:
                        lt_r = torch.nn.functional.interpolate(
                            lt.to(torch.float32), size=(src_h, src_w),
                            mode="bilinear", align_corners=False)
                        nkf["latent"] = lt_r[:, :, :, r0:r0 + tr, c0:c0 + tc].contiguous()
                cropped.append(nkf)
            nd["minimax_keyframes"] = cropped
        out.append([tensor, nd])
    return out

# ---------------------------------------------------------------------------
# 空间网格
# ---------------------------------------------------------------------------
def _grid_1d(size, tile, ol, min_tile):
    if size <= tile:
        return [0], [size], [0]
    sh = tile - ol
    n = math.ceil((size - ol) / sh)
    if (n - 1) * sh + tile < size:
        n += 1
    rows = [i * sh for i in range(n)]
    trows = [min(tile, size - r) for r in rows]
    if min_tile > 0 and n >= 2:
        edge = size - rows[-1]
        if edge < min_tile:
            new_last = size - min_tile
            if rows[-2] < new_last < rows[-2] + trows[-2]:
                rows[-1] = new_last
                trows[-1] = size - new_last
    ovl = [0] * n
    for i in range(1, n):
        ovl[i] = max(0, rows[i - 1] + trows[i - 1] - rows[i])
    return rows, trows, ovl

def compute_spatial_grid(h, w, th, tw, ol_h, ol_w, min_th=0, min_tw=0):
    rows, trows, row_ovl = _grid_1d(h, th, ol_h, min_th)
    cols, tcols, col_ovl = _grid_1d(w, tw, ol_w, min_tw)
    return rows, cols, trows, tcols, row_ovl, col_ovl

def spatial_grid(h, w, sp):
    """The upscale's tile grid of an h x w latent: (rows, cols, trows, tcols, row_ovl, col_ovl), in cells."""
    if sp is None:
        return [0], [0], [h], [w], [0], [0]
    return compute_spatial_grid(h, w, sp["th"], sp["tw"], sp["ol_h"], sp["ol_w"], sp["mt"], sp["mt"])

def _square_window(b0, b1, dim, size):
    """Start of a size-long window (even cells) around [b0, b1), kept inside [0, dim)."""
    size = min(size, dim - dim % ALIGN if dim >= ALIGN else dim)
    start = snap_align((b0 + b1) / 2 - size / 2)
    start = min(start, b0 - b0 % ALIGN)          # cover the start of the area
    start = max(start, b1 - size + (b1 - size) % ALIGN)  # and its end, rounding up onto the grid
    start = max(0, min(start, dim - size))
    return start - start % ALIGN, size


def mask_tile_plan(um_chunk, tile_h, tile_w, ctx, lp, H, W):
    """Tiles around the white areas of a chunk's mask instead of a frame grid.

    um_chunk: [T, H, W] latent mask of the chunk. Every white blob (union over the
    chunk's tokens) plus ctx cells of context gets one tile_h x tile_w window centred
    on it; a blob too big for one window gets its own small grid with lp's overlaps.
    Windows that would overlap are merged first. Returns (tiles, windows): tiles as
    (group, ri, cj, r0, c0, tr, tc, ovh, ovw) in latent cells, windows as (r0, c0, h, w).
    """
    union = (um_chunk > FROZEN_EPS).any(0)
    if not bool(union.any()):
        return [], []
    try:
        from scipy import ndimage
        labels, count = ndimage.label(union.cpu().numpy())
        boxes = [(int(sl[0].start), int(sl[0].stop), int(sl[1].start), int(sl[1].stop))
                 for sl in ndimage.find_objects(labels) if sl is not None]
    except Exception:  # no scipy: one window over all the white
        rr, cc = union.any(1).nonzero(), union.any(0).nonzero()
        boxes = [(int(rr[0]), int(rr[-1]) + 1, int(cc[0]), int(cc[-1]) + 1)]
    areas = [(max(0, r0 - ctx), min(H, r1 + ctx), max(0, c0 - ctx), min(W, c1 + ctx)) for r0, r1, c0, c1 in boxes]

    def window(a):
        # the area widened outward to the 32 px grid, so an aligned window can always hold it
        a = (a[0] - a[0] % ALIGN, a[1] + a[1] % ALIGN, a[2] - a[2] % ALIGN, a[3] + a[3] % ALIGN)
        r0, h = _square_window(a[0], a[1], H, max(tile_h, a[1] - a[0]))
        c0, w = _square_window(a[2], a[3], W, max(tile_w, a[3] - a[2]))
        return r0, c0, h, w

    merged = True
    while merged:  # windows that would overlap become one area
        merged = False
        wins = [window(a) for a in areas]
        for x in range(len(areas)):
            for y in range(x + 1, len(areas)):
                (ar, ac, ah, aw), (br, bc, bh, bw) = wins[x], wins[y]
                if ar < br + bh and br < ar + ah and ac < bc + bw and bc < ac + aw:
                    a, b = areas[x], areas[y]
                    areas[x] = (min(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), max(a[3], b[3]))
                    del areas[y]
                    merged = True
                    break
            if merged:
                break
    wins = sorted(window(a) for a in areas)
    tiles = []
    for g, (wr, wc, wh, ww) in enumerate(wins):
        th, tw = min(tile_h, wh), min(tile_w, ww)
        rows, trows, rovl = _grid_1d(wh, th, min(lp["ol_h"], max(0, th - ALIGN)), lp["mt"])
        cols, tcols, covl = _grid_1d(ww, tw, min(lp["ol_w"], max(0, tw - ALIGN)), lp["mt"])
        for ri in range(len(rows)):
            for cj in range(len(cols)):
                tiles.append((g, ri, cj, wr + rows[ri], wc + cols[cj], trows[ri], tcols[cj], rovl[ri], covl[cj]))
    return tiles, wins


def spatial_fade_mask(tile_h, tile_w, ovh, ovw, done_top, done_left,
                      fade_h=0, fade_w=0, seam_cap=1.0):
    """1=自由重采样, 0=冻结。
    重叠带 = 冻结段(缝侧) + 渐变段(0->seam_cap);
    seam_cap<1 时重叠带后再加一段等宽渐变 (cap->1), 让缝邻域以中等降噪
    "续写"冻结条内容, 高降噪下防止运动物体被切断。seam_cap=1.0 = 经典行为。"""
    mask = torch.ones(tile_h, tile_w, dtype=torch.float32)

    def profile(n, ov, fade):
        p = torch.ones(n, dtype=torch.float32)
        f = min(fade, ov)
        frozen = ov - f
        p[:frozen] = 0.0
        if f > 0:
            p[frozen:ov] = torch.linspace(0.0, seam_cap, f)
        ramp = min(ov, n - ov)
        if ramp > 0 and seam_cap < 1.0:
            start = seam_cap if f > 0 else 0.0
            p[ov:ov + ramp] = torch.linspace(start, 1.0, ramp)
        return p

    if done_left and ovw > 0:
        mask = torch.minimum(mask, profile(tile_w, ovw, fade_w)[None, :])
    if done_top and ovh > 0:
        mask = torch.minimum(mask, profile(tile_h, ovh, fade_h)[:, None])
    return mask

# ---------------------------------------------------------------------------
# 校正 + 门控
# ---------------------------------------------------------------------------
def dc_correct(new, refs, clamp=COLOR_CLAMP, min_samples=256):
    pairs = [p for p in refs if p is not None and p[0] is not None
             and p[0].numel() >= min_samples]
    if not pairs:
        return new, None
    pa = torch.cat([a.float().permute(0, 2, 3, 4, 1).reshape(-1, a.shape[1]) for a, _ in pairs])
    pb = torch.cat([b.float().permute(0, 2, 3, 4, 1).reshape(-1, b.shape[1]) for _, b in pairs])
    dc = (pa - pb).median(dim=0).values.clamp(-clamp, clamp)
    return new - dc.view(1, -1, 1, 1, 1).to(new.device, new.dtype), dc

def grade_pin(chunk, ref, clamp=COLOR_CLAMP):
    a = chunk.float().permute(0, 2, 3, 4, 1).reshape(-1, chunk.shape[1])
    b = ref.float().permute(0, 2, 3, 4, 1).reshape(-1, ref.shape[1])
    dc = (a - b).median(dim=0).values.clamp(-clamp, clamp)
    return chunk - dc.view(1, -1, 1, 1, 1).to(chunk.device, chunk.dtype), dc

def seam_metrics(sub, region):
    if sub.numel() < 4096:
        return None, None
    sp = sub.float().permute(0, 2, 3, 4, 1).reshape(-1, sub.shape[1])
    rp = region.float().permute(0, 2, 3, 4, 1).reshape(-1, region.shape[1])
    dc = (sp - rp).median(dim=0).values.clamp(-DC_MATCH_CLAMP, DC_MATCH_CLAMP)
    a = sp - sp.mean(dim=0)
    b = rp - rp.mean(dim=0)
    corr = ((a * b).mean(dim=0) / (a.std(dim=0) * b.std(dim=0) + 1e-6)).median()
    return dc, float(corr)

def _should_polish(mode, sub, region):
    if mode == "all":
        return True
    dc, corr = seam_metrics(sub, region)
    if dc is None:
        return False
    return ((corr is not None and corr < SEAM_CORR_GATE)
            or dc.abs().max().item() > SEAM_DC_GATE * DC_MATCH_CLAMP)

# ---------------------------------------------------------------------------
# 时间缝合
# ---------------------------------------------------------------------------
def _crossfade(a, b, dim):
    n = a.shape[dim]
    w = torch.linspace(0.0, 1.0, n, device=a.device, dtype=a.dtype)
    shape = [1] * a.ndim
    shape[dim] = n
    return a + (b - a) * w.view(shape)

def temporal_append(acc_v, acc_a, chunk_v, chunk_a, index, k0, f0, color_match=True):
    if acc_v is None:
        return chunk_v, chunk_a
    gi, agi = k0, round(f0 * FRAME_RESCALE)
    total_v = max(acc_v.shape[2], gi + chunk_v.shape[2])
    total_a = max(acc_a.shape[-1], agi + chunk_a.shape[-1])
    rv = torch.zeros((1, acc_v.shape[1], total_v, acc_v.shape[3], acc_v.shape[4]),
                     device=acc_v.device, dtype=acc_v.dtype)
    ra = torch.zeros((1, 32, 2, total_a), device=acc_a.device, dtype=acc_a.dtype)
    rv[:, :, :acc_v.shape[2]] = acc_v
    ra[:, :, :, :acc_a.shape[-1]] = acc_a
    v, a = chunk_v, chunk_a
    if index > 0:
        ov = min(acc_v.shape[2] - gi, v.shape[2])
        if ov > 0:
            if color_match:
                v, dc = dc_correct(v, [(v[:, :, :ov], rv[:, :, gi:gi + ov])])
                if dc is not None:
                    print(f"[H3] 🎨 chunk {index} 时间颜色匹配 |dc|max={dc.abs().max():.4f}")
            rv[:, :, gi:gi + ov] = _crossfade(rv[:, :, gi:gi + ov], v[:, :, :ov], dim=2)
            v = v[:, :, ov:]
        gi += ov
        ova = min(acc_a.shape[-1] - agi, a.shape[-1])
        if ova > 0:
            ra[:, :, :, agi:agi + ova] = _crossfade(ra[:, :, :, agi:agi + ova], a[:, :, :, :ova], dim=3)
            a = a[:, :, :, ova:]
        agi += ova
    if v.shape[2] > 0:
        rv[:, :, gi:gi + v.shape[2]] = v
    if a.shape[-1] > 0:
        ra[:, :, :, agi:agi + a.shape[-1]] = a
    return rv, ra

# ---------------------------------------------------------------------------
# 采样
# ---------------------------------------------------------------------------
def build_guider(model, cond, negative, cfg):
    guider = comfy.samplers.CFGGuider(model)
    if negative is not None:
        guider.set_conds(cond, negative)
        guider.set_cfg(cfg)
    else:
        guider.inner_set_conds({"positive": cond})
    return guider

def sample_piece(piece, guider, noise_tensor, seed, sampler, sigmas, callback=None):
    latent = dict(piece)
    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(
        model=guider.model_patcher.model, latent_image=latent_image)
    latent["samples"] = latent_image
    if callback is None:
        x0_output = {}
        callback = latent_preview.prepare_callback(guider.model_patcher, sigmas.shape[-1] - 1, x0_output)
    samples = guider.sample(noise_tensor, latent_image, sampler, sigmas,
                            denoise_mask=latent.get("noise_mask"), callback=callback,
                            disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=seed)
    return samples.to(comfy.model_management.intermediate_device())

def make_tile_progress(model_patcher, steps, n_tiles):
    previewer = latent_preview.get_previewer(model_patcher.load_device, model_patcher.model.latent_format)
    total = steps * n_tiles
    pbar = comfy.utils.ProgressBar(total)
    def for_tile(idx):
        def callback(step, x0, x, total_steps):
            preview = None
            if previewer is not None and x0 is not None:
                px0 = x0.tensors[0] if getattr(x0, "is_nested", False) else x0
                try:
                    preview = previewer.decode_latent_to_preview_image("JPEG", px0)
                except Exception:
                    preview = None
            pbar.update_absolute(idx * steps + step + 1, total, preview)
        return callback
    return for_tile

# ---------------------------------------------------------------------------
# 参数节点
# ---------------------------------------------------------------------------
class H3TemporalSplitParams(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ToolsH3TemporalSplitParams", display_name="H3 Temporal Split Params (tools)",
            category="tools/latent",
            description="H3 原生网格时间分块 (5+17m 帧) + 三重锚。",
            inputs=[
                io.Int.Input("chunk_frames", default=73, min=5, max=100000, step=1),
                io.Int.Input("temporal_overlap_frames", default=22, min=0, max=100000, step=1),
                io.Float.Input("anchor_strength", default=0.999, min=0.0, max=1.0, step=0.001),
                io.Combo.Input("motion_anchor_frames", options=["0", "5", "22", "39"], default="22"),
                io.Int.Input("identity_anchor_frames", default=24, min=0, max=240, step=1),
            ],
            outputs=[H3_TEMPORAL_PARAM.Output("temporal_split_param")],
        )

    @classmethod
    def execute(cls, chunk_frames, temporal_overlap_frames, anchor_strength,
                motion_anchor_frames, identity_anchor_frames) -> io.NodeOutput:
        chunk = snap_clip_frames(int(chunk_frames))
        overlap = snap_overlap_frames(int(temporal_overlap_frames))
        if overlap >= chunk:
            overlap = snap_overlap_frames(chunk - 17)
        return io.NodeOutput({"p": (chunk, overlap, float(anchor_strength),
                                    int(motion_anchor_frames), int(identity_anchor_frames))})


class H3SpatialSplitParams(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ToolsH3SpatialSplitParams", display_name="H3 Spatial Split Params (tools)",
            category="tools/latent",
            description="简化空间分块: 百分比重叠/渐变 + 缝邻域降噪上限。",
            inputs=[
                io.Int.Input("tile_width", default=512, min=64, max=16384, step=32),
                io.Int.Input("tile_height", default=512, min=64, max=16384, step=32),
                io.Float.Input("overlap_ratio", default=0.25, min=0.0, max=0.90, step=0.05),
                io.Float.Input("fade_ratio", default=0.50, min=0.0, max=1.0, step=0.05),
                io.Int.Input("min_tile_size", default=256, min=0, max=16384, step=32),
                io.Float.Input("seam_denoise", default=1.0, min=0.1, max=1.0, step=0.05,
                               tooltip="缝邻域降噪上限: <1 时高降噪下缝附近以中等降噪续写邻居内容, "
                                       "防止快运动物体在缝处被切断; 建议 0.5~0.8; 1.0=关。"),
            ],
            outputs=[
                H3_SPATIAL_PARAM.Output("spatial_split_param"),
                io.String.Output("grid_preview"),
            ],
        )

    @classmethod
    def execute(cls, tile_width, tile_height, overlap_ratio, fade_ratio,
                min_tile_size, seam_denoise) -> io.NodeOutput:
        tw, th = px_to_lat(tile_width), px_to_lat(tile_height)
        ol_w = min(tw - ALIGN, snap_align(tw * overlap_ratio))
        ol_h = min(th - ALIGN, snap_align(th * overlap_ratio))
        fw = min(ol_w, int(round(ol_w * fade_ratio)))
        fh = min(ol_h, int(round(ol_h * fade_ratio)))
        mt = min(px_to_lat(min_tile_size), th, tw) if min_tile_size > 0 else 0
        param = {"tw": tw, "th": th, "ol_w": ol_w, "ol_h": ol_h,
                 "fw": fw, "fh": fh, "mt": mt, "cap": float(seam_denoise)}
        preview = (f"Tile: {tile_width}x{tile_height}px -> {tw}x{th}lat | "
                   f"Overlap: {overlap_ratio:.0%} -> {ol_w}x{ol_h}lat | "
                   f"Fade: {fade_ratio:.0%} -> {fw}x{fh}lat | SeamDenoise: {seam_denoise:.2f}")
        print(f"[H3] 📊 {preview}")
        return io.NodeOutput(param, preview)


class H3TilePrompts(io.ComfyNode):
    """A prompt per tile of H3 Masked Split Upscale's grid, encoded once each.

    The grid is the upscale's own (same latent, same spatial_split_param);
    tiles are r<row>c<col> from the top left, shown on tile_map. A tile's
    prompt is template with {tile} replaced; tiles without one, and seam
    polish, use the upscale's conditioning. Only the text changes per tile:
    keyframes and references stay those of the upscale's conditioning.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ToolsH3TilePrompts", display_name="H3 Tile Prompts (tools)",
            category="tools/latent",
            description="A prompt per tile of H3 Masked Split Upscale; shows the grid with the tile names.",
            inputs=[
                io.Clip.Input("clip"),
                io.Latent.Input("latent", tooltip="The latent going into the upscale: the grid follows its size."),
                io.String.Input("prompts", multiline=True,
                                default="# r<row>c<col>: prompt, tiles from the top left (see tile_map)\n"
                                        "# r0c* = the top row, r*c0 = the left column, * = every tile\n",
                                tooltip="One tile (or a comma list) per line, 'r0c1: <prompt>'; lines without a "
                                        "tile continue the prompt above. The more specific key wins. Tiles not "
                                        "named use the upscale's conditioning."),
                io.String.Input("template", multiline=True, default="{tile}",
                                tooltip="Every tile prompt is this with {tile} replaced by the tile's text: keep "
                                        "the shared style, light and lens here, so neighbouring tiles match at "
                                        "the seams."),
                H3_SPATIAL_PARAM.Input("spatial_split_param", optional=True,
                                       tooltip="The same one the upscale gets."),
                io.Image.Input("image", optional=True,
                               tooltip="A frame of the shot to draw the grid on, any size; else a grey canvas."),
            ],
            outputs=[
                H3_TILE_PROMPTS.Output("tile_prompts"),
                io.Image.Output("tile_map"),
                io.String.Output("info"),
            ],
        )

    @classmethod
    def execute(cls, clip, latent, prompts, template="{tile}", spatial_split_param=None,
                image=None) -> io.NodeOutput:
        samples = latent["samples"]
        if not is_h3_av_latent(samples):
            raise ValueError("H3 Tile Prompts needs the MiniMax H3 AV latent that goes into the upscale")
        H, W = samples.tensors[0].shape[3], samples.tensors[0].shape[4]
        rows, cols, trows, tcols, _, _ = spatial_grid(H, W, spatial_split_param)
        texts = parse_tile_prompts(prompts, len(rows), len(cols))
        encoded, conds = {}, {}
        for tile, text in texts.items():
            full = fill_template(template, text)
            if full not in encoded:  # tiles sharing a prompt share its encoding
                encoded[full] = clip.encode_from_tokens_scheduled(clip.tokenize(full))
            conds[tile] = encoded[full]

        lines = [f"grid {len(rows)} rows x {len(cols)} columns over {W * 16}x{H * 16} px: "
                 f"{len(conds)} tile(s) with their own prompt ({len(encoded)} encoded), "
                 f"the rest use the upscale's conditioning"]
        for ri in range(len(rows)):
            for cj in range(len(cols)):
                text = " ".join(texts.get((ri, cj), "").split())
                lines.append(f"{tile_name(ri, cj)} (x {cols[cj] * 16}-{(cols[cj] + tcols[cj]) * 16}, "
                             f"y {rows[ri] * 16}-{(rows[ri] + trows[ri]) * 16}): "
                             + (text[:80] + ("…" if len(text) > 80 else "") if text else "main conditioning"))
        info = "\n".join(lines)
        print("[H3] [PROMPT] " + info.replace("\n", "\n[H3] [PROMPT]   "))
        tile_map = draw_tile_map(image, H, W, rows, cols, trows, tcols, conds)
        made = {"size": (H, W), "grid": (rows, cols, trows, tcols), "conds": conds}
        return io.NodeOutput(made, tile_map, info, ui=ui.PreviewImage(tile_map, cls=cls))


# ---------------------------------------------------------------------------
# 主节点
# ---------------------------------------------------------------------------
class MMH3SplitUpscale(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ToolsH3MaskedSplitUpscale", display_name="H3 Masked Split Upscale (tools)",
            category="tools/latent",
            description="预防(冻结预填+三重锚) + 校正(颜色匹配+钉源) + 抗分叉(seam_denoise) + probe 门控 polish。",
            inputs=[
                io.Model.Input("model"),
                io.Conditioning.Input("conditioning"),
                io.Conditioning.Input("negative", optional=True),
                io.Latent.Input("latent"),
                io.Noise.Input("noise"),
                io.Sampler.Input("sampler"),
                io.Sigmas.Input("sigmas"),
                io.Float.Input("cfg", default=1.0, min=0.0, max=100.0, step=0.1, round=0.01),
                H3_TEMPORAL_PARAM.Input("temporal_split_param", optional=True),
                H3_SPATIAL_PARAM.Input("spatial_split_param", optional=True),
                io.Combo.Input("seam_polish", options=["off", "auto", "all"], default="off"),
                io.Boolean.Input("color_match", default=True),
                io.Mask.Input("mask", optional=True,
                              tooltip="White = redraw, black = keep the input latent. One frame or one per "
                                      "frame; give it a soft edge (32+ px)."),
                io.Boolean.Input("skip_frozen_tiles", default=True,
                                 tooltip="Do not sample tiles and seams the mask leaves all black: same "
                                         "result, faster. Only with a mask."),
                io.Combo.Input("tiles", options=["grid", "around mask"], default="grid",
                               tooltip="grid: tiles over the whole frame (spatial_split_param). around mask: per "
                                       "temporal chunk, one tile_size square centred on every white blob of the "
                                       "mask (plus context); a blob too big for it gets its own small grid. "
                                       "Needs a mask."),
                io.Int.Input("tile_size", default=768, min=256, max=4096, step=32,
                             tooltip="around mask: side of the square, px. H3 works best around 768."),
                io.Int.Input("context", default=128, min=0, max=2048, step=32,
                             tooltip="around mask: px kept around a blob inside its square, so the square's edge "
                                     "lies in the frozen area."),
                io.Int.Input("mask_grow", default=32, min=-256, max=256, step=1,
                             tooltip="Widen (+) or shrink (-) the white area by this many px of the latent's frame, "
                                     "before feathering. Half of the feather falls inside the grown edge, so keep "
                                     "grow >= feather for the object's own edge to be fully redrawn; more for a "
                                     "fast, motion-blurred object."),
                io.Int.Input("mask_feather", default=32, min=0, max=256, step=1,
                             tooltip="Gaussian radius of the soft edge, px of the latent's frame: the hand-over to "
                                     "the plate runs about 2x this wide, centred on the grown edge (32 -> ~64 px, "
                                     "4 latent cells). 0 if the mask is already feathered."),
                H3_TILE_PROMPTS.Input("tile_prompts", optional=True,
                                      tooltip="From H3 Tile Prompts (same latent and spatial_split_param): a "
                                              "prompt per tile. Tiles without one, and seam polish, use "
                                              "conditioning; around mask, a window takes the prompt of the "
                                              "grid tile nearest its centre."),
            ],
            outputs=[io.Latent.Output("latent")],
        )

    @classmethod
    def execute(cls, latent, conditioning, model, noise, sampler, sigmas,
                negative=None, cfg=1.0, temporal_split_param=None,
                spatial_split_param=None, seam_polish="off",
                color_match=True, mask=None, skip_frozen_tiles=True, tiles="grid",
                tile_size=768, context=128, mask_grow=32, mask_feather=32,
                tile_prompts=None) -> io.NodeOutput:
        samples = latent["samples"]
        if not is_h3_av_latent(samples):
            raise ValueError("期望 MiniMax H3 AV latent (嵌套 video+audio)")
        video, audio = samples.tensors[0], samples.tensors[1]
        if video.shape[0] != 1:
            raise ValueError("仅支持 Batch 1")
        B, C, T, H, W = video.shape

        umask = None
        if mask is not None:
            from .latent_nodes import mask_to_h3_latent

            # grow / feather are px of the latent's frame; the mask may be smaller (drawn on the plate
            # before the upscale), so they are scaled to its pixels
            scale = mask.shape[-1] / (W * VAE_DOWNSAMPLE)
            umask = mask_to_h3_latent(mask, T, H, W, "min", round(mask_grow * scale),
                                      round(mask_feather * scale)).to(video.device)
            if mask_grow or mask_feather:
                print(f"[H3] [MASK] grow {mask_grow} px, feather {mask_feather} px")
            n_frames = frames_for_tokens(T)
            if mask.dim() == 3 and mask.shape[0] not in (1, n_frames):
                print(f"[H3] [MASK] WARNING: mask has {mask.shape[0]} frames, latent "
                      f"covers {n_frames}; extra latent tokens reuse the last mask frame")
            print(f"[H3] [MASK] mask {tuple(mask.shape)} -> latent {tuple(umask.shape)}, "
                  f"frozen share {(umask < 0.5).float().mean().item():.1%}")

        if temporal_split_param is not None:
            cl, ov, anchor_strength, motion_n, identity_n = temporal_split_param["p"]
            bounds, _ = compute_h3_segments_adaptive(T, cl, ov)
        else:
            bounds = [(0, 0, T, frames_for_tokens(T))]
            anchor_strength, motion_n, identity_n = 0.999, 0, 0

        rows, cols, trows, tcols, row_ovl, col_ovl = spatial_grid(H, W, spatial_split_param)
        seam_cap = spatial_split_param.get("cap", 1.0) if spatial_split_param is not None else 1.0
        nrows, ncols = len(rows), len(cols)

        if tile_prompts is not None:
            p_rows, p_cols, p_trows, p_tcols = tile_prompts["grid"]
            if tile_prompts["size"] != (H, W) or (tiles == "grid" and
                                                  (p_rows, p_cols, p_trows, p_tcols) != (rows, cols, trows, tcols)):
                raise ValueError(
                    f"tile_prompts were made for {len(p_rows)}x{len(p_cols)} tiles over "
                    f"{tile_prompts['size'][1] * 16}x{tile_prompts['size'][0] * 16} px, this upscale has "
                    f"{nrows}x{ncols} over {W * 16}x{H * 16} px: give H3 Tile Prompts the same latent and "
                    f"spatial_split_param")
            print(f"[H3] [PROMPT] {len(tile_prompts['conds'])} of {len(p_rows) * len(p_cols)} tiles have their "
                  f"own prompt; the others and seam polish use conditioning")
        fade_h, fade_w = (spatial_split_param["fh"], spatial_split_param["fw"]) if spatial_split_param else (0, 0)

        grid_plan = [(0, ri, cj, rows[ri], cols[cj], trows[ri], tcols[cj], row_ovl[ri], col_ovl[cj])
                     for ri in range(nrows) for cj in range(ncols)]
        plans = [grid_plan] * len(bounds)
        if tiles == "around mask":
            if umask is None:
                raise ValueError("tiles = around mask needs a mask")
            sq = px_to_lat(tile_size)
            sp = spatial_split_param or {"th": sq, "tw": sq, "ol_h": snap_align(sq * 0.25),
                                         "ol_w": snap_align(sq * 0.25), "fh": 0, "fw": 0, "mt": 16, "cap": 1.0}
            ol_ratio_h, ol_ratio_w = sp["ol_h"] / max(sp["th"], 1), sp["ol_w"] / max(sp["tw"], 1)
            fade_ratio_h = sp["fh"] / max(sp["ol_h"], 1) if sp["ol_h"] else 0.5
            fade_ratio_w = sp["fw"] / max(sp["ol_w"], 1) if sp["ol_w"] else 0.5
            lp = {"ol_h": min(sq - ALIGN, snap_align(sq * ol_ratio_h)), "ol_w": min(sq - ALIGN, snap_align(sq * ol_ratio_w)),
                  "mt": min(sp["mt"], sq)}
            fade_h, fade_w = int(round(lp["ol_h"] * fade_ratio_h)), int(round(lp["ol_w"] * fade_ratio_w))
            seam_cap = sp.get("cap", 1.0)
            ctx = int(round(context / VAE_DOWNSAMPLE))
            plans = []
            for i, (k0, f0, k1, f1) in enumerate(bounds):
                plan, wins = mask_tile_plan(umask[0, 0, k0:k1], sq, sq, ctx, lp, H, W)
                plans.append(plan)
                where = ", ".join(f"{w * 16}x{h * 16} at {c * 16},{r * 16}" for r, c, h, w in wins) or "none (no white)"
                print(f"[H3] [MASK] chunk {i}, frames {f0}-{f1 - 1}: {len(plan)} tile(s) around the mask: {where}")

        steps = max(int(sigmas.shape[-1]) - 1, 1)
        n_tiles = max(1, sum(len(p) for p in plans))
        for_tile = make_tile_progress(model.model_patcher if hasattr(model, "model_patcher") else model,
                                      steps, n_tiles)

        source = video
        noise_v = noise.generate_noise({"samples": torch.zeros_like(video, dtype=torch.float32)})
        noise_a = noise.generate_noise({"samples": torch.zeros_like(audio, dtype=torch.float32)})

        acc_v = acc_a = None
        polish_queue = {}
        tile_idx = 0

        for i, (k0, f0, k1, f1) in enumerate(bounds):
            chunk_v = video[:, :, k0:k1].contiguous()
            a0, a1 = audio_range(f0, f1)
            a1 = min(a1, audio.shape[-1])
            chunk_a = audio[:, :, :, a0:a1].contiguous()

            cond_i = reanchor_conditioning(conditioning, f0, f1, (H, W))
            if i > 0 and acc_v is not None:
                if motion_n > 0:
                    cond_i = prepend_keyframes(cond_i, motion_keyframes(acc_v, k0, f0, motion_n))
                if identity_n > 0:
                    cond_i = prepend_keyframes(cond_i, identity_keyframes(source, f0, f1, identity_n))
                if anchor_strength > 0.0:
                    cond_i = anchor_conditioning(cond_i, acc_v, f0, anchor_strength)
            elif identity_n > 0:
                cond_i = prepend_keyframes(cond_i, identity_keyframes(source, f0, f1, identity_n))

            chunk_out = chunk_v.clone()
            noise_vc = noise_v[:, :, k0:k1]
            noise_ac = noise_a[:, :, :, a0:a1]

            # ================= 空间内循环 =================
            skipped = 0
            for gi, ri, cj, r0, c0, tr, tc, ovh, ovw in plans[i]:
                comfy.model_management.throw_exception_if_processing_interrupted()

                tile = chunk_out[:, :, :, r0:r0 + tr, c0:c0 + tc].clone()

                fh, fw = fade_h, fade_w
                m = spatial_fade_mask(tr, tc, ovh, ovw,
                                      done_top=(ri > 0), done_left=(cj > 0),
                                      fade_h=fh, fade_w=fw, seam_cap=seam_cap)
                mv = m[None, None, None].to(chunk_out.device)
                um_t = None
                if umask is not None:
                    um_t = umask[:, :, k0:k1, r0:r0 + tr, c0:c0 + tc].to(chunk_out.device)
                    if skip_frozen_tiles and float(um_t.max()) <= FROZEN_EPS:
                        # all black: the sampler would hand back the input latent anyway
                        skipped += 1
                        tile_idx += 1
                        continue
                    mv = mv * um_t
                ma = torch.zeros((1, 32, 2, chunk_a.shape[-1]),
                                 device=chunk_a.device, dtype=torch.float32)
                piece = {"samples": comfy.nested_tensor.NestedTensor((tile, chunk_a)),
                         "noise_mask": comfy.nested_tensor.NestedTensor((mv, ma))}
                tile_noise = comfy.nested_tensor.NestedTensor((
                    noise_vc[:, :, :, r0:r0 + tr, c0:c0 + tc].contiguous(),
                    noise_ac.contiguous()))

                cond_src = cond_i
                if tile_prompts is not None:
                    if tiles == "grid":
                        cell = (ri, cj)
                    else:  # a window around the mask: the grid tile nearest its centre
                        cell = (cell_at(p_rows, p_trows, r0 + tr / 2), cell_at(p_cols, p_tcols, c0 + tc / 2))
                    own = tile_prompts["conds"].get(cell)
                    if own is not None:
                        # only the text changes: keyframes, refs and anchors stay those of cond_i
                        cond_src = with_tile_text(cond_i, own)
                cond_tile = crop_keyframes_to_tile(cond_src, H, W, r0, c0, tr, tc)
                guider = build_guider(model, cond_tile, negative, cfg)
                out = sample_piece(piece, guider, tile_noise, noise.seed, sampler, sigmas,
                                   for_tile(tile_idx))
                tile_v = (out.tensors[0] if out.is_nested else out).to(chunk_out.device)

                region = chunk_out[:, :, :, r0:r0 + tr, c0:c0 + tc].clone()

                if color_match:
                    tile_v, _ = dc_correct(tile_v, [
                        (tile_v[:, :, :, :, :ovw], region[:, :, :, :, :ovw])
                            if (cj > 0 and ovw > 0) else None,
                        (tile_v[:, :, :, :ovh, :], region[:, :, :, :ovh, :])
                            if (ri > 0 and ovh > 0) else None,
                        (tile_v, chunk_v[:, :, :, r0:r0 + tr, c0:c0 + tc])
                            if (ri == 0 and cj == 0) else None,
                    ])

                if um_t is not None:
                    # [MASK] hard-restore frozen area to the input latent
                    src_t = chunk_v[:, :, :, r0:r0 + tr, c0:c0 + tc].to(tile_v.device, tile_v.dtype)
                    w_t = um_t.to(tile_v.device, tile_v.dtype)
                    tile_v = src_t + (tile_v - src_t) * w_t

                if seam_polish != "off":
                    if cj > 0 and ovw > 0 and \
                            _should_polish(seam_polish, tile_v[:, :, :, :, :ovw], region[:, :, :, :, :ovw]):
                        polish_queue[(i, gi, ri, cj, "W")] = (c0, ovw, (k0, k1))
                    if ri > 0 and ovh > 0 and \
                            _should_polish(seam_polish, tile_v[:, :, :, :ovh, :], region[:, :, :, :ovh, :]):
                        polish_queue[(i, gi, ri, cj, "H")] = (r0, ovh, (k0, k1))

                if cj > 0 and ovw > 0:
                    wts = torch.linspace(0.0, 1.0, ovw, device=region.device,
                                         dtype=region.dtype).view(1, 1, 1, 1, ovw)
                    region[:, :, :, :, :ovw] = (region[:, :, :, :, :ovw] * (1.0 - wts)
                                                + tile_v[:, :, :, :, :ovw] * wts)
                if ri > 0 and ovh > 0:
                    wts = torch.linspace(0.0, 1.0, ovh, device=region.device,
                                         dtype=region.dtype).view(1, 1, 1, ovh, 1)
                    region[:, :, :, :ovh, :] = (region[:, :, :, :ovh, :] * (1.0 - wts)
                                                + tile_v[:, :, :, :ovh, :] * wts)
                band = torch.zeros((1, 1, 1, tr, tc), dtype=torch.bool, device=region.device)
                if cj > 0 and ovw > 0:
                    band[:, :, :, :, :ovw] = True
                if ri > 0 and ovh > 0:
                    band[:, :, :, :ovh, :] = True
                region = torch.where(band, region, tile_v)
                chunk_out[:, :, :, r0:r0 + tr, c0:c0 + tc] = region

                tile_idx += 1
                comfy.model_management.soft_empty_cache()

            if umask is not None and skip_frozen_tiles:
                print(f"[H3] [MASK] chunk {i}: sampled {len(plans[i]) - skipped} of {len(plans[i])} tiles")
            if color_match:
                chunk_out, dcg = grade_pin(chunk_out, chunk_v)
                if dcg is not None and dcg.abs().max().item() > 1e-4:
                    print(f"[H3] 🎯 chunk {i} 全局钉源 |dc|max={dcg.abs().max():.4f}")

            # ================= 修复版 polish 二道缝 =================
            chunk_polish = [(k, v) for k, v in polish_queue.items() if k[0] == i]
            if chunk_polish:
                print(f"[H3] 🔧 chunk {i}: polish {len(chunk_polish)} 条缝")
                pbar2 = comfy.utils.ProgressBar(steps * len(chunk_polish))
                for pi, (key, (s0, band_w, (tk0, tk1))) in enumerate(chunk_polish):
                    comfy.model_management.throw_exception_if_processing_interrupted()
                    axis = key[-1]
                    t_len = tk1 - tk0
                    if axis == "W":
                        w0 = max(0, s0 - POLISH_HALO)
                        w1 = min(W, s0 + band_w + POLISH_HALO)
                        win = chunk_out[:, :, :, :, w0:w1].clone()
                        b0, b1 = s0 - w0, s0 - w0 + band_w
                        mv = torch.zeros((1, 1, t_len, H, w1 - w0), dtype=torch.float32, device=win.device)
                        mv[:, :, :, :, b0:b1] = 1.0
                        r0c, c0c, trc, tcc, ax = 0, w0, H, w1 - w0, 4
                        nsl = (slice(None), slice(None), slice(tk0, tk1), slice(None), slice(w0, w1))
                    else:
                        h0 = max(0, s0 - POLISH_HALO)
                        h1 = min(H, s0 + band_w + POLISH_HALO)
                        win = chunk_out[:, :, :, h0:h1, :].clone()
                        b0, b1 = s0 - h0, s0 - h0 + band_w
                        mv = torch.zeros((1, 1, t_len, h1 - h0, W), dtype=torch.float32, device=win.device)
                        mv[:, :, :, b0:b1, :] = 1.0
                        r0c, c0c, trc, tcc, ax = h0, 0, h1 - h0, W, 3
                        nsl = (slice(None), slice(None), slice(tk0, tk1), slice(h0, h1), slice(None))
                    if umask is not None:
                        # [MASK] polish must not regenerate frozen area either
                        um_p = umask[nsl].to(mv.device)
                        mv = mv * um_p
                        if skip_frozen_tiles and float(mv.max()) <= FROZEN_EPS:
                            continue
                    ma = torch.zeros((1, 32, 2, chunk_a.shape[-1]),
                                     device=chunk_a.device, dtype=torch.float32)
                    piece = {"samples": comfy.nested_tensor.NestedTensor((win, chunk_a)),
                             "noise_mask": comfy.nested_tensor.NestedTensor((mv, ma))}
                    tn = comfy.nested_tensor.NestedTensor((noise_v[nsl].contiguous(), noise_ac.contiguous()))

                    cond_p = crop_keyframes_to_tile(cond_i, H, W, r0c, c0c, trc, tcc)
                    g = build_guider(model, cond_p, negative, cfg)

                    def _cb(step, x0, x, ts, _pi=pi):
                        pbar2.update_absolute(_pi * steps + step + 1, steps * len(chunk_polish))

                    out = sample_piece(piece, g, tn, noise.seed, sampler, sigmas, callback=_cb)
                    pv = (out.tensors[0] if out.is_nested else out).to(chunk_out.device)

                    nlen = (w1 - w0) if ax == 4 else (h1 - h0)
                    alpha = torch.ones(nlen, dtype=torch.float32)
                    if b0 > 0:
                        alpha[:b0] = torch.linspace(0.0, 1.0, b0)
                    if nlen - b1 > 0:
                        alpha[b1:] = torch.linspace(1.0, 0.0, nlen - b1)
                    view = [1, 1, 1, 1, 1]
                    view[ax] = nlen
                    avv = alpha.view(view).to(chunk_out.device)
                    if ax == 4:
                        chunk_out[:, :, :, :, w0:w1] = avv * pv + (1.0 - avv) * win
                    else:
                        chunk_out[:, :, :, h0:h1, :] = avv * pv + (1.0 - avv) * win
                    comfy.model_management.soft_empty_cache()

            acc_v, acc_a = temporal_append(acc_v, acc_a, chunk_out, chunk_a, i, k0, f0, color_match)

        if umask is not None:
            # [MASK] final restore: grade_pin, polish and the temporal colour
            # match shift whole chunks after the per-tile restore
            n = acc_v.shape[2]
            src_all = source[:, :, :n].to(acc_v.device, acc_v.dtype)
            w_all = umask[:, :, :n].to(acc_v.device, acc_v.dtype)
            acc_v = src_all + (acc_v - src_all) * w_all

        if hasattr(model, "clone_base_uuid"):
            comfy.model_management.unload_model_and_clones(model, unload_additional_models=False)
            comfy.model_management.soft_empty_cache()
        return io.NodeOutput({"samples": comfy.nested_tensor.NestedTensor((acc_v, acc_a))})


NODE_CLASS_MAPPINGS = {
    "ToolsH3TemporalSplitParams": H3TemporalSplitParams,
    "ToolsH3SpatialSplitParams": H3SpatialSplitParams,
    "ToolsH3TilePrompts": H3TilePrompts,
    "ToolsH3MaskedSplitUpscale": MMH3SplitUpscale,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ToolsH3TemporalSplitParams": "H3 Temporal Split Params (tools)",
    "ToolsH3SpatialSplitParams": "H3 Spatial Split Params (tools)",
    "ToolsH3TilePrompts": "H3 Tile Prompts (tools)",
    "ToolsH3MaskedSplitUpscale": "H3 Masked Split Upscale (tools)",
}
