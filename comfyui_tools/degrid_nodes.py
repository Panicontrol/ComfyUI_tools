"""H3 DeGrid: remove the MiniMax H3 decoder's 8 px block grid from decoded frames.

The H3 VAE leaves a grid of 8x8 px blocks tied to the frame origin: a jump in
tone across every block boundary, flat plateaus inside blocks, and a 2 px
parity (steps on odd positions larger than on even ones). The blocks shift
independently, with their own sign, frame to frame, so no fixed pattern or
notch filter removes them; this is deblocking -- the jump across each block
boundary is measured against the steps beside it and spread back as a ramp.

All work is per RGB channel in float32 on the GPU, a few frames at a time; the
result stays float (no 8-bit rounding), alpha passes through. Values below
are in 0-255 units, as in the node's inputs.
"""

import math

import torch
import torch.nn.functional as F

from .utils import CATEGORY

U = 1.0 / 255.0
PARITY_SIGMA = 2.0
PARITY_LIMIT = 1.5 * U
EDGE_WEIGHTS = ["above grain", "linear"]
CLEAN_BELOW = 1.05
PLATEAU_FLAT_SHARE = 0.75  # a block takes part in plateau when this share of it is flat
OTHER_GRID_BELOW = 1.15  # a weak grid at period, the frame's own grid elsewhere: resized, leave it
FLAT_PERCENTILE = 0.4


def compute_device():
    try:
        import comfy.model_management

        return comfy.model_management.get_torch_device()
    except Exception:
        return torch.device("cpu")


# ---------------------------------------------------------------------------
# filters
# ---------------------------------------------------------------------------

def gaussian_kernel(sigma, device):
    radius = max(1, int(math.ceil(3 * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def conv_axis(t, kernel, dim):
    """t [N, C, H, W] convolved with a 1-D kernel along dim (2 or 3), edges replicated."""
    c, r = t.shape[1], (kernel.numel() - 1) // 2
    if dim % 4 == 3:
        weight, pad = kernel.view(1, 1, 1, -1), (r, r, 0, 0)
    else:
        weight, pad = kernel.view(1, 1, -1, 1), (0, 0, r, r)
    return F.conv2d(F.pad(t, pad, mode="replicate"), weight.expand(c, 1, -1, -1).contiguous(), groups=c)


def blur(t, sigma, dims):
    """Gaussian blur of t [N, C, H, W] along dims (2, 3)."""
    if sigma <= 0:
        return t
    k = gaussian_kernel(sigma, t.device)
    for dim in dims:
        t = conv_axis(t, k, dim)
    return t


def sample_last(t, pos):
    """t at fractional positions pos along its last dimension (linear interpolation)."""
    pos = pos.clamp(0, t.shape[-1] - 1)
    i0 = pos.floor().long().clamp(max=t.shape[-1] - 2) if t.shape[-1] > 1 else pos.long()
    f = (pos - i0).to(t.dtype)
    a = t.index_select(-1, i0)
    b = t.index_select(-1, (i0 + 1).clamp(max=t.shape[-1] - 1))
    return a + (b - a) * f


# ---------------------------------------------------------------------------
# grid
# ---------------------------------------------------------------------------

def boundary_centres(size, period, phase):
    """Block boundaries x = k * period + phase (between pixels x-1 and x) as pixel-centre
    coordinates x - 0.5, for every boundary with two pixels on each side."""
    k0 = math.ceil((2.0 - phase) / period)
    k1 = math.floor((size - 2.0 - phase) / period)
    return [phase + k * period - 0.5 for k in range(k0, k1 + 1)]


def deblock_axis(x, period, phase, max_corr, edge_thr, smooth_along, dim, edge_weight="above grain"):
    """Correction for the block boundaries across dim (3 = vertical lines, 2 = horizontal).

    Per boundary: the step across it minus the mean of the steps beside it,
    faded out where those steps are a real edge, smoothed along the line,
    limited, and spread as a ramp over period / 2 px on each side (half up on
    the low side, half down on the high side, most at the boundary).

    The fade: linear, 1 - step / edge_thr, already takes ~13% off every jump
    on flat ground, whose grain alone makes steps of ~0.8 (0-255); above grain
    measures the steps above the frame's usual one (its median, at most a
    third of edge_thr), so flat ground is corrected in full and edges still
    are not.
    """
    t = x.movedim(dim, -1)
    length = t.shape[-1]
    centres = boundary_centres(length, period, phase)
    if not centres:
        return torch.zeros_like(x)
    u = torch.tensor(centres, device=t.device, dtype=torch.float32)
    v = [sample_last(t, u + o) for o in (-1.5, -0.5, 0.5, 1.5)]
    d_before, d_across, d_after = v[1] - v[0], v[2] - v[1], v[3] - v[2]
    excess = d_across - (d_before + d_after) / 2
    beside = torch.maximum(d_before.abs(), d_after.abs())
    if edge_weight == "linear":
        weight = (1 - beside / edge_thr).clamp(0, 1)
    else:
        grain = beside.flatten(-2).median(-1).values[..., None, None].clamp(max=edge_thr / 3)
        weight = (1 - (beside - grain).clamp(min=0) / (edge_thr - grain)).clamp(0, 1)
    corr = blur(excess * weight, smooth_along, [2]).clamp(-max_corr, max_corr)

    # every pixel takes the ramp of its nearest boundary
    pos = torch.arange(length, device=t.device, dtype=torch.float32)
    nearest = ((pos + 0.5 - phase) / period).round() - math.ceil((2.0 - phase) / period)
    nearest = nearest.long().clamp(0, len(centres) - 1)
    s = pos - u[nearest]
    ramp = (1 - s.abs() / (period / 2)).clamp(min=0)
    coef = -torch.sign(s) * ramp / 2
    return (corr.index_select(-1, nearest) * coef).movedim(-1, dim)


def parity_axis(x, dim, weight):
    """The 2 px alternation along dim: the Nyquist component, demodulated and smoothed
    with weights (a normalized convolution, so dense texture next to a calm area does not
    leak into its estimate), limited. Subtract it."""
    k = torch.tensor([1.0, -4.0, 6.0, -4.0, 1.0], device=x.device) / 16
    n = x.shape[dim]
    shape = [1, 1, 1, 1]
    shape[dim] = n
    sign = (1.0 - 2.0 * (torch.arange(n, device=x.device) % 2).to(x.dtype)).view(shape)
    amp = blur(conv_axis(x, k, dim) * sign * weight, PARITY_SIGMA, [3, 2])
    amp = amp / blur(weight, PARITY_SIGMA, [3, 2]).clamp(min=0.05)
    return amp.clamp(-PARITY_LIMIT, PARITY_LIMIT) * sign


def parity_correction(x, edge_thr):
    """Correction for the 2 px parity along x and y, weighted by how calm the picture is:
    where the mean step reaches edge_thr (dense texture) the filter would take real detail,
    so it does nothing there.

    One pass: a second one, before deblocking, mostly takes the filter's answer to the
    block steps, and after it the answer to the ramps' kinks; both made grid and parity
    worse in tests."""
    dx = (x[..., :, 1:] - x[..., :, :-1]).abs()
    dy = (x[..., 1:, :] - x[..., :-1, :]).abs()
    steps = (F.pad(dx, (1, 0, 0, 0), mode="replicate") + F.pad(dy, (0, 0, 1, 0), mode="replicate")) / 2
    calm = (1 - blur(steps, PARITY_SIGMA, [3, 2]) / edge_thr).clamp(0, 1)
    return -(parity_axis(x, 3, calm) + parity_axis(x, 2, calm)) * calm


def block_index(size, period, phase, device):
    pos = torch.arange(size, device=device, dtype=torch.float32)
    idx = torch.floor((pos - phase) / period).long()
    return idx - idx.min()


def bilinear_from_centres(values, size, period, phase, dim):
    """Values per block (along dim) bilinearly spread over pixels, block centres exact."""
    t = values.movedim(dim, -1)
    n = t.shape[-1]
    pos = torch.arange(size, device=t.device, dtype=torch.float32)
    first = math.floor((0 - phase) / period)  # block index 0 starts at this k
    q = (pos + 0.5 - phase) / period - 0.5 - first
    q0 = q.floor().clamp(0, max(n - 1, 0))
    f = (q - q0).clamp(0, 1)
    i0 = q0.long()
    i1 = (i0 + 1).clamp(max=n - 1)
    out = t.index_select(-1, i0) * (1 - f) + t.index_select(-1, i1) * f
    return out.movedim(-1, dim)


def box(t, size):
    """size x size mean (separable), borders averaged over the pixels inside."""
    t = F.avg_pool2d(t, (size, 1), 1, (size // 2, 0), count_include_pad=False)
    return F.avg_pool2d(t, (1, size), 1, (0, size // 2), count_include_pad=False)


def local_sigma(x, size=9):
    mean = box(x, size)
    return (box(x * x, size) - mean * mean).clamp(min=0).sqrt()


def percentile_per_frame(t, q):
    """q-quantile of every [.., H, W] map, from a strided sample (fast on 4K frames)."""
    step = max(1, int(math.sqrt(t.shape[-1] * t.shape[-2] / 250_000)))
    flat = t[..., ::step, ::step].reshape(*t.shape[:-2], -1)
    k = max(1, int(round(q * flat.shape[-1])))
    return flat.kthvalue(k, dim=-1).values[..., None, None]


def plateau_correction(x, period, phase_x, phase_y, max_corr, edge_thr):
    """Pull the tone of flat blocks towards a smooth surface through their neighbours.

    Only flat ground takes part: a block's tone is the mean of its flat pixels
    (local sigma below the frame's 40th percentile), only blocks that are
    mostly flat are corrected or used as neighbours, and neighbours across a
    real step in tone are left out. The difference to the neighbours' mean,
    limited, is spread bilinearly between block centres and faded out near
    edges and texture, so an edge through a block (a windowsill) is never
    shifted and only the plateau's tone moves, not its texture.
    """
    n, c, h, w = x.shape
    bx = block_index(w, period, phase_x, x.device)
    by = block_index(h, period, phase_y, x.device)
    nbx, nby = int(bx.max()) + 1, int(by.max()) + 1

    def per_block(t):
        t = torch.zeros(n, c, h, nbx, device=x.device).index_add_(3, bx, t)
        return torch.zeros(n, c, nby, nbx, device=x.device).index_add_(2, by, t)

    sigma = local_sigma(x)
    flat = (sigma < percentile_per_frame(sigma, FLAT_PERCENTILE)).to(x.dtype)
    flat_count = per_block(flat)
    counts = torch.outer(torch.bincount(by), torch.bincount(bx)).to(x.dtype)
    valid = (flat_count >= PLATEAU_FLAT_SHARE * counts).to(x.dtype)
    means = per_block(x * flat) / flat_count.clamp(min=1)

    padded_m = F.pad(means, (1, 1, 1, 1), mode="replicate")
    padded_v = F.pad(valid, (1, 1, 1, 1))
    total = torch.zeros_like(means)
    used = torch.zeros_like(means)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            if dy == 1 and dx == 1:
                continue
            nb = padded_m[..., dy:dy + nby, dx:dx + nbx]
            ok = padded_v[..., dy:dy + nby, dx:dx + nbx] * ((nb - means).abs() <= edge_thr).to(x.dtype)
            total += nb * ok
            used += ok
    delta = torch.where(used >= 3, total / used.clamp(min=1) - means, torch.zeros_like(means))
    delta = delta.clamp(-max_corr, max_corr) * valid

    field = bilinear_from_centres(delta, w, period, phase_x, 3)
    field = bilinear_from_centres(field, h, period, phase_y, 2)
    return field * blur(flat, 1.5, [3, 2])


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------

def step_profiles(x, flat):
    """Mean |step| over flat pixels: per column (steps along x) and per row (along y).

    x, flat: [N, C, H, W]. Profile index i is the step between pixels i-1 and i;
    index 0 is unused.
    """
    dx = (x[..., 1:] - x[..., :-1]).abs()
    mx = flat[..., 1:] * flat[..., :-1]
    px = (dx * mx).sum(-2) / mx.sum(-2).clamp(min=1)
    dy = (x[..., 1:, :] - x[..., :-1, :]).abs()
    my = flat[..., 1:, :] * flat[..., :-1, :]
    py = (dy * my).sum(-1) / my.sum(-1).clamp(min=1)
    return F.pad(px, (1, 0)), F.pad(py, (1, 0))


def grid_ratio(profile, period, phase=None):
    """Mean step on the grid lines / between them (1.0 = no grid), max over phase when
    phase is None. profile: [..., L]; returns [...]."""
    length = profile.shape[-1]
    pos = torch.arange(length, device=profile.device, dtype=torch.float32)
    valid = pos >= 1
    integer = abs(period - round(period)) < 1e-3
    phases = [phase] if phase is not None else (
        list(range(int(round(period)))) if integer else [p * 0.25 for p in range(int(period * 4))])
    best = None
    for ph in phases:
        rel = torch.remainder(pos - ph, period)
        line = (torch.minimum(rel, period - rel) < 0.5) & valid
        mid = ((rel - period / 2).abs() < 0.5) & valid
        if not line.any() or not mid.any():
            continue
        r = (profile[..., line].mean(-1)) / profile[..., mid].mean(-1).clamp(min=1e-9)
        best = r if best is None else torch.maximum(best, r)
    return best if best is not None else torch.ones(profile.shape[:-1], device=profile.device)


def measure(x):
    """Local sigma and the step profiles over flat pixels (sigma below the frame's 40th
    percentile) of x [N, C, H, W]: what grid_ratio reads."""
    sigma = local_sigma(x)
    flat = (sigma < percentile_per_frame(sigma, FLAT_PERCENTILE)).to(x.dtype)
    px, py = step_profiles(x, flat)
    return sigma, px, py


def tone_ratio(profile, period):
    """Grid strength from the amplitude of the period's own frequency in a step profile,
    in grid_ratio's units (1.0 = no grid). Unlike grid_ratio it does not alias: a grid of
    another period (8 px after a 4/3 resize is 10.67) reads ~1.0, not 1.2."""
    p = profile[..., 1:]
    pos = torch.arange(1, profile.shape[-1], device=profile.device, dtype=torch.float32)
    ang = 2 * math.pi * pos / period
    re, im = (p * torch.cos(ang)).mean(-1), (p * torch.sin(ang)).mean(-1)
    amp = 2 * (re * re + im * im).sqrt() / p.mean(-1).clamp(min=1e-9)
    return 1 + amp * period / 2


def find_period(profile, lo=5.0, hi=32.0, step=0.01, prominence=6.0):
    """Grid period from a step profile; None when there is no grid.

    A grid line is a narrow peak of the profile's spectrum, the picture a broad
    hump, so amplitudes are judged against the spectrum's local level around
    them. The period with the strongest prominent fundamental may still be a
    harmonic of the grid (after a 2x resize the 16 px grid's 2nd harmonic
    outweighs its fundamental), so its multiples are checked: one whose own
    fundamental is clearly there is the grid, the longest such one wins; a
    real grid has nothing at its multiples' fundamentals.
    """
    window = int(hi) + 1 if int(hi) % 2 == 0 else int(hi) + 2
    trend = F.avg_pool1d(profile[None, None].float(), window, 1, window // 2, count_include_pad=False)[0, 0]
    p = (profile - trend[: profile.numel()])[1:].double().cpu()
    n_fft = 1 << max(17, math.ceil(math.log2(p.numel() * 16)))
    spec = torch.fft.rfft(p, n=n_fft).abs()
    # the local level: mean over a ring from 2 to 20 peak widths around each bin, the peak left out
    peak = n_fft / p.numel()
    total = F.pad(spec.cumsum(0), (1, 0))
    idx = torch.arange(spec.numel())

    def ring_sum(radius):
        lo_, hi_ = (idx - radius).clamp(0, spec.numel()), (idx + radius + 1).clamp(0, spec.numel())
        return total[hi_] - total[lo_], (hi_ - lo_).double()

    wide, n_wide = ring_sum(int(20 * peak))
    near, n_near = ring_sum(int(2 * peak))
    level = (wide - near) / (n_wide - n_near).clamp(min=1)

    def at(freq):
        idx = (freq * n_fft).round().long().clamp(0, spec.numel() - 1)
        return spec[idx], spec[idx] / level[idx].clamp(min=1e-12)

    periods = torch.arange(lo, hi + step / 2, step, dtype=torch.float64)
    amp, prom = at(1 / periods)
    amp = torch.where(prom >= prominence, amp, torch.zeros_like(amp))
    if float(amp.max()) <= 0:
        return None
    period, top = float(periods[int(amp.argmax())]), float(amp.max())
    best = 1
    for m in range(2, int(hi / period) + 1):
        a, pr = at(torch.tensor([1 / (m * period)], dtype=torch.float64))
        if float(pr) >= prominence / 2 and float(a) >= 0.2 * top:  # weaker than the harmonic, still a peak
            best = m
    if best > 1:  # refine the multiple on its own peak
        near = torch.arange(best * period - 0.15, best * period + 0.15, step, dtype=torch.float64)
        period = float(near[int(at(1 / near)[0].argmax())])
    return float(round(period)) if abs(period - round(period)) < 0.05 else period


def find_phase(profile, period):
    """Where the grid lines of a step profile are: x0 such that lines sit at x0 + k * period."""
    p = (profile - profile.mean()).double()
    pos = torch.arange(p.numel(), dtype=torch.float64)
    ang = 2 * math.pi * pos / period
    re, im = float((p * torch.cos(ang)).sum()), float((p * torch.sin(ang)).sum())
    phase = (math.atan2(im, re) * period / (2 * math.pi)) % period
    if abs(period - round(period)) < 1e-3:
        phase = float(round(phase) % round(period))
    return phase


# ---------------------------------------------------------------------------
# node
# ---------------------------------------------------------------------------

class H3DeGrid:
    """Remove the MiniMax H3 decoder's 8 px block grid (and its 2 px parity) from frames.

    Use on the H3 resolution itself (1344 / 1920 / 2880 ...), after VAE Decode
    or on a loaded PNG sequence, before any resize: a resize by s moves the
    grid to 8 * s px and blurs it. Frames keep their size; only block
    boundaries (and, with plateau, the tone of flat blocks) change.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "period": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 64.0, "step": 0.01,
                                     "tooltip": "Grid step in px; 8 on H3's own resolution, 8 * s after a "
                                                "resize by s. 0 = find it."}),
                "phase_x": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 64.0, "step": 0.01,
                                      "tooltip": "Column of the first vertical block boundary; 0 on H3's own "
                                                 "resolution. -1 = find it, once for the whole sequence."}),
                "phase_y": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 64.0, "step": 0.01,
                                      "tooltip": "Row of the first horizontal block boundary; -1 = find it."}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                       "tooltip": "Share of the correction applied."}),
                "max_correction": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 32.0, "step": 0.1,
                                             "tooltip": "Largest change at a block boundary, in 0-255 units."}),
                "edge_threshold": ("FLOAT", {"default": 6.0, "min": 0.1, "max": 64.0, "step": 0.1,
                                             "tooltip": "Steps beside a boundary larger than this (0-255) mean a "
                                                        "real edge: the correction fades to 0 there."}),
                "smooth_along": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 32.0, "step": 0.1,
                                           "tooltip": "Gaussian smoothing (px) of the measured jump along each "
                                                      "boundary, so a block is corrected, not the noise."}),
                "plateau": (["off", "on"], {"default": "off",
                                            "tooltip": "Also pull the tone of flat blocks towards a smooth "
                                                       "surface through their neighbours (the squares a strong "
                                                       "grade shows); texture inside blocks stays."}),
                "parity": ("BOOLEAN", {"default": True, "tooltip": "Remove the 2 px alternation."}),
                "skip_when_clean": ("BOOLEAN", {"default": True,
                                                "tooltip": "Frames whose grid (and parity) measure below 1.05 "
                                                           "pass through untouched."}),
                "h3_sizes_only": ("BOOLEAN", {"default": True,
                                              "tooltip": "Only frames whose sides are multiples of 32 px -- H3's "
                                                         "own sizes (1344x768, 1920x1088, 2880x1632, ...) -- are "
                                                         "processed; others (fal 4K 3840x2160, 1080p) pass "
                                                         "through untouched."}),
                "edge_weight": (EDGE_WEIGHTS, {"default": "above grain",
                                               "tooltip": "How the correction fades towards real edges. above "
                                                          "grain: by the steps beside a boundary above the frame's "
                                                          "usual (grain) step, so flat ground is corrected in full. "
                                                          "linear: by the whole step, as first specified."}),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Where to work (white); none = the whole frame."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("images", "removed", "info")
    FUNCTION = "degrid"
    CATEGORY = f"{CATEGORY}/image"
    DESCRIPTION = ("Remove the MiniMax H3 decoder's 8 px block grid and 2 px parity from frames, at H3's own "
                   "resolution, without blurring detail.")

    def degrid(self, images, period=8.0, phase_x=0.0, phase_y=0.0, strength=1.0, max_correction=4.0,
               edge_threshold=6.0, smooth_along=3.0, plateau="off", parity=True, skip_when_clean=True,
               h3_sizes_only=True, edge_weight="above grain", mask=None):
        device = compute_device()
        count, height, width = int(images.shape[0]), int(images.shape[1]), int(images.shape[2])
        max_corr, edge_thr = max_correction * U, edge_threshold * U

        if h3_sizes_only and (width % 32 or height % 32):
            info = (f"{width}x{height} is not an H3 size (sides not multiples of 32 px): all {count} frames "
                    f"passed through untouched (h3_sizes_only)")
            grey = torch.full((*images.shape[:3], 3), 0.5, dtype=images.dtype)
            return {"ui": {"text": [info]}, "result": (images, grey, info)}

        # period and phase: once for the whole sequence, from up to 8 frames
        notes = []
        if period <= 0 or phase_x < 0 or phase_y < 0:
            picks = sorted({round(i * (count - 1) / 7) for i in range(8)}) if count > 1 else [0]
            sample = images[picks, ..., :3].to(device, torch.float32).permute(0, 3, 1, 2)
            _, px, py = measure(sample)
            px, py = px.mean((0, 1)).cpu(), py.mean((0, 1)).cpu()
            if period <= 0:
                # the axes may disagree on a resized grid: keep the period that measures strongest on both
                found = {f for f in (find_period(px), find_period(py)) if f}
                strength_at = lambda p: float(tone_ratio(px[None], p) + tone_ratio(py[None], p))
                period = max(found, key=strength_at) if found else 8.0
                notes.append(f"period {period:g} found" if found else "no grid found, period 8 used")
            if phase_x < 0:
                phase_x = find_phase(px, period)
            if phase_y < 0:
                phase_y = find_phase(py, period)
        period, phase_x, phase_y = float(period), float(phase_x) % period, float(phase_y) % period

        per_frame = height * width * 3 * 4
        chunk = max(1, min(count, int(1.5e9 // (per_frame * 14))))
        out_frames, removed_frames = [], []
        stats = {"before": [], "after": []}
        skipped = resized = 0
        for start in range(0, count, chunk):
            src = images[start:start + chunk]
            x = src[..., :3].to(device, torch.float32).permute(0, 3, 1, 2).contiguous()
            _, px, py = measure(x)
            before = self._ratios(px, py, period, parity)
            stats["before"].append(before)

            clean = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
            if skip_when_clean:
                grid = torch.stack([tone_ratio(px, period), tone_ratio(py, period)], 0).amax((0, 2))
                tones = grid
                if parity:
                    tones = torch.maximum(tones, torch.stack([tone_ratio(px, 2.0), tone_ratio(py, 2.0)], 0).amax((0, 2)))
                clean = tones < CLEAN_BELOW
                for i in range(x.shape[0]):
                    if not clean[i] and grid[i] < OTHER_GRID_BELOW:
                        own = find_period(px[i].mean(0).cpu())
                        # the frame's own grid elsewhere, and clearly stronger there
                        if own and abs(own - period) > 0.5 and \
                                float(tone_ratio(px[i:i + 1], own).amax()) - 1 > 2 * (float(grid[i]) - 1):
                            clean[i] = True
                            resized += 1
            skipped += int(clean.sum())

            y = x
            if not bool(clean.all()):
                if parity:  # first: the alternation would read as a jump at every boundary
                    y = y + strength * parity_correction(y, edge_thr)
                y = y + strength * deblock_axis(y, period, phase_x, max_corr, edge_thr, smooth_along, 3, edge_weight)
                y = y + strength * deblock_axis(y, period, phase_y, max_corr, edge_thr, smooth_along, 2, edge_weight)
                if plateau == "on":
                    y = y + strength * plateau_correction(y, period, phase_x, phase_y, max_corr, edge_thr)
                y = torch.where(clean[:, None, None, None], x, y)
            if mask is not None:
                y = x + (y - x) * self._mask(mask, start, x.shape[0], height, width, x.device)
            y = y.clamp(0, 1)

            _, qx, qy = measure(y)
            stats["after"].append(self._ratios(qx, qy, period, parity))

            frames = y.permute(0, 2, 3, 1).cpu()
            if src.shape[-1] > 3:  # alpha untouched
                frames = torch.cat([frames, src[..., 3:].to(frames.dtype)], -1)
            out_frames.append(frames)
            removed_frames.append(((y - x) * 10 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).cpu())

        if resized:
            notes.append(f"{resized} frame(s) carry a grid of another period (resized?) and were left alone")
        info = self._info(stats, period, phase_x, phase_y, parity, skipped, count, notes)
        return {"ui": {"text": [info]}, "result": (torch.cat(out_frames, 0), torch.cat(removed_frames, 0), info)}

    @staticmethod
    def _ratios(px, py, period, parity):
        ratios = [grid_ratio(px, period), grid_ratio(py, period)]
        if parity:
            ratios += [grid_ratio(px, 2.0), grid_ratio(py, 2.0)]
        return ratios

    @staticmethod
    def _mask(mask, start, n, height, width, device):
        m = mask if mask.dim() == 3 else mask[None]
        idx = [min(start + i, m.shape[0] - 1) for i in range(n)]
        m = m[idx].to(device, torch.float32)[:, None]
        if m.shape[-2:] != (height, width):
            m = F.interpolate(m, size=(height, width), mode="bilinear", align_corners=False)
        return m.clamp(0, 1)

    @staticmethod
    def _info(stats, period, phase_x, phase_y, parity, skipped, count, notes):
        def red(key, i):  # mean over frames, R channel
            return float(torch.cat([r[i][:, 0] for r in stats[key]]).mean())

        line = (f"period {period:g}, phase x {phase_x:g} / y {phase_y:g}"
                + (f" ({'; '.join(notes)})" if notes else "")
                + f"; grid x/y (R): {red('before', 0):.2f}/{red('before', 1):.2f} -> "
                  f"{red('after', 0):.2f}/{red('after', 1):.2f}")
        if parity:
            line += (f"; parity x/y (R): {red('before', 2):.2f}/{red('before', 3):.2f} -> "
                     f"{red('after', 2):.2f}/{red('after', 3):.2f}")
        return line + f"; {skipped} of {count} frames clean, passed through"


NODE_CLASS_MAPPINGS = {
    "ToolsH3DeGrid": H3DeGrid,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ToolsH3DeGrid": "H3 DeGrid (tools)",
}
