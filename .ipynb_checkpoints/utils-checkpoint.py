import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import as_strided
from scipy.ndimage import distance_transform_edt

def fun(r, p, rnorm=None):
    if rnorm is not None:
        m = r >= rnorm
        r /= rnorm
        r = np.minimum(1, r)

    a = 1 - np.exp(-(1-r)**p)
    b = 1 - np.exp(-1)
    res = a/b
    res[m] = 0
    return a/b

# ──────────────────────────────────────────────
# Core routines
# ──────────────────────────────────────────────

def validate_extent(bds, resolution='1km', ext_vrt=None, buffer=0):
    if resolution == '100m':
        CRS_TRANSFORM = [-180.0, 0.00083333333, 0.0, 84.0, 0.0, -0.00083333333]
    if resolution in ['1km', '1km_ua', '1000m']:
        CRS_TRANSFORM = [-180.0, 0.0083333333, 0.0, 84.0, 0.0, -0.0083333333]
    
    a = CRS_TRANSFORM[1]
    b = CRS_TRANSFORM[5]
    if buffer > 0:
        bds[0] -= (buffer-0.5)*a
        bds[1] += (buffer-0.5)*b
        bds[2] += (buffer-0.5)*a
        bds[3] -= (buffer-0.5)*b
        
    if ext_vrt is not None:
        bds[0] = max(bds[0], ext_vrt[0]+0.5*a)
        bds[1] = max(bds[1], ext_vrt[1]-0.5*b)
        bds[2] = min(bds[2], ext_vrt[2]-0.5*a)
        bds[3] = min(bds[3], ext_vrt[3]+0.5*b)
        
    b0 = CRS_TRANSFORM[0] + a*np.floor((bds[0]-CRS_TRANSFORM[0])/a)
    b1 = CRS_TRANSFORM[3] + b*np.ceil((bds[1]-CRS_TRANSFORM[3])/b)
    b2 = CRS_TRANSFORM[0] + a*np.ceil((bds[2]-CRS_TRANSFORM[0])/a)
    b3 = CRS_TRANSFORM[3] + b*np.floor((bds[3]-CRS_TRANSFORM[3])/b)
            
    ext = (float(b0), float(b1), float(b2), float(b3))
    return ext

def get_trf(bds, resolution='1km'):
    if resolution == '100m':
        d = 0.00083333333
    if resolution in ['1km', '1km_ua', '1000m']:
        d = 0.0083333333
    bds = validate_extent(bds, resolution=resolution)
    trf = [bds[0], d, 0.0, bds[3], 0.0, -d]
    return trf

def max_conv_1d(
    arr: np.ndarray,
    kernel: np.ndarray,
    mode: str = "valid",
) -> np.ndarray:
    """1-D max-convolution.

    Parameters
    ----------
    arr    : 1-D input array.
    kernel : 1-D kernel (only its length is used).
    mode   : 'valid', 'same', or 'full'.

    Returns
    -------
    1-D output array of dtype float64.
    """
    arr = np.asarray(arr, dtype=np.float64)
    k = len(kernel)

    pad = _pad_width_1d(len(arr), k, mode)
    padded = np.pad(arr, pad, constant_values=0.0)

    n_out = len(padded) - k + 1
    # Build a view with shape (n_out, k) — no copy
    shape   = (n_out, k)
    strides = (padded.strides[0], padded.strides[0])
    windows = as_strided(padded, shape=shape, strides=strides)
    return windows.max(axis=1)


def max_conv_2d(
    arr: np.ndarray,
    kernel: np.ndarray,
    mode: str = "valid",
    *,
    weighted: bool = False,
) -> np.ndarray:
    """2-D max-convolution.

    Parameters
    ----------
    arr      : 2-D input array.
    kernel   : 2-D kernel.  When weighted=False only its shape is used;
               when weighted=True its values scale each window element
               before the max is taken.
    mode     : 'valid', 'same', or 'full'.
    weighted : If False (default), returns max(window) — unweighted.
               If True, returns max(window * kernel) at each position,
               so kernel values act as importance weights that suppress
               or amplify individual elements before the maximum is found.

    Returns
    -------
    2-D output array of dtype float64.
    """
    arr    = np.asarray(arr,    dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    kh, kw = kernel.shape
    H, W   = arr.shape

    pad_h = _pad_width_1d(H, kh, mode)
    pad_w = _pad_width_1d(W, kw, mode)
    padded = np.pad(arr, (pad_h, pad_w), constant_values=0.0)

    Ph, Pw = padded.shape
    out_h  = Ph - kh + 1
    out_w  = Pw - kw + 1

    # Build a zero-copy view with shape (out_h, out_w, kh, kw)
    s0, s1  = padded.strides
    windows = as_strided(
        padded,
        shape   = (out_h, out_w, kh, kw),
        strides = (s0, s1, s0, s1),
    )

    if weighted:
        # Scale each window element by the corresponding kernel weight,
        # then take the max.  Materialises the full broadcast array —
        # unavoidable since max has no fused weighted form like einsum.
        return (windows * kernel).max(axis=(-2, -1))
    else:
        return windows.max(axis=(-2, -1))


# ──────────────────────────────────────────────
# Standard (multiply-accumulate) convolution
# ──────────────────────────────────────────────

def conv_2d(
    arr: np.ndarray,
    kernel: np.ndarray,
    mode: str = "valid",
) -> np.ndarray:
    """Low-level 2-D convolution (multiply-accumulate).

    Slides a flipped kernel over *arr* and computes the dot product at every
    position — the standard linear convolution definition.  Uses the same
    zero-copy as_strided window view as max_conv_2d, with a single einsum
    replacing the max reduction, so there are no Python-level loops.

    The kernel is flipped along both axes (as the strict mathematical
    definition requires).  For symmetric kernels (Gaussian, Laplacian, …)
    flipping is a no-op.  Pass the kernel already flipped, or use
    correlate_2d (not included here) if you want cross-correlation instead.

    Parameters
    ----------
    arr    : 2-D input array.
    kernel : 2-D kernel whose values are used as weights.
    mode   : 'valid', 'same', or 'full'.

    Returns
    -------
    np.ndarray of shape determined by *mode*, dtype float64.

    Examples
    --------
    >>> # Gaussian blur
    >>> σ = 1.0
    >>> g = radial_kernel(5, lambda r: np.exp(-r**2 / (2*σ**2)), normalize=True)
    >>> blurred = conv_2d(image, g, mode='same')

    >>> # Laplacian edge detection
    >>> lap = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=float)
    >>> edges = conv_2d(image, lap, mode='same')
    """
    arr    = np.asarray(arr,    dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)

    if arr.ndim != 2 or kernel.ndim != 2:
        raise ValueError("conv_2d requires 2-D arr and kernel.")

    kh, kw = kernel.shape
    H,  W  = arr.shape

    pad_h = _pad_width_1d(H, kh, mode)
    pad_w = _pad_width_1d(W, kw, mode)
    padded = np.pad(arr, (pad_h, pad_w), constant_values=0.0)

    Ph, Pw = padded.shape
    out_h  = Ph - kh + 1
    out_w  = Pw - kw + 1

    # Zero-copy view: shape (out_h, out_w, kh, kw)
    s0, s1  = padded.strides
    windows = as_strided(
        padded,
        shape   = (out_h, out_w, kh, kw),
        strides = (s0, s1, s0, s1),
    )

    # Flip kernel along both axes (convolution vs. cross-correlation)
    k_flipped = kernel[::-1, ::-1]

    # Dot product at every position — equivalent to (windows * k_flipped).sum((-2,-1))
    # but einsum avoids materialising the full broadcast product array.
    return np.einsum("ijkl,kl->ij", windows, k_flipped)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _pad_width_1d(n: int, k: int, mode: str) -> tuple[int, int]:
    """Return (before, after) padding for one dimension."""
    if mode == "valid":
        return (0, 0)
    if mode == "full":
        return (k - 1, k - 1)
    if mode == "same":
        total = max(k - 1, 0)
        before = total // 2
        return (before, total - before)
    raise ValueError(f"mode must be 'valid', 'same', or 'full'; got {mode!r}")


# ──────────────────────────────────────────────
# Radial kernel factory
# ──────────────────────────────────────────────

def radial_kernel(
    size: int,
    fn: callable,
    *,
    normalize: bool = False,
) -> np.ndarray:
    """Build a 2-D radial kernel of shape (size, size).

    Each cell (i, j) is set to fn(r), where r is the Euclidean distance from
    the centre of the kernel.  The centre has r = 0; corner cells of a size-n
    kernel have r = (size-1)/2 * sqrt(2).

    Parameters
    ----------
    size      : Odd integer giving the side length of the square kernel.
                Even sizes are accepted but the centre falls between pixels.
    fn        : Callable f(r: np.ndarray) -> np.ndarray.
                Receives a 2-D array of radii and must return a same-shaped
                array.  NumPy ufuncs (np.exp, np.cos, …), lambdas, and
                scipy functions all work directly.
    normalize : If True, divide the kernel by its sum so weights sum to 1.
                Useful for weighted-average convolutions (ignored by
                max_conv_2d, but handy when the kernel is used elsewhere).

    Returns
    -------
    np.ndarray of shape (size, size) and dtype float64.

    Examples
    --------
    >>> # Gaussian with σ = 1.0
    >>> σ = 1.0
    >>> g = radial_kernel(5, lambda r: np.exp(-r**2 / (2 * σ**2)), normalize=True)

    >>> # Hard disk: 1 inside radius, 0 outside
    >>> disk = radial_kernel(7, lambda r: (r <= 3).astype(float))

    >>> # Linear falloff
    >>> cone = radial_kernel(9, lambda r: np.maximum(0.0, 1.0 - r / 4.0))

    >>> # Ripple / Mexican-hat
    >>> ricker = radial_kernel(11, lambda r: (1 - r**2) * np.exp(-r**2 / 2))
    """
    if size < 1:
        raise ValueError(f"size must be >= 1; got {size}")

    # Coordinate grid centred at (0, 0)
    half = (size - 1) / 2.0
    ax = np.linspace(-half, half, size)
    x, y = np.meshgrid(ax, ax)
    r = np.hypot(x, y)          # Euclidean distance from centre

    kernel = np.asarray(fn(r), dtype=np.float64)
    if kernel.shape != (size, size):
        raise ValueError(
            f"fn must return an array of shape ({size}, {size}); "
            f"got {kernel.shape}"
        )

    if normalize:
        total = kernel.sum()
        if total == 0:
            raise ValueError("Cannot normalise: kernel sums to zero.")
        kernel = kernel / total

    return kernel


def fill_nearest(source: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fill masked pixels with the nearest non-zero value from source.

    For every pixel where mask is non-zero, the output copies the source
    value directly.  For every pixel where mask is zero, the output takes
    the value of the nearest (Euclidean) non-zero pixel in source.

    Parameters
    ----------
    source : 2-D array of values.  Non-zero pixels are the fill candidates.
    mask   : 2-D boolean or numeric array, same shape as source.
             Non-zero  → pixel is already known (copied from source as-is).
             Zero      → pixel needs to be filled from the nearest source.

    Returns
    -------
    np.ndarray of the same shape and dtype as source.

    Raises
    ------
    ValueError  if source and mask have different shapes, or if source
                contains no non-zero pixels (nothing to fill from).
    """
    source = np.asarray(source)
    mask   = np.asarray(mask)

    if source.shape != mask.shape:
        raise ValueError(
            f"source and mask must have the same shape; "
            f"got {source.shape} vs {mask.shape}"
        )
    if not np.any(source != 0):
        raise ValueError("source contains no non-zero pixels to fill from.")

    # distance_transform_edt treats zeros as "background" and non-zeros as
    # "foreground".  indices=True returns, for every background pixel, the
    # row/col of the nearest foreground pixel — exactly what we need.
    empty   = source == 0
    _, nearest_idx = distance_transform_edt(empty, return_indices=True)

    # Build output: start from source, then overwrite masked-off pixels
    # with the value of their nearest non-zero neighbour.
    out = source.copy()
    fill_rows, fill_cols = np.where(mask == 0)
    out[fill_rows, fill_cols] = source[
        nearest_idx[0][fill_rows, fill_cols],
        nearest_idx[1][fill_rows, fill_cols],
    ]
    return out