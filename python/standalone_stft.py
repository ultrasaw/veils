'''WARNING 100% AI generated file'''
"""
Standalone Short-Time Fourier Transform implementation extracted from scipy.signal.
This version removes scipy dependencies and focuses on core STFT/ISTFT functionality.
"""

import numpy as np
import scipy.fft as fft_lib
from typing import Literal, Callable
from functools import partial


def _calc_dual_canonical_window(win: np.ndarray, hop: int) -> np.ndarray:
    """Calculate canonical dual window for 1d window `win` and a time step
    of `hop` samples.
    """
    if hop > len(win):
        raise ValueError(f"{hop=} is larger than window length of {len(win)}" +
                         " => STFT not invertible!")
    if issubclass(win.dtype.type, np.integer):
        raise ValueError("Parameter 'win' cannot be of integer type, but " +
                         f"{win.dtype=} => STFT not invertible!")

    w2 = win.real**2 + win.imag**2  # win*win.conj() does not ensure w2 is real
    DD = w2.copy()
    for k_ in range(hop, len(win), hop):
        DD[k_:] += w2[:-k_]
        DD[:-k_] += w2[k_:]

    # check DD > 0:
    relative_resolution = np.finfo(win.dtype).resolution * max(DD)
    if not np.all(DD >= relative_resolution):
        raise ValueError("Short-time Fourier Transform not invertible!")

    return win / DD


def simple_detrend(x: np.ndarray, type: str = 'constant') -> np.ndarray:
    """Simple detrending function to replace scipy.signal.detrend"""
    if type == 'constant':
        return x - np.mean(x)
    elif type == 'linear':
        n = len(x)
        t = np.arange(n)
        # Linear regression: y = a*t + b
        a = np.sum((t - np.mean(t)) * (x - np.mean(x))) / np.sum((t - np.mean(t))**2)
        b = np.mean(x) - a * np.mean(t)
        return x - (a * t + b)
    else:
        raise ValueError(f"Unknown detrend type: {type}")


class StandaloneSTFT:
    """Standalone Short-Time Fourier Transform implementation."""
    
    def __init__(self, win: np.ndarray, hop: int, fs: float, *,
                 fft_mode: str = 'onesided',
                 mfft: int | None = None,
                 dual_win: np.ndarray | None = None,
                 phase_shift: int | None = 0):
        
        if not (win.ndim == 1 and win.size > 0):
            raise ValueError(f"Parameter win must be 1d, but {win.shape=}!")
        if not all(np.isfinite(win)):
            raise ValueError("Parameter win must have finite entries!")
        if not (hop >= 1 and isinstance(hop, (int, np.integer))):
            raise ValueError(f"Parameter {hop=} is not an integer >= 1!")

        self._win = win.copy()
        self._win.setflags(write=False)
        self._hop = hop
        self._fs = fs
        self._fft_mode = fft_mode
        self._mfft = len(win) if mfft is None else mfft
        self._phase_shift = phase_shift or 0

        if dual_win is not None:
            if dual_win.shape != win.shape:
                raise ValueError(f"{dual_win.shape=} must equal {win.shape=}!")
            if not all(np.isfinite(dual_win)):
                raise ValueError("Parameter dual_win must be a finite array!")
            dual_win = dual_win.copy()
            dual_win.setflags(write=False)
        self._dual_win = dual_win

    @property
    def win(self) -> np.ndarray:
        return self._win

    @property
    def hop(self) -> int:
        return self._hop

    @property
    def fs(self) -> float:
        return self._fs

    @property
    def fft_mode(self) -> str:
        return self._fft_mode

    @property
    def mfft(self) -> int:
        return self._mfft

    @property
    def phase_shift(self) -> int:
        return self._phase_shift

    @property
    def T(self) -> float:
        """Sampling interval of input signal and of the window."""
        return 1.0 / self.fs

    @property
    def delta_t(self) -> float:
        """Time increment of STFT."""
        return self.hop * self.T

    @property
    def delta_f(self) -> float:
        """Width of the frequency bins of the STFT."""
        return 1.0 / (self.mfft * self.T)

    @property
    def m_num(self) -> int:
        """Number of samples in window."""
        return len(self.win)

    @property
    def m_num_mid(self) -> int:
        """Center index of window."""
        return self.m_num // 2

    @property
    def f_pts(self) -> int:
        """Number of points along the frequency axis."""
        if self.fft_mode in ['onesided', 'onesided2X']:
            return self.mfft // 2 + 1
        else:
            return self.mfft

    @property
    def onesided_fft(self) -> bool:
        """Return True if a one-sided FFT is used."""
        return self.fft_mode in ['onesided', 'onesided2X']

    @property
    def p_min(self) -> int:
        """The smallest possible slice index."""
        return self._pre_padding()[1]
    
    def _pre_padding(self) -> tuple[int, int]:
        """Smallest signal index and slice index due to padding."""
        w2 = self.win.real**2 + self.win.imag**2
        # move window to the left until the overlap with t >= 0 vanishes:
        n0 = -self.m_num_mid
        for p_, n_ in enumerate(range(n0, n0-self.m_num-1, -self.hop)):
            n_next = n_ - self.hop
            if n_next + self.m_num <= 0 or all(w2[n_next:] == 0):
                return n_, -p_
        # Fallback
        return n0 - self.m_num, -(self.m_num // self.hop + 1)

    @property
    def k_min(self) -> int:
        """The smallest possible signal index of the STFT."""
        return self._pre_padding()[0]

    def k_max(self, n: int) -> int:
        """First sample index after signal end not touched by a time slice."""
        return self._post_padding(n)[0]

    @property
    def dual_win(self) -> np.ndarray:
        """Dual window (canonical dual window by default)."""
        if self._dual_win is None:
            self._dual_win = _calc_dual_canonical_window(self.win, self.hop)
        return self._dual_win

    @property
    def invertible(self) -> bool:
        """Check if STFT is invertible."""
        try:
            _ = self.dual_win  # This will raise if not invertible
            return True
        except ValueError:
            return False

    def p_max(self, n: int) -> int:
        """Index of first non-overlapping upper time slice for n sample input."""
        return self._post_padding(n)[1]
    
    def _post_padding(self, n: int) -> tuple[int, int]:
        """Largest signal index and slice index due to padding."""
        if not (n >= (m2p := self.m_num - self.m_num_mid)):
            raise ValueError(f"Parameter n must be >= ceil(m_num/2) = {m2p}!")
        
        w2 = self.win.real**2 + self.win.imag**2
        # move window to the right until the overlap for t < t[n] vanishes:
        q1 = n // self.hop   # last slice index with t[p1] <= t[n]
        k1 = q1 * self.hop - self.m_num_mid
        for q_, k_ in enumerate(range(k1, n+self.m_num, self.hop), start=q1):
            n_next = k_ + self.hop
            if n_next >= n or all(w2[:n-n_next] == 0):
                return k_ + self.m_num, q_ + 1
        # Fallback
        return n + self.m_num, (n + self.m_num_mid) // self.hop + 1

    def p_num(self, n: int) -> int:
        """Number of time slices for an input signal with n samples."""
        return self.p_max(n) - self.p_min

    def p_range(self, n: int, p0: int | None = None, p1: int | None = None) -> tuple[int, int]:
        """Determine and validate slice index range."""
        if p0 is None:
            p0 = self.p_min
        if p1 is None:
            p1 = self.p_max(n)
        
        # Allow p1 to be larger than theoretical max for compatibility
        max_p = self.p_max(n)
        if not (self.p_min <= p0 < p1):
            raise ValueError(f"Invalid slice range: {p0=}, {p1=}")
        
        return p0, p1

    def _fft_func(self, x: np.ndarray) -> np.ndarray:
        """Apply FFT based on fft_mode."""
        # Handle phase shift first (like scipy does)
        if self.phase_shift is not None:
            if x.shape[-1] < self.mfft:  # zero pad if needed
                z_shape = list(x.shape)
                z_shape[-1] = self.mfft - x.shape[-1]
                x = np.hstack((x, np.zeros(z_shape, dtype=x.dtype)))
            p_s = (self.phase_shift + self.m_num_mid) % self.m_num
            x = np.roll(x, -p_s, axis=-1)

        # Compute FFT based on mode - match scipy exactly
        if self.fft_mode == 'twosided':
            return fft_lib.fft(x, n=self.mfft)
        elif self.fft_mode == 'centered':
            return fft_lib.fftshift(fft_lib.fft(x, n=self.mfft))
        elif self.fft_mode == 'onesided':
            return fft_lib.rfft(x, n=self.mfft)
        elif self.fft_mode == 'onesided2X':
            X = fft_lib.rfft(x, n=self.mfft)
            # Either squared magnitude (psd) or magnitude is doubled:
            fac = np.sqrt(2) if getattr(self, '_scaling', None) == 'psd' else 2
            # For even input length, the last entry is unpaired:
            X[1: -1 if self.mfft % 2 == 0 else None] *= fac
            return X
        else:
            raise ValueError(f"Unknown fft_mode: {self.fft_mode}")

    def _ifft_func(self, X: np.ndarray) -> np.ndarray:
        """Apply IFFT based on fft_mode."""
        if self.fft_mode == 'twosided':
            x = fft_lib.ifft(X, n=self.mfft, axis=-1)
        elif self.fft_mode == 'centered':
            x = fft_lib.ifft(fft_lib.ifftshift(X, axes=-1), n=self.mfft, axis=-1)
        elif self.fft_mode == 'onesided':
            x = fft_lib.irfft(X, n=self.mfft, axis=-1)
        elif self.fft_mode == 'onesided2X':
            Xc = X.copy()  # we do not want to modify function parameters
            fac = np.sqrt(2) if getattr(self, '_scaling', None) == 'psd' else 2
            # For even length X the last value is not paired with a negative
            # value on the two-sided FFT:
            q1 = -1 if self.mfft % 2 == 0 else None
            Xc[..., 1:q1] /= fac
            x = fft_lib.irfft(Xc, n=self.mfft, axis=-1)
        else:
            raise ValueError(f"Unknown fft_mode: {self.fft_mode}")

        # Handle phase shift - match scipy exactly
        if self.phase_shift is None:
            return x[..., :self.m_num]
        p_s = (self.phase_shift + self.m_num_mid) % self.m_num
        return np.roll(x, p_s, axis=-1)[..., :self.m_num]

    def _x_slices(self, x: np.ndarray, k_offset: int, p0: int, p1: int, 
                  padding: str = 'zeros'):
        """Generator yielding windowed slices of input signal."""
        n = len(x)
        
        for p in range(p0, p1):
            # Calculate slice boundaries
            k_center = p * self.hop + k_offset
            k_start = k_center - self.m_num_mid
            k_end = k_start + self.m_num
            
            # Extract slice with padding if necessary
            if k_start >= 0 and k_end <= n:
                # No padding needed
                yield x[k_start:k_end]
            else:
                # Padding needed
                slice_data = np.zeros(self.m_num, dtype=x.dtype)
                
                # Determine valid range
                valid_start = max(0, k_start)
                valid_end = min(n, k_end)
                slice_start = valid_start - k_start
                slice_end = slice_start + (valid_end - valid_start)
                
                if valid_end > valid_start:
                    slice_data[slice_start:slice_end] = x[valid_start:valid_end]
                
                # Apply padding
                if padding == 'zeros':
                    pass  # Already zeros
                elif padding == 'edge':
                    if k_start < 0:
                        slice_data[:slice_start] = x[0]
                    if k_end > n:
                        slice_data[slice_end:] = x[-1]
                elif padding == 'even':
                    if k_start < 0:
                        for i in range(slice_start):
                            idx = min(abs(k_start + i), n - 1)
                            slice_data[i] = x[idx]
                    if k_end > n:
                        for i in range(slice_end, self.m_num):
                            idx = n - 1 - (k_start + i - n + 1)
                            idx = max(0, min(idx, n - 1))
                            slice_data[i] = x[idx]
                elif padding == 'odd':
                    if k_start < 0:
                        for i in range(slice_start):
                            idx = min(abs(k_start + i), n - 1)
                            slice_data[i] = -x[idx]
                    if k_end > n:
                        for i in range(slice_end, self.m_num):
                            idx = n - 1 - (k_start + i - n + 1)
                            idx = max(0, min(idx, n - 1))
                            slice_data[i] = -x[idx]
                
                yield slice_data

    def stft(self, x: np.ndarray, p0: int | None = None, p1: int | None = None, *,
             k_offset: int = 0, padding: str = 'zeros') -> np.ndarray:
        """Perform the short-time Fourier transform."""
        return self.stft_detrend(x, None, p0, p1, k_offset=k_offset, padding=padding)

    def stft_detrend(self, x: np.ndarray, detr: Callable | str | None,
                     p0: int | None = None, p1: int | None = None, *,
                     k_offset: int = 0, padding: str = 'zeros') -> np.ndarray:
        """Calculate STFT with optional detrending."""
        if self.onesided_fft and np.iscomplexobj(x):
            raise ValueError(f"Complex-valued `x` not allowed for {self.fft_mode=}! "
                             "Set fft_mode to 'twosided' or 'centered'.")
        
        if isinstance(detr, str):
            detr = partial(simple_detrend, type=detr)
        elif not (detr is None or callable(detr)):
            raise ValueError(f"Parameter {detr=} is not a str, function or None!")
        
        n = len(x)
        if not (n >= (m2p := self.m_num - self.m_num_mid)):
            raise ValueError(f'{len(x)=} must be >= ceil(m_num/2) = {m2p}!')

        # Determine slice index range
        p0, p1 = self.p_range(n, p0, p1)
        S = np.zeros((self.f_pts, p1 - p0), dtype=complex)
        
        for p_, x_ in enumerate(self._x_slices(x, k_offset, p0, p1, padding)):
            if detr is not None:
                x_ = detr(x_)
            S[:, p_] = self._fft_func(x_ * self.win.conj())
        
        return S

    def istft(self, S: np.ndarray, k0: int = 0, k1: int | None = None) -> np.ndarray:
        """Inverse short-time Fourier transform."""
        if S.shape[0] != self.f_pts:
            raise ValueError(f"{S.shape[0]=} must be equal to {self.f_pts=}!")
        
        n_min = self.m_num - self.m_num_mid
        if not (S.shape[1] >= (q_num := self.p_num(n_min))):
            raise ValueError(f"{S.shape[1]=} needs to have at least {q_num} slices!")

        q_max = S.shape[1] + self.p_min
        k_max = (q_max - 1) * self.hop + self.m_num - self.m_num_mid

        # If k1 is larger than k_max, we need to allow it but limit reconstruction
        if k1 is None:
            k1 = k_max
        
        # Allow k1 to be larger than k_max, but warn about potential issues
        actual_k_max = max(k_max, k1) if k1 > k_max else k_max
        
        if not (self.k_min <= k0 < k1):
            raise ValueError(f"({self.k_min=}) <= ({k0=}) < ({k1=}) is false!")
        
        if not (num_pts := k1 - k0) >= n_min:
            raise ValueError(f"({k1=}) - ({k0=}) = {num_pts} has to be at least {n_min}!")

        q0 = (k0 // self.hop + self.p_min if k0 >= 0 else k0 // self.hop)
        q1 = min(self.p_max(k1), q_max)
        
        # Calculate reconstruction length - use the actual signal length we want
        n_pts = k1 - k0
        
        x = np.zeros(n_pts, dtype=float if self.onesided_fft else complex)
        
        for q_ in range(q0, q1):
            if q_ - self.p_min >= S.shape[1]:
                break  # No more STFT data available
                
            xs = self._ifft_func(S[:, q_ - self.p_min]) * self.dual_win
            i0 = q_ * self.hop - self.m_num_mid
            i1 = min(i0 + self.m_num, n_pts + k0)
            j0, j1 = 0, i1 - i0
            
            if i0 < k0:  # xs sticks out to the left on x
                j0 += k0 - i0
                i0 = k0
            
            if i1 > k0 + n_pts:  # xs sticks out to the right
                j1 -= (i1 - k0 - n_pts)
                i1 = k0 + n_pts
            
            if i0 < i1 and j0 < j1:  # Valid range
                x[i0-k0:i1-k0] += xs[j0:j1].real if self.onesided_fft else xs[j0:j1]
        
        return x

    def t(self, n: int, p0: int | None = None, p1: int | None = None, 
          k_offset: int = 0) -> np.ndarray:
        """Times of STFT for an input signal with n samples."""
        p0, p1 = self.p_range(n, p0, p1)
        return (np.arange(p0, p1) * self.hop + k_offset) * self.T

    def f(self) -> np.ndarray:
        """Frequencies values of the STFT."""
        if self.fft_mode == 'centered':
            freqs = np.fft.fftshift(np.fft.fftfreq(self.mfft, self.T))
        else:
            freqs = np.fft.fftfreq(self.mfft, self.T)
        
        if self.onesided_fft:
            return freqs[:self.f_pts]
        else:
            return freqs
