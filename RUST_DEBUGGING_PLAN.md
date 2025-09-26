# Rust STFT Implementation Debugging Plan

## Problem Analysis

Based on the comparison results and code analysis, the Rust ISTFT has significant reconstruction errors while the STFT forward transform is perfect. This indicates the issue is specifically in the ISTFT reconstruction logic.

## Critical Differences Identified

### 1. **MAJOR ISSUE: STFT Data Format Mismatch**
**Python**: `S[:, q_ - self.p_min]` - accesses column q_ (frequency bins for time slice q_)
**Rust**: `stft_data[stft_idx as usize]` - accesses row stft_idx (time slice)

**Problem**: The Rust code expects `stft_data` in format `[time_slices][frequency_bins]` but Python ISTFT expects `[frequency_bins][time_slices]`. This is a **fundamental data layout mismatch**.

### 2. **ISTFT Reconstruction Logic Differences**

#### Python (Line 412):
```python
xs = self._ifft_func(S[:, q_ - self.p_min]) * self.dual_win
```

#### Rust (Lines 574-580):
```rust
let xs = self.ifft_func(stft_data[stft_idx as usize].clone());
let windowed: Vec<Complex<f64>> = xs.iter()
    .zip(dual_win.iter())
    .map(|(x, w)| x * w)
    .collect();
```

**Problem**: Rust applies dual window AFTER IFFT, Python applies it as part of the same operation.

### 3. **Index Calculation Differences**

#### Python (Lines 400-401):
```python
q0 = (k0 // self.hop + self.p_min if k0 >= 0 else k0 // self.hop)
q1 = min(self.p_max(k1), q_max)
```

#### Rust (Lines 562-563):
```rust
let q0 = if k0 >= 0 { k0 / self.hop as i32 + self.p_min() } else { k0 / self.hop as i32 };
let q1 = (self.p_max(k1 as usize)).min(q_max);
```

**Issue**: Potential integer division differences and type conversion issues.

### 4. **Overlap-Add Implementation**

#### Python (Line 426):
```python
x[i0-k0:i1-k0] += xs[j0:j1].real if self.onesided_fft else xs[j0:j1]
```

#### Rust (Lines 601-605):
```rust
for (i, &val) in windowed[j0 as usize..j1 as usize].iter().enumerate() {
    if start_idx + i < x.len() {
        x[start_idx + i] += if self.onesided_fft() { val.re } else { val.re };
    }
}
```

**Problems**: 
- Rust always uses `val.re` even for complex case
- Different boundary checking logic

## Debugging Action Plan

### Phase 1: Fix Data Format Issue ⚠️ **CRITICAL**
1. **Verify STFT data layout** in test harness
2. **Fix ISTFT to expect correct format** - either transpose input or change access pattern
3. **Test with simple signal** to verify fix

### Phase 2: Fix Dual Window Application
1. **Match Python's dual window application** exactly
2. **Verify dual window calculation** matches Python
3. **Test reconstruction with known STFT data**

### Phase 3: Fix Overlap-Add Logic
1. **Fix complex number handling** in overlap-add
2. **Match Python's slice indexing** exactly
3. **Add bounds checking** that matches Python behavior

### Phase 4: Comprehensive Validation
1. **Test all signal types** with fixes
2. **Verify numerical precision** matches Python
3. **Performance benchmarking**

## Implementation Steps

### Step 1: Create Debug ISTFT Function
Create a debug version that:
- Prints intermediate values
- Matches Python logic exactly
- Uses the same variable names as Python

### Step 2: Fix Data Layout
Either:
- **Option A**: Transpose STFT data in Rust before ISTFT
- **Option B**: Change ISTFT to access data in Python format

### Step 3: Match Python Logic Exactly
- Use same index calculations
- Same dual window application
- Same overlap-add implementation

### Step 4: Add Comprehensive Tests
- Test each component individually
- Compare intermediate results with Python
- Validate edge cases

## Expected Outcome

After these fixes, the reconstruction error should drop from ~1e2-1e3 to ~1e-15, matching Python's performance.

## Priority Order

1. **🔥 CRITICAL**: Fix STFT data format mismatch (likely root cause)
2. **🔧 HIGH**: Fix dual window application timing
3. **🔧 HIGH**: Fix complex number handling in overlap-add
4. **📊 MEDIUM**: Optimize index calculations
5. **✅ LOW**: Performance optimizations

## Validation Strategy

For each fix:
1. Test with impulse signal (simplest case)
2. Test with sine wave (known frequency content)
3. Test with all signal types
4. Compare intermediate values with Python at each step
