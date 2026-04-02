# Naga Optimization Plan for GPU Compute Performance

## Context

meganeura's conv2d GEMM kernels are 3.3× slower than PyTorch/MIOpen on the same AMD 890M hardware. After exhausting algorithmic optimizations (implicit GEMM, Winograd, tile selection), the remaining gap is in per-dispatch shader efficiency. This plan identifies concrete naga changes to close the gap.

Target backends: **SPIR-V** (Vulkan/RADV) and **MSL** (Metal).

## 1. Vectorized Storage Buffer Loads

**Impact: ~30-40% of the gap**

### Problem

Every `array<f32>` load in WGSL compiles to a scalar `OpLoad` in SPIR-V. Our conv GEMM shaders load 1024 floats per shared memory tile (4 loads per thread × 256 threads). Each load is an independent scalar memory transaction.

MIOpen uses `buffer_load_dwordx4` (128-bit) — one instruction loads 4 consecutive f32s.

### Current IR representation

```
Expression::Load { pointer }  // always scalar
```

The `pointer` comes from `Access { base: GlobalVariable(array<f32>), index: computed }`. There's no way to express "load 4 consecutive elements as vec4" at the WGSL or IR level.

### Proposed change

**Option A (WGSL-level):** Add a new builtin or syntax for vectorized array loads:
```wgsl
// New: load 4 consecutive f32s as vec4
let v: vec4<f32> = arrayLoad4(&src[base]);
```

This would lower to a new IR expression or to an `Access` + `Load` with vector type, then emit `OpLoad` of `vec4<f32>` in SPIR-V.

**Option B (IR-level optimization pass):** Detect consecutive scalar loads from the same array with sequential indices, and merge them into a single vector load. This is a `proc/` pass:

```
// Before (4 expressions):
Load { Access { base: arr, index: i } }
Load { Access { base: arr, index: i+1 } }
Load { Access { base: arr, index: i+2 } }
Load { Access { base: arr, index: i+3 } }

// After (1 expression + 4 extracts):
Load { Access { base: arr_as_vec4, index: i/4 } }
CompositeExtract { ... }
```

**Option C (Backend peephole):** Let the SPIR-V backend detect the pattern during emission and coalesce loads. Less disruptive to IR.

### Files to change

- IR: `naga/src/ir/mod.rs` — potentially add `VectorLoad` expression or modify `Load` semantics
- WGSL frontend: `naga/src/front/wgsl/lower/mod.rs` — parse new syntax
- SPIR-V backend: `naga/src/back/spv/block.rs` — emit vector `OpLoad`
- MSL backend: `naga/src/back/msl/writer.rs` — emit `device float4*` cast + load
- Validation: `naga/src/valid/expression.rs` — validate new expression

### Measurement

Before: each conv2d small-tile dispatch loads 2 × 512 floats from global → shared. At scalar: 1024 load instructions.
After: 256 vec4 load instructions. Expected ~2× faster staging loop.

## 2. Integer Division Strength Reduction

**Impact: ~10-15% of the gap**

### Problem

Every im2col index decomposition does 3-6 unsigned integer divisions:
```wgsl
let ci = k_idx / kernel_hw;     // uniform divisor
let kh = k_rem / params.kernel_w; // uniform divisor
let oh = hw_idx / params.out_w;   // uniform divisor
```

In SPIR-V, the backend emits plain `OpUDiv`. The Vulkan driver JIT *may* optimize division by uniform values, but this is not guaranteed.

In the MSL backend, unsigned division goes through `naga_div()` which adds a zero-check (`metal::select(rhs, 1u, rhs == 0u)`) — an extra comparison + select per division.

### Current code

**SPIR-V** (`naga/src/back/spv/writer.rs:554-564`): Integer divide/modulo always go through `write_wrapped_binary_op()` which generates a helper function with:
- Zero-divisor check (`OpIEqual` + `OpSelect`)
- For signed: also MIN/-1 overflow check
- For unsigned: still wraps with zero check

**MSL** (`naga/src/back/msl/writer.rs:2223-2230`): `naga_div(lhs, rhs)` function with `metal::select()` guard.

### Proposed change

**Phase 1: Skip wrapping for unsigned division** when the WGSL source uses `u32` types. WGSL spec says unsigned division by zero returns 0 (defined behavior), so the wrap is unnecessary if backends can emit a simple `OpUDiv` that drivers handle correctly. The SPIR-V spec says `OpUDiv` result is undefined for div-by-zero, but in practice all GPU drivers return 0.

Add a `WriterFlags::UNSAFE_UINT_DIV` flag (or per-capability check) to skip the wrapper for unsigned integer division.

**Phase 2: Multiply-by-magic-number** for division by uniform constants. When the divisor is loaded from a uniform buffer (detectable from `AddressSpace::Uniform`), emit the multiply-shift sequence:

```
// a / d where d is uniform → a * magic >> shift
let magic = compute_magic_number(d);
result = (a * magic) >> shift;
```

This is a new IR-level optimization pass in `naga/src/proc/strength_reduce.rs`:
- Walk expressions looking for `Binary { op: Divide, right: Load { pointer_to_uniform } }`
- Replace with multiply + shift sequence
- Requires knowledge that the uniform value doesn't change between invocations (true for compute shaders within a dispatch)

### Files to change

- SPIR-V backend: `naga/src/back/spv/writer.rs` — add flag to skip zero-check for uint
- MSL backend: `naga/src/back/msl/writer.rs` — same for `naga_div`
- New pass: `naga/src/proc/strength_reduce.rs` — uniform division optimization
- Proc mod: `naga/src/proc/mod.rs` — register new pass

### Measurement

6 divisions per im2col element × 1024 elements per tile load × 256 K-iterations for grad_weight.
Current: ~20 instructions per division (wrapped). After: 1-3 instructions.

## 3. Cooperative Matrix from Storage Buffers

**Impact: potentially ~2× if it works on the driver**

### Problem

Our coop matmul shader stages data to shared memory, then `coopLoadT` from shared. This adds a staging loop + barrier that a direct load from storage would avoid. Blade's matmul example loads directly from storage:

```wgsl
let a = coopLoadT<coop_mat16x16<f16,A>>(&matrix_a[offset], stride);
```

### Current IR representation

`CooperativeLoad` takes a `CooperativeData { pointer, stride, row_major }` where `pointer` can point to any address space. The IR already supports this.

### Issue

The SPIR-V backend at `block.rs:2127-2170` correctly emits `OpCooperativeMatrixLoadKHR` regardless of address space. The question is whether our WGSL frontend allows `coopLoadT` from `var<storage>` arrays.

**Check:** In `front/wgsl/lower/mod.rs:3072-3115`, the `coopLoad`/`coopLoadT` handling extracts a pointer from the first argument. If this pointer is from a storage buffer, it should work.

### Proposed investigation

1. Write a test shader that does `coopLoadT` directly from `var<storage>` (no shared memory staging)
2. Check if naga compiles it correctly
3. If it works in naga but not in RADV: file a driver bug
4. If naga rejects it: fix the frontend/validation to allow it

### Files to check

- Frontend: `naga/src/front/wgsl/lower/mod.rs:3072-3115` — does it accept storage pointers?
- Validation: `naga/src/valid/expression.rs` — does it validate coop loads from storage?
- SPIR-V backend: `naga/src/back/spv/block.rs:2127-2170` — does it emit correct address space?

## 4. Subgroup Broadcast for Inner Loop

**Impact: ~10% for K-tile inner loop**

### Problem

The GEMM inner loop reads from shared memory with each thread loading the same A-row value:
```wgsl
for (var kk = 0u; kk < 16u; kk++) {
    let a0 = shared_a[(ty * 4u + 0u) * 16u + kk];  // same for all tx
    ...
}
```

All threads in the same row (`ty`) read the same `a0` value. This creates shared memory bank contention. Instead, one thread could read and broadcast via `subgroupBroadcast`.

### Current IR support

Naga already supports subgroup operations:
- `Statement::SubgroupGather { mode: GatherMode::Broadcast { index }, argument, result }`
- SPIR-V: emits `OpGroupNonUniformBroadcast` (`naga/src/back/spv/subgroup.rs`)
- MSL: emits `simd_broadcast`

### What's needed

This is a WGSL-level change in our shaders, not a naga change. Naga already supports `subgroupBroadcast()`. However, we need to verify:

1. WGSL `subgroupBroadcast(value, lane_id)` is parsed correctly
2. The SPIR-V output uses `OpGroupNonUniformBroadcast` with `Scope::Subgroup`
3. The MSL output uses `simd_broadcast`

### Files to verify

- Frontend: `naga/src/front/wgsl/lower/mod.rs` — search for subgroup gather lowering
- SPIR-V: `naga/src/back/spv/subgroup.rs` — verify broadcast emission
- MSL: `naga/src/back/msl/writer.rs` — verify simd_broadcast

## 5. Loop Unrolling Hints

**Impact: ~5% for outer K-loop**

### Problem

The outer K-loop in conv GEMM:
```wgsl
var t = 0u;
loop {
    if t >= k_total { break; }
    ...
    t += 16u;
}
```

This compiles to a SPIR-V `OpLoopMerge` + `OpBranch` loop. The driver JIT may or may not unroll it. A hint would help.

### Proposed change

Add `@unroll` or `@unroll(N)` attribute support in WGSL that lowers to SPIR-V `OpLoopMerge` with `Unroll` control:

```wgsl
@unroll loop { ... }  // → OpLoopMerge with LoopControl::Unroll
```

### Files to change

- Frontend: `naga/src/front/wgsl/parse/` — parse `@unroll` attribute
- IR: attribute on `Statement::Loop` (add `unroll: bool` field)
- SPIR-V: `naga/src/back/spv/block.rs` — emit `LoopControl::Unroll` in `OpLoopMerge`
- MSL: `naga/src/back/msl/writer.rs` — emit `#pragma unroll` or `[[unroll]]`

## Priority Order

| # | Feature | Impact | Effort | Dependencies |
|---|---------|--------|--------|--------------|
| 1 | Unsafe uint division (skip zero-check) | 10-15% | Small | None |
| 2 | Vec4 storage loads | 30-40% | Medium | IR change |
| 3 | Coop matrix from storage | 2× potential | Investigation | Driver support |
| 4 | Subgroup broadcast | 10% | Small (shader change) | Already in naga |
| 5 | Loop unroll hints | 5% | Small | None |
| 6 | Uniform division strength reduction | 10-15% | Large | Analysis pass |

## Appendix: Key Naga Source Locations

| Component | Path | Key lines |
|-----------|------|-----------|
| IR types | `naga/src/ir/mod.rs` | Expression (L1617), Statement (L2050), TypeInner (L867) |
| WGSL lowering | `naga/src/front/wgsl/lower/mod.rs` | Coop matrix (L3072), binary ops |
| SPIR-V int div | `naga/src/back/spv/writer.rs` | write_wrapped_binary_op (L642) |
| SPIR-V coop | `naga/src/back/spv/block.rs` | CooperativeLoad (L2127), Store (L4144) |
| MSL int div | `naga/src/back/msl/writer.rs` | naga_div (L5833), Division (L2223) |
| MSL coop | `naga/src/back/msl/writer.rs` | NagaCooperativeLoad (L6456) |
| Subgroup ops | `naga/src/back/spv/subgroup.rs` | Broadcast, shuffle, reduce |
| Processing | `naga/src/proc/` | No optimization passes exist yet |
