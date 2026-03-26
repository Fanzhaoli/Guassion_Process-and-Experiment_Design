# Comparison: `S2_gen_data_optimized_cp_v2` to `v8`

## Purpose

This document summarizes the differences across the following notebooks so the next round of model optimization can be done in a controlled way:

- `S2_gen_data_optimized_cp_v2.ipynb`
- `S2_gen_data_optimized_cp_v3.ipynb`
- `S2_gen_data_optimized_cp_v4.ipynb`
- `S2_gen_data_optimized_cp_v5.ipynb`
- `S2_gen_data_optimized_cp_v6.ipynb`
- `S2_gen_data_optimized_cp_v7.ipynb`
- `S2_gen_data_optimized_cp_v8_additive.ipynb`

The comparison focuses on the DDM-related components:

- drift rate `v`
- boundary separation `a`
- non-decision time `t0`
- starting point `z`
- trial generation and validity handling
- condition-specific self vs stranger manipulation

---

## 1. High-level evolution

These versions fall into two major stages.

### Stage A: Legacy Study2-style generator (`v2-v4`)

These versions preserve the original `Study2`-style logic:

- `v` is driven by sigmoid transforms of `T` and `P`
- `a` is driven by a sigmoid transform of `M = T + W`
- `t0` is fixed at `0.2`
- `z` is implicitly set at the midpoint: `z = a / 2`
- self vs stranger is implemented through multiplicative changes on `v`

This stage is close to the original simulation logic, but it tends to generate unrealistic left-tail RTs.

### Stage B: Bounded and human-like RT regime (`v5-v8`)

These versions redesign the parameterization to better match plausible human data:

- `v` is explicitly bounded to `0.05-0.40`
- `a` is explicitly bounded to `0.10-0.30`
- `t0` becomes a subject-level distribution instead of a fixed constant
- `RT < 0.25s` is explicitly marked as anticipatory / invalid
- self vs stranger is still implemented through `v`, but the exact form changes across versions

This stage improves RT realism, especially the left tail, but also makes it easier for self/stranger differences to be attenuated.

---

## 2. Version-by-version summary

| Version | Main role | Core change |
|---|---|---|
| `v2` | Python-runnable adaptation of optimized Study2 generator | Keeps original sigmoid-based DDM parameter logic |
| `v3` | Check-oriented runnable notebook | Same model as `v2`, plus diagnostics and more stable workflow |
| `v4` | First left-tail fix attempt | Adds lower bound to `a` to avoid very small boundaries |
| `v5` | Major redesign | Introduces bounded `v/a`, variable `t0`, fast-RT filtering |
| `v6` | Stronger fixed self advantage | Keeps `v5` structure but increases additive self/stranger difference in `v` |
| `v7` | Return to multiplicative condition effect | Keeps `v5` structure but uses multiplicative `ALPHA1/ALPHA2` on `v` |
| `v8` | Stronger additive condition effect | Keeps `v5` structure but uses larger additive shifts on `v` |

---

## 3. Detailed comparison of `v`, `a`, `t0`, `z`

## 3.1 Drift rate `v`

### `v2-v4`: original sigmoid-based formulation

For `v2`, `v3`, and `v4`, drift rate is built from two sigmoid components.

#### Step 1: time-based component

```python
v_T = 1 / (1 + exp(-0.01 * (T - 100)))
```

Interpretation:

- larger `T` increases drift
- effect is nonlinear and saturating

#### Step 2: practice-based component

First define a practice-dependent slope:

```python
k(P) = 0.01 + (0.15 - 0.01) / (1 + exp(-0.1 * (P - 32)))
```

Then compute:

```python
v_P = 1 / (1 + exp(-k(P) * (P - 4)))
```

Interpretation:

- larger `P` increases drift
- the sensitivity of `P` itself changes with practice level

#### Step 3: baseline drift

```python
v_0 = 3 * v_T * v_P
```

#### Step 4: self vs stranger manipulation

```python
if self:
    v = v_0 * (1 + ALPHA1)
else:
    v = v_0 * (1 + ALPHA2)
```

with:

- `ALPHA1 = 1.5`
- `ALPHA2 = -0.4`

So effectively:

- self: `v = 2.5 * v_0`
- stranger: `v = 0.6 * v_0`

#### Step 5: trial-level noise

Each trial further samples:

```python
v_trial ~ Normal(v_center, 1)
```

This is a very large amount of trial-level noise.

### `v5-v8`: bounded ease-based formulation

From `v5` onward, `v` no longer comes from the original sigmoid formulas.
Instead, `P/T/W` are first normalized and combined into an overall task-ease score.

#### Step 1: normalize `P/T/W`

```python
Pn = clip((P - 0) / 149, 0, 1)
Tn = clip((T - 10) / 589, 0, 1)
Wn = clip((W - 200) / 1299, 0, 1)
```

#### Step 2: compute task ease

```python
ease = 0.35 * Pn + 0.40 * Tn + 0.25 * Wn
```

Interpretation:

- `T` is weighted most strongly
- `P` is second
- `W` is third
- ease is explicitly restricted to `[0, 1]`

#### Step 3: compute bounded baseline drift

Piecewise mapping:

```python
if ease < 0.5:
    v_base = 0.05 + (0.20 - 0.05) * (ease / 0.5)
else:
    v_base = 0.20 + (0.40 - 0.20) * ((ease - 0.5) / 0.5)
```

Then:

```python
v_base = clip(v_base, 0.05, 0.40)
```

Interpretation:

- difficult tasks map to `v = 0.05-0.20`
- easier tasks map to `v = 0.20-0.40`
- this is much closer to typical DDM scale conventions

#### Version-specific self/stranger manipulations

##### `v5`: additive, weak

```python
self:     v = v_base + 0.03
stranger: v = v_base - 0.02
```

Then clipped to `[0.05, 0.40]`.

##### `v6`: additive, stronger

```python
self:     v = v_base + 0.08
stranger: v = v_base - 0.05
```

Then clipped to `[0.05, 0.40]`.

##### `v7`: multiplicative

```python
self:     v = v_base * (1 + 0.60)
stranger: v = v_base * (1 - 0.25)
```

Then clipped to `[0.05, 0.40]`.

##### `v8`: additive, strongest among bounded versions

```python
self:     v = v_base + 0.15
stranger: v = v_base - 0.10
```

Then clipped to `[0.05, 0.40]`.

Note: `v8` still contains leftover unreachable multiplicative code after the additive `return`, but the actual active logic is additive.

#### Trial-level noise in `v5-v8`

Each trial samples:

```python
v_trial ~ Normal(v_center, 0.04)
```

Then clips again to `[0.05, 0.40]`.

This is far smaller and more controlled than the `sd=1` used in `v2-v4`.

---

## 3.2 Boundary separation `a`

### `v2-v3`: original sigmoid-based boundary

Boundary depends only on:

```python
M = T + W
```

Then:

```python
a_0 = 3 / (1 + exp(-0.01 * (M - 600)))
```

Condition on `M`:

```python
if M > 600:
    a = a_0 * (1 + BETA1)
else:
    a = a_0 * (1 + BETA2)
```

with:

- `BETA1 = 0.2`
- `BETA2 = 0`

So:

- for larger `M`, boundary is boosted
- for smaller `M`, it can become extremely small

Subject-level variability:

```python
a_subject ~ Normal(a_base, a_base * 0.15)
```

In optimized vectorized form, this is implemented as multiplicative noise:

```python
a = a_base * Normal(1, 0.15)
```

and then bounded only by:

```python
a = max(a, 0.01)
```

Interpretation:

- `a` can be extremely small in low-`M` regions
- this is the main reason these versions produced many very fast RTs

### `v4`: same boundary plus hard lower bound

`v4` keeps the same sigmoid logic but adds:

```python
a = max(a_1, 0.3)
```

So `a` can no longer drop below `0.3`.

This was a direct attempt to reduce unrealistic early boundary crossings.

### `v5-v8`: bounded ease/window-based boundary

From `v5` onward, boundary is no longer driven by the original sigmoid in `M`.
Instead:

```python
a = 0.12 + 0.10 * Wn + 0.05 * (1 - ease)
```

Then:

```python
a = clip(a, 0.10, 0.30)
```

Interpretation:

- larger `W` allows slightly larger boundaries
- harder tasks also have slightly larger boundaries
- overall `a` is explicitly constrained to a realistic range

Subject-level variability:

```python
a_subject ~ Normal(a_base, a_base * 0.08)
```

Then clipped to `[0.10, 0.30]`.

Compared to `v2-v4`, this makes `a` much more stable and prevents the extremely small values that created the left-tail problem.

---

## 3.3 Non-decision time `t0`

### `v2-v4`: fixed non-decision time

```python
t0 = 0.2
```

This is constant for all subjects and all trials.

Interpretation:

- easy to implement
- but tends to produce visible pile-ups just above `0.20s`
- unrealistic if the goal is to approximate human RT distributions

### `v5-v8`: subject-level distributed non-decision time

Each subject samples:

```python
t0_subject ~ Normal(0.28, 0.02)
```

Then clipped to:

```python
t0_subject in [0.24, 0.35]
```

Interpretation:

- preserves a plausible central tendency near `280 ms`
- reduces artificial pile-up at a single absolute left-tail point
- better matches between-subject differences in encoding/motor time

This is one of the most important reasons why `v5-v8` achieved a more realistic RT left tail.

---

## 3.4 Starting point `z`

Across all versions examined, `z` is not an independently manipulated parameter.
It is implicitly fixed at the midpoint between the two boundaries.

In simulation:

```python
evidence = a / 2 + cumsum(drift + noise)
```

So effectively:

```python
z = a / 2
```

Interpretation:

- no start-point bias is modeled
- all self vs stranger effects are assumed to arise from `v`, not `z`
- no version from `v2` to `v8` introduces a condition-specific or PTW-specific `z`

This is important for future optimization: if you later want to test attention or prior-bias mechanisms more directly, `z` is currently untouched and remains available as an independent modeling dimension.

---

## 4. Trial simulation and validity handling

## 4.1 Shared simulation backbone

Across versions, the trial simulation itself is broadly the same.

Evidence accumulation is simulated as:

```python
evidence_t = z + cumulative_sum(v * dt + noise_t)
```

where:

- `dt = 0.001`
- noise variance scales with `sqrt(dt)`
- upper boundary is `a`
- lower boundary is `0`
- the first crossing determines response and RT

Thus, the basic DDM-like simulation engine is stable across versions.

## 4.2 Validity logic in `v2-v4`

These versions mainly keep a trial if:

- a boundary was crossed
- RT is below the maximum allowed time window

There is no explicit fast-RT exclusion rule.

This means left-tail artifacts are preserved in the final retained data.

## 4.3 Validity logic in `v5-v8`

These versions add explicit validity flags:

- `is_timeout`
- `is_fast_rt`
- `is_valid`

with:

```python
is_fast_rt = RT < 0.25
is_valid = (not timeout) and (not fast_rt) and (RT < max_time)
```

Interpretation:

- anticipatory responses are explicitly identified
- the final analysis dataset is cleaner and more human-like
- some condition effects may also be attenuated because extreme early responses are removed

---

## 5. Parameter table

## 5.1 Core parameter settings by version

| Parameter | `v2` | `v3` | `v4` | `v5` | `v6` | `v7` | `v8` |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ALPHA1` | 1.5 | 1.5 | 1.5 | - | - | 0.60 | 0.60 leftover only |
| `ALPHA2` | -0.4 | -0.4 | -0.4 | - | - | -0.25 | -0.25 leftover only |
| `SELF_V_BONUS` | - | - | - | 0.03 | 0.08 | - | - |
| `STRANGER_V_PENALTY` | - | - | - | 0.02 | 0.05 | - | - |
| `V_ADD_SELF` | - | - | - | - | - | - | 0.15 default |
| `V_ADD_STRANGER` | - | - | - | - | - | - | -0.10 default |
| `A_CV` | 0.15 | 0.15 | 0.15 | 0.08 | 0.08 | 0.08 | 0.08 |
| `TRIAL_V_SD` | effectively 1 | effectively 1 | effectively 1 | 0.04 | 0.04 | 0.04 | 0.04 |
| `t0` mean | 0.20 fixed | 0.20 fixed | 0.20 fixed | 0.28 | 0.28 | 0.28 | 0.28 |
| `t0` SD | 0 | 0 | 0 | 0.02 | 0.02 | 0.02 | 0.02 |
| `t0` range | fixed | fixed | fixed | 0.24-0.35 | 0.24-0.35 | 0.24-0.35 | 0.24-0.35 |
| `a` lower bound | 0.01 numeric floor | 0.01 numeric floor | 0.30 | 0.10 | 0.10 | 0.10 | 0.10 |
| `a` upper bound | none practical | none practical | none practical | 0.30 | 0.30 | 0.30 | 0.30 |
| `MIN_VALID_RT` | none | none | none | 0.25 | 0.25 | 0.25 | 0.25 |

---

## 6. Interpretation of major transitions

## 6.1 `v2-v4` to `v5`: the most important change

This is the real turning point.

The old regime (`v2-v4`) had:

- strong nonlinear drift amplification
- potentially tiny boundaries
- fixed `t0`
- large trial-level drift noise
- no fast-RT cleaning

This combination naturally produced too many very early RTs.

The new regime (`v5-v8`) addresses exactly that by:

- bounding `v`
- bounding `a`
- adding subject-level `t0` variability
- reducing trial-level `v` noise
- explicitly excluding `RT < 0.25s`

So if the goal is realistic RT distributions, `v5-v8` is the better base.

## 6.2 `v5-v8`: the remaining challenge

Once RT realism improved, another issue appeared:

- self vs stranger differences became much weaker behaviorally

This makes theoretical sense because `v5-v8` removed several mechanisms that had previously exaggerated condition differences:

- extreme low `a`
- fixed `t0`
- very large trial-level `v` dispersion

As a result, the remaining self/stranger effect must now come mostly from the explicit condition modifier on `v`.

This is why `v6-v8` mainly differ in how they manipulate `v`:

- weak additive
- stronger additive
- multiplicative
- strongest additive

---

## 7. Practical summary for future model optimization

## 7.1 What stayed constant across all versions

These components are effectively unchanged:

- diffusion-style evidence accumulation
- `dt = 0.001`
- response decided by first boundary crossing
- `z = a / 2`
- no explicit start-point bias manipulation

## 7.2 What changed the most

The biggest modeling changes concern:

- how `v` is mapped from `P/T/W`
- how self vs stranger changes `v`
- how tightly `a` is constrained
- whether `t0` is fixed or distributed
- whether fast RTs are retained or excluded

## 7.3 What each stage is best for

### Use `v2-v4` if your priority is:

- staying close to the original Study2 formulation
- preserving the original theoretical mapping from `P/T/W` to DDM parameters

But note:

- these versions are much more prone to unrealistic early RT artifacts

### Use `v5-v8` if your priority is:

- approximating realistic human RT distributions
- avoiding obvious left-tail artifacts
- creating a stable base for iterative fitting to empirical data

But note:

- self-prioritization effects may need deliberate strengthening because the model is now more constrained and conservative

---

## 8. Most likely next optimization targets

If the next goal is to preserve the realistic RT distribution from `v5-v8` while recovering a clearer self-prioritization effect, the most likely levers are:

1. the self/stranger modifier on `v`
2. whether that modifier should be additive or multiplicative
3. whether the self/stranger effect should remain constant or vary with `P/T/W`

At present, `z` is untouched and could later become a new mechanism to test.
This may be theoretically valuable if future modeling wants to distinguish:

- faster evidence extraction or prioritization (`v` mechanism)
- versus initial decisional bias or attentional starting advantage (`z` mechanism)

---

## 9. Bottom-line takeaway

The versions do not differ only in parameter values. They represent two different modeling philosophies.

### `v2-v4`

- closer to the original Study2 generator
- theoretically continuous and expressive
- but prone to unrealistic left-tail RT concentration

### `v5-v8`

- more bounded and empirically disciplined
- better RT realism
- but condition effects must now be rebuilt more carefully

For future re-optimization, `v5-v8` is likely the safer base if the ultimate goal is to fit real participants while exploring how `P/T/W` shapes the self-prioritization effect.
