# Guassion-Process-Experiment-Design: Study 2 Generative Tuning Handoff

## 1. Project Background

This project is trying to build a generative cognitive model for the **Self-Prioritization Effect (SPE)** in a perceptual matching task.

The broad scientific goal is:

- use a DDM-style generative model to simulate trial-level behavior,
- make the simulated data look as close as possible to real participants,
- and understand how the experimental design parameters `P | T | W` shape the self-advantage mechanism.

In this project:

- `P` = practice amount / practice count
- `T` = stimulus preview time
- `W` = response window

The core theoretical question is:

- how do different combinations of `P`, `T`, and `W` change task difficulty,
- and through which psychological mechanism does that eventually change the self-prioritization effect?

At the modeling level, the current working assumption is:

- SPE is expressed mainly through **drift rate (`v`) differences** between `self` and `stranger`,
- while other parts of the model are temporarily kept simpler and more constrained,
- so we can isolate whether a fixed self/stranger drift advantage is enough to recover the observed effect.

This is important: the user does **not** want to jump too quickly into a very complex PTW-dependent self-effect model.

The current strategy is:

1. first stabilize the RT distribution and remove unrealistic left-tail artifacts,
2. then test whether a **fixed** self vs stranger drift difference can recover SPE,
3. only after that, if needed, move to a model where the self/stranger difference itself varies with `P/T/W`.

So the current phase is still a **controlled hypothesis test**, not the final model.

## 2. What Has Already Been Learned

Several notebook versions have already been tried:

- `S2_gen_data_optimized_cp_v3.ipynb`: added runnable checks and fast-RT diagnostics
- `S2_gen_data_optimized_cp_v4.ipynb`: tried controlling `a` with a lower bound
- `S2_gen_data_optimized_cp_v5.ipynb`: moved to bounded parameter ranges
- `S2_gen_data_optimized_cp_v6.ipynb`: kept bounded RT-friendly structure and increased fixed self/stranger `v` separation additively
- `S2_gen_data_optimized_cp_v7.ipynb`: reverted to a **multiplicative** self/stranger modifier on `v`

### Important empirical/diagnostic lesson from earlier versions

Earlier versions had an unrealistic left-tail RT cluster.

The main causes appeared to be:

- `a` becoming too small for some PTW combinations,
- fixed `t0`,
- and ultra-fast boundary crossing close to `t0`.

To address this, the later bounded versions introduced:

- `v` bounded to `0.05 - 0.40`
- `a` bounded to `0.10 - 0.30`
- subject-level `t0` distribution instead of a single fixed value
- explicit filtering / marking of anticipatory responses with `RT < 0.25s`

These changes successfully reduced the abnormal left tail.

## 3. Current State: v7

The current notebook to continue from is:

- `D:\GitHub_programe\GitHub\Guassion-Process-Experiment-Design\1_Code\Python_for_Check\S2_gen_data_optimized_cp_v7.ipynb`

### Current v7 modeling assumptions

The model defines:

- a PTW-derived `ease_score`
- a bounded `v_base` from task difficulty
- a bounded `a`
- subject-level `t0`
- multiplicative self/stranger modulation on `v`

Current condition effect form in v7:

- self: `v = clip(v_0 * (1 + ALPHA1), 0.05, 0.40)`
- stranger: `v = clip(v_0 * (1 + ALPHA2), 0.05, 0.40)`

Current parameter values in v7:

- `ALPHA1 = 0.60`
- `ALPHA2 = -0.25`

### Current v7 results

The latest reported results were:

#### Benchmark summary

- valid rate about `89%`
- fast RT rate about `1.78%`
- very fast RT rate `< 0.23s` = `0%`
- mean valid RT about `0.295s`
- mean `a` about `0.198`
- mean `v` about `0.245`

#### Label summary

- self mean RT: about `0.293958`
- stranger mean RT: about `0.293898`
- self mean `v`: about `0.314`
- stranger mean `v`: about `0.159`
- self mean `a` and stranger mean `a`: both about `0.195`
- self and stranger `t0`: both about `0.280`

### Interpretation of current v7

This result means:

- the **left-tail problem is largely controlled**
- but the **self vs stranger behavioral difference is still essentially absent**

So:

- the bounded RT-friendly structure is doing its job,
- but the current fixed multiplicative `v` difference has **not yet been strong enough behaviorally** to bring SPE back in RT.

This does **not** necessarily mean the multiplicative idea is impossible.
It means that under the current constrained regime, it has not yet recovered the effect.

## 4. What the Next AI Must Do

The next AI should help with **iterative tuning**, but under a very strict rule:

### Only change:

- `ALPHA1`
- `ALPHA2`

### Do not change for now:

- the overall notebook structure
- the PTW-to-difficulty logic
- the bounded `v` range `0.05 - 0.40`
- the bounded `a` range `0.10 - 0.30`
- the `t0` distribution settings
- the anticipatory-response threshold (`RT < 0.25s`)
- the `ease_score` critical point
- the `a` computation
- the fixed-hypothesis assumption that self/stranger difference is **not yet PTW-dependent**

The user explicitly wants to first test the simple hypothesis:

- self-prioritization comes from a fixed attention-based information-priority effect,
- expressed as a stronger drift for self than stranger,
- while the rest of the task-difficulty structure stays unchanged.

Only after this fixed-effect hypothesis is exhausted should later AIs move to PTW-dependent self/stranger differences.

## 5. Tuning Objective

The tuning goal is:

- keep the improved RT distribution properties from `v5/v6/v7`,
- while increasing self vs stranger behavioral separation enough that SPE appears again.

The immediate behavioral target is:

- self and stranger should separate in RT and/or valid-response behavior,
- but the left-tail artifact should not come back.

### Practical tuning criteria

When testing new `ALPHA1 / ALPHA2` combinations, check:

1. **Fast RT control**
   - fast RT rate (`RT < 0.25s`) should stay reasonably low
   - very fast RT rate (`RT < 0.23s`) should stay near zero or very low

2. **Behavioral separation**
   - self mean RT should become detectably different from stranger mean RT
   - ideally in the expected SPE direction

3. **Parameter plausibility**
   - `v` should still mostly respect the intended bounded range
   - the simulation should not just collapse into clipped values at the upper or lower bound for most trials

4. **Distribution sanity**
   - do not reintroduce the old abnormal left-tail spike

## 6. Suggested Tuning Workflow

The next AI should use the following loop:

1. Open `S2_gen_data_optimized_cp_v7.ipynb`
2. Duplicate it to a new version if needed, or patch it carefully
3. Change only `ALPHA1` and `ALPHA2`
4. Run the notebook fully
5. Record:
   - benchmark summary
   - fast RT summary
   - label summary
   - risky PTW combinations
6. Compare against the previous run
7. Decide whether the change:
   - improves self/stranger separation
   - keeps left-tail control acceptable

### Suggested search direction

Since v7 still has almost no RT difference despite:

- self mean `v` > stranger mean `v`

the next AI should try **stronger self/stranger contrast**.

Reasonable next moves:

- increase `ALPHA1`
- decrease `ALPHA2` further (more negative)
- or both

Examples to try next:

- `ALPHA1 = 0.80`, `ALPHA2 = -0.30`
- `ALPHA1 = 1.00`, `ALPHA2 = -0.35`
- `ALPHA1 = 1.20`, `ALPHA2 = -0.40`

But the next AI should proceed gradually, not jump to extreme values blindly.

## 7. How to Judge Success

The next AI should treat a tuning step as promising if:

- self vs stranger RT difference becomes clearly non-zero,
- the difference is in the theoretically expected direction,
- fast RT rate does not blow up,
- very fast RT rate stays very low,
- and the left-tail shape remains much more realistic than the old unconstrained versions.

If a stronger `ALPHA1 / ALPHA2` combination successfully restores SPE while keeping left-tail control, then:

- that supports the user's current fixed-effect hypothesis,
- and only after that should later AIs test whether PTW-dependent condition effects are needed.

If repeated tuning of `ALPHA1 / ALPHA2` still cannot recover behavioral SPE without damaging the RT distribution, then:

- the next theoretical step would be to let self/stranger modulation vary with `P/T/W`,
- but that is **not** the current task unless the user explicitly asks for it.

## 8. Non-Negotiable Constraints for the Next AI

The next AI must:

- preserve the current bounded RT-friendly regime
- only tune `ALPHA1 / ALPHA2`
- keep the hypothesis fixed-effect for now
- document each attempted `ALPHA1 / ALPHA2` pair and resulting summaries

The next AI must **not**:

- redesign the whole model
- change `a` logic again
- change `t0` logic again
- move to PTW-dependent self/stranger effects yet
- silently alter multiple parameters at once

## 9. Recommended Immediate Next Action

Start from `S2_gen_data_optimized_cp_v7.ipynb` and run a controlled tuning sweep over `ALPHA1 / ALPHA2` only.

The most useful first objective is:

- find the strongest fixed self/stranger multiplicative contrast that restores SPE
- without pushing the fast-RT left tail back into an unrealistic range

