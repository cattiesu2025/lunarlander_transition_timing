# Formal v7 protocol amendment — original-scenario checkpoint gate

Date: 2026-09-17 AEST. Status: **post-freeze, pre-held-out amendment**. The original frozen protocol and failed selection output remain unchanged and must be reported alongside this amendment. This amendment does not claim that the original v7 preregistration was followed without deviation.

## Trigger and timing

The original v7 selector stopped before sealed held-out evaluation because ECON seed 2011 had no checkpoint with at least 15/18 landings under *both* original and low-descent-speed development interventions. At 1M it landed 14/18 original and 15/18 low-speed; at 600k it landed 18/18 original and 14/18 low-speed. Both had 18/18 primary persistent-thrust events in each intervention. The sealed v7 held-out grid has not been evaluated. No v7 cross-condition ONSET-time contrast was used to formulate this amendment.

## Revised gate

The confirmatory estimand remains the original-scenario `tau_ECON - tau_DESC` persistent-thrust ONSET contrast. For **every** one of the 60 models, inspect the same five fixed checkpoints (600k, 700k, 800k, 900k, 1M) and select the latest checkpoint with original-development `landed >= 15/18` and original-development primary events `>= 15/18`. The low-speed development records must still be complete, free of technical errors, and hash-matched, but low-speed landing and event counts do not determine checkpoint eligibility. No ONSET time, pairwise contrast, or desired ordering enters selection.

All 60 models must have a checkpoint under this uniform rule before the existing sealed held-out grid is opened. The amended selection and its audit receive a new output directory and SHA-256 lock. The original 59/60 selection directory is retained as evidence of the failed frozen gate.

## Unchanged elements and reporting

Training seeds and models, rewards, detector definitions, development and sealed held-out grids, original/low-speed interventions, primary hypothesis, 12/18 joint-event estimability threshold, whole-seed bootstrap, sign test, and terminal-outcome reporting remain as frozen. Neither the 300 development evaluations nor the 60 trainings are repeated. The low-speed intervention remains a supporting result with full event and landing denominators, not an eligibility condition.

The paper must disclose the original gate failure, the exact revised gate, and that the revision occurred after inspecting development task-quality counts but before any v7 held-out result. The amended held-out analysis is distinct from a wholly preregistered v7 analysis.
