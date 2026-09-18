# Architecture detail

## Data flow and tensor shapes

Let `B` = batch size, `T_obs` = 10 (observed frames), `T_pred` = 15 (future
frames), `H,W` = 224 (crop size fed to ResNet-50)


## Why the intent embedding is soft, not a hard branch

`IntentHead` returns softmax-probability-derived embeddings
`if/else`, so the whole model stays end-to-end differentiable.

`tests/test_model.py::test_intent_conditioning_changes_trajectory` for a
sanity check that this conditioning actually changes the trajectory output.

## Why the maneuver decision is rule-based, not LLM-generated

`explain/prompt_templates.py::recommend_maneuver()` is deterministic code,
not an LLM decision.

## Coordinate frame approximation

`utils/coordinate_transform.py` projects pedestrian bbox feet points onto a
flat ground plane using PIE's approximate published camera intrinsics.