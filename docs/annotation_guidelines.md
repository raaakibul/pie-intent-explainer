# Human justification annotation guidelines

`evaluate.py` scores generated explanations against human references with
BLEU-4. This document specifies the crowdsourcing protocol and file format.

## Output format

A single JSON file at `data/human_explanations.json`, mapping clip IDs to a
single reference:

```json
{
  "set07_video0003_0091_0": "The pedestrian is about to step off the curb, so the car should slow down.",
  "set07_video0003_0091_1": "The person is standing still, facing away from traffic, so no change is needed."
}
```

## Annotation task

For each of 100 sampled test-split clips, annotators see the 10 observed
frames with bbox overlay, the ground-truth crossing label, and a stated
synthetic ego-vehicle speed.

## Sampling

Sample from the test split only, stratified across crossing/not-crossing
labels (~50/50) and across the synthetic ego-speed range.

