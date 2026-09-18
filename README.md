# pie-intent-explainer
Intent-Explainable Trajectory: pedestrian intent-conditioned trajectory prediction with LLM-generated maneuver explanations.

Built on top of the [PIE dataset](https://github.com/aras62/PIE) (Pedestrian Intention
Estimation). This project links intent-conditioned trajectory prediction to
robot action justification via an LLM.

## Architecture
Input: PIE video clip (10 frames, ~0.5s) + pedestrian bbox sequence
CNN backbone (ResNet-50) → spatial features per frame
Bi-LSTM → temporal encoding of pedestrian motion
Intent head: crossing/not-crossing classifier + confidence
Trajectory head: LSTM decoder conditioned on intent embedding
LLM explainer (Llama-3.1-8B or GPT-4): explains the recommended maneuver


See [`docs/architecture.md`](docs/architecture.md) for full tensor shapes.


## Installation

```bash
https://github.com/raaakibul/pie-intent-explainer.git
cd pie-intent-explainer
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Get the PIE dataset

```bash
bash data/download_pie.sh /path/to/PIE_dataset
```

Point `configs/default.yaml -> data.pie_root` at the resulting directory.

