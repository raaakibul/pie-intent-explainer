# What this project does

Predicts what a pedestrian near the road is about to do? cross or not. And where they'll physically move. Uses an LLM to explain why the car should react the way it does, in plain English (for example: "A pedestrian is crossing with 87% confidence, moving toward the curb. The car should slow to 6 m/s and give 1.5m clearance.")

# How it works?
Watches ~0.5s of pedestrian motion (video + bounding boxes)
Predicts intent (crossing / not crossing) with a confidence score
Predicts future trajectory, shaped by that intent
A simple rule (not AI) decides the maneuver — target speed, clearance
An LLM explains that decision in natural language

#Where it's used?
Self-driving cars, delivery robots real-time, human-readable reasons for braking or yielding.