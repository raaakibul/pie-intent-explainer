from setuptools import setup, find_packages

setup(
    name="pie-intent-explainer",
    version="0.1.0",
    description="Intent-conditioned pedestrian trajectory prediction with "
                 "LLM-generated maneuver justification on the PIE dataset.",
    packages=find_packages(include=["data", "data.*", "models", "models.*",
                                     "explain", "explain.*", "utils", "utils.*"]),
    python_requires=">=3.9",
)
