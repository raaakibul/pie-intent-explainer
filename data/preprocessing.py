from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import yaml

from utils.coordinate_transform import bounding_box_sequence_to_ego_xy


def _load_pie_toolkit(pie_root: str):
    pie_repo = os.path.join(pie_root, "PIE")
    if pie_repo not in sys.path:
        sys.path.insert(0, pie_repo)
    try:
        from pie_data import PIE  # type: ignore
    except ImportError as e:
        raise ImportError(
            f"Could not import pie_data.PIE from {pie_repo}. "
            "Run `bash data/download_pie.sh <pie_root>` first."
        ) from e
    return PIE(data_path=pie_root)

