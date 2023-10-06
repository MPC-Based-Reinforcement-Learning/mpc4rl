"""
    file_utils.py

    Summary:
        Contains general non-math related utility functions.

    Author: Trym Tengesdal
"""


from pathlib import Path

import yaml


def read_yaml_into_dict(file_name: Path) -> dict:
    with file_name.open(mode="r", encoding="utf-8") as file:
        output_dict = yaml.safe_load(file)
    return output_dict
