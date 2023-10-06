"""
    config_parsing.py

    Summary:
        Contains functionality for reading and validating configuration files from schemas.

    Author: Trym Tengesdal
"""
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any, List, Optional

import rlmpc.common.file_utils as futils
import dacite
from cerberus import Validator


def extract(
    data_class: Any,
    config_file: Path,
    config_schema: Path,
    converter: Optional[dict] = None,
    **kwargs,
) -> Any:
    """Extracts configuration settings from a configuration file, and converts them to a dataclass.
    Validation is performed using the input schema.

    Args:
        data_class (Any): Dataclass to convert the settings to.
        config_file (Path): Path to the configuration file.
        config_schema (Path): Path to the schema used for config validation.
        converter (Optional[dict]): Dictionary specifying data types to convert to in the settings to the dataclass. Defaults to None.

    Returns:
        Any: Configuration settings as a dataclass.
    """

    schema = futils.read_yaml_into_dict(config_schema)

    settings = parse(config_file, schema, **kwargs)
    settings = override(settings, schema, **kwargs)

    settings = convert_settings_dict_to_dataclass(data_class, settings, converter)

    return settings


def convert_settings_dict_to_dataclass(
    data_class, config_dict: dict, converter: Optional[dict] = None
) -> Any:
    """Converts a settings dictionary to a dataclass.

    Args:
        dataclass (Any): Data class to convert to.
        config_dict (dict): Dictionary containing the settings.
        converter (Optional[dict]): Dictionary specifying data types to convert to in the settings to the dataclass. Defaults to None.

    Returns:
        Any: The dataclass.
    """
    if not is_dataclass(data_class):
        raise ValueError(
            f"Desired class is not a dataclass type, its type is {data_class}"
        )

    if hasattr(data_class, "from_dict") and callable(getattr(data_class, "from_dict")):
        return data_class.from_dict(config_dict)

    if converter is not None:
        return dacite.from_dict(
            data_class=data_class,
            data=config_dict,
            config=dacite.Config(type_hooks=converter),
        )

    return dacite.from_dict(data_class=data_class, data=config_dict)


def validate(settings: dict, schema: dict) -> None:
    """Validates the settings against the schema.

    Args:
        settings (dict): Configuration settings to validate.

    Raises:
        ValueError: On empty settings/schema or invalid settings.
    """
    if not settings:
        raise ValueError("Empty settings!")

    if not schema:
        raise ValueError("Empty schema!")

    validator = Validator(schema)

    if not validator.validate(settings):
        raise ValueError(f"Cerberus validation Error: {validator.errors}")


def extract_valid_sections(schema: dict) -> List[str]:
    """Extracts the valid main sections from the schema.

    Args:
        schema (dict): Configuration schema.

    Raises:
        ValueError: On empty schema.

    Returns:
        List[str]: List of valid main sections as strings.
    """
    if schema is None:
        raise ValueError("No configuration schema provided!")

    sections = []
    for section in schema.keys():
        sections.append(section)

    return sections


def parse(file_name: Path, schema: dict, section: Optional[str] = None) -> dict:
    """Parses a configuration file into a dictionary, and validates the settings.

    Args:
        file_name (Path): Path to the configuration file.
        schema (dict): Configuration schema to validate the settings against.
        section (Optional[str]): Main section to parse. Defaults to None.

    Returns:
        dict: Configuration settings.
    """
    settings = futils.read_yaml_into_dict(file_name)

    section_settings = settings
    section_schema = schema
    if section:
        section_settings = settings[section]
        section_schema = schema[section]

    validate(section_settings, section_schema)
    return section_settings


def override(
    settings: dict, schema: dict, section: Optional[str] = None, **kwargs
) -> dict:
    """Overrides settings with keyword arguments, and validates the new values.

    NOTE: Assumes only one section in the provided configuration schema.

    Args:
        settings (dict): Configuration settings to override.
        schema (dict): Configuration schema to validate the new settings against.
        section (Optional[str]): Main section to override. Defaults to None.

    Raises:
        ValueError: On empty keyword arguments or invalid settings.

    Returns:
        dict: The new settings.
    """
    if not kwargs:
        return settings

    new_settings = settings
    section_schema = schema
    if section:
        new_settings = settings[section]
        section_schema = schema[section]

    for key, value in kwargs.items():
        new_settings[key] = value

    validate(new_settings, section_schema)
    return new_settings
