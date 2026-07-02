# `src/configs`

**Purpose.** Python-side representation of the YAML configuration files
under `configs/` — typed dataclasses / schemas plus validation.

Planned responsibilities (not yet implemented):

- `DatasetConfig`, `ModelConfig`, `TrainConfig`, `PathsConfig` dataclasses.
- YAML → dataclass loading with defaults and validation.
- CLI override merging.

Note the distinction:
- `configs/` at the project root holds the **YAML files themselves**.
- `src/configs/` holds the **Python code that loads and validates** them.
