from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "inmet"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def _parse_config_value(value: str) -> object:
    """Parse simple scalar values from YAML-like config files."""
    value = value.strip()
    if value.lower() in {"", "null", "none"}:
        return None
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.startswith(("'", '"')) and value.endswith(("'", '"')):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def load_output_config(path: Path) -> dict[str, object]:
    """Load a small key-value YAML file without an external dependency."""
    config: dict[str, object] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        config[key.strip()] = _parse_config_value(value)
    return config


def output_root_from_config(config: dict[str, object]) -> Path:
    """Resolve the configured output root relative to the project root."""
    raw_output_root = config.get("output_root")
    if raw_output_root is None:
        return OUTPUTS_DIR

    output_root = Path(str(raw_output_root))
    if output_root.is_absolute():
        return output_root
    return PROJECT_ROOT / output_root

