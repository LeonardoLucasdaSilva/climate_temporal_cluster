import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "inmet"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def _strip_inline_comment(line: str) -> str:
    """Remove YAML comments while preserving hashes inside quoted strings."""
    quote: str | None = None
    escaped = False
    for index, char in enumerate(line):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char in {"'", '"'}:
            quote = None if quote == char else char if quote is None else quote
            continue
        if char == "#" and quote is None:
            return line[:index]
    return line


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
    if value.startswith(("[", "{", "(")):
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            pass
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def load_output_config(path: Path) -> dict[str, object]:
    """Load a small YAML config file without an external dependency."""
    config: dict[str, object] = {}
    stack: list[tuple[int, dict[str, object]]] = [(-1, config)]
    for line in path.read_text(encoding="utf-8").splitlines():
        line = _strip_inline_comment(line).rstrip()
        if not line.strip() or ":" not in line:
            continue
        indent = len(line) - len(line.lstrip())
        line = line.strip()
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()

        parent = stack[-1][1]
        if value:
            parent[key] = _parse_config_value(value)
            continue

        child: dict[str, object] = {}
        parent[key] = child
        stack.append((indent, child))
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

