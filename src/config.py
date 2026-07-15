import ast
from datetime import date, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "inmet"
OUTPUTS_BASE_DIR = PROJECT_ROOT / "outputs"
DEFAULT_OUTPUT_DATE_FORMAT = "%d_%m_%y"


def output_date_folder(
    run_date: date | datetime | None = None,
    date_format: str = DEFAULT_OUTPUT_DATE_FORMAT,
) -> str:
    """Return the dated folder name used to group generated outputs."""
    if run_date is None:
        run_date = datetime.now()
    return run_date.strftime(date_format)


def dated_output_root(
    output_root: Path,
    run_date: date | datetime | None = None,
    date_format: str = DEFAULT_OUTPUT_DATE_FORMAT,
) -> Path:
    """Append the daily output folder to a root path, unless already present."""
    output_root = Path(output_root)
    folder_name = output_date_folder(run_date=run_date, date_format=date_format)
    if output_root.name == folder_name:
        return output_root
    return output_root / folder_name


OUTPUTS_DIR = dated_output_root(OUTPUTS_BASE_DIR)


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
        output_root = OUTPUTS_BASE_DIR
    else:
        output_root = Path(str(raw_output_root))
        if not output_root.is_absolute():
            output_root = PROJECT_ROOT / output_root

    group_by_day = config.get("group_outputs_by_day", True)
    if isinstance(group_by_day, str):
        group_by_day = group_by_day.strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }

    if not group_by_day:
        return output_root

    date_format = str(config.get("date_folder_format") or DEFAULT_OUTPUT_DATE_FORMAT)
    return dated_output_root(output_root, date_format=date_format)

