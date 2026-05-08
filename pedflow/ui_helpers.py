from __future__ import annotations

from pathlib import Path


def display_path(project_root: Path, path: Path) -> str:
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return str(path)


def resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path


def file_options(
    project_root: Path,
    patterns: tuple[str, ...],
    fallback_values: tuple[str, ...] = (),
) -> list[str]:
    found: list[Path] = []
    for pattern in patterns:
        found.extend(path for path in project_root.glob(pattern) if path.is_file())

    found = sorted(set(found), key=lambda path: path.stat().st_mtime, reverse=True)
    options = [display_path(project_root, path) for path in found]
    for fallback in fallback_values:
        if fallback not in options:
            options.append(fallback)
    return options


def directory_options(
    project_root: Path,
    roots: tuple[str, ...],
    fallback_values: tuple[str, ...] = (),
) -> list[str]:
    found: list[Path] = []
    for root in roots:
        path = resolve_path(project_root, root)
        if path.exists():
            found.extend(candidate for candidate in path.rglob("*") if candidate.is_dir())
            if path.is_dir():
                found.append(path)

    found = sorted(set(found), key=lambda path: display_path(project_root, path))
    options = [display_path(project_root, path) for path in found]
    for fallback in fallback_values:
        if fallback not in options:
            options.append(fallback)
    return options


def keep_or_first(current: str, options: list[str]) -> str:
    if current in options:
        return current
    if options:
        return options[0]
    return current


def serial_port_options(fallback: str = "COM5") -> list[str]:
    try:
        from serial.tools import list_ports
    except ImportError:
        return [fallback]

    ports = [port.device for port in list_ports.comports()]
    if fallback not in ports:
        ports.append(fallback)
    return ports
