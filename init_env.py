import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def log(message: str) -> None:
    print(f"[init-env] {message}")


def run_command(command: list[str], description: str) -> None:
    log(f"{description}...")
    log(f"Running command: {' '.join(command)}")
    subprocess.run(command, check=True)


def get_pip_path(venv_path: Path) -> Path:
    if os.name == "nt":
        return venv_path / "Scripts" / "pip.exe"
    return venv_path / "bin" / "pip"


def get_python_path(venv_path: Path) -> Path:
    if os.name == "nt":
        return venv_path / "Scripts" / "python.exe"
    return venv_path / "bin" / "python"


def get_activate_hint(venv_path: Path) -> str:
    if os.name == "nt":
        return f".\\{venv_path}\\Scripts\\activate"
    return f"source {venv_path}/bin/activate"


def get_vscode_interpreter_instruction() -> str:
    if platform.system() == "Darwin":
        return (
            "VS Code: Cmd+Shift+P -> Python: Select Interpreter "
            "-> choose the printed interpreter path."
        )
    return (
        "VS Code: Ctrl+Shift+P -> Python: Select Interpreter "
        "-> choose the printed interpreter path."
    )


def install_requirements(pip_path: Path, requirements_file: Path) -> None:
    if requirements_file.exists():
        run_command(
            [str(pip_path), "install", "-r", str(requirements_file)],
            f"Installing dependencies from {requirements_file}",
        )
    else:
        log(f"{requirements_file} not found. Skipping.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Initialize Python virtual environment and install dependencies."
    )
    parser.add_argument(
        "--venv",
        default="venv",
        help="Virtual environment directory name (default: venv).",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate the virtual environment if it already exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    project_root = Path.cwd()
    venv_path = project_root / args.venv

    log(f"Project root: {project_root}")
    log(f"Python executable: {sys.executable}")
    log(f"Detected OS: {platform.system()} (os.name={os.name})")
    log(f"Target virtual environment: {venv_path}")

    if venv_path.exists():
        if args.recreate:
            log(f"Removing existing environment at {venv_path}")
            shutil.rmtree(venv_path)
        else:
            log(
                f"Environment already exists at {venv_path}. Reusing it. "
                "Use --recreate to rebuild."
            )

    if not venv_path.exists():
        run_command(
            [sys.executable, "-m", "venv", str(venv_path)],
            "Creating virtual environment",
        )

    pip_path = get_pip_path(venv_path)
    if not pip_path.exists():
        log(f"Expected pip was not found at: {pip_path}")
        return 1

    run_command(
        [str(pip_path), "install", "--upgrade", "pip", "setuptools", "wheel"],
        "Upgrading packaging tools",
    )

    install_requirements(pip_path, project_root / "requirements.txt")
    install_requirements(pip_path, project_root / "requirements-dev.txt")

    pyproject_path = project_root / "pyproject.toml"
    if pyproject_path.exists():
        log(
            "Detected pyproject.toml. If this repo uses Poetry/PDM/uv, "
            "install deps with that tool instead of requirements files."
        )

    log("Environment setup complete.")
    log(f"Activate with: {get_activate_hint(Path(args.venv))}")
    log(
        "Interpreter path for VS Code: "
        f"{get_python_path(venv_path)}"
    )
    log(get_vscode_interpreter_instruction())
    log(f"Deactivate with 'deactivate'")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as error:
        log(f"Command failed with exit code {error.returncode}")
        raise SystemExit(error.returncode)
