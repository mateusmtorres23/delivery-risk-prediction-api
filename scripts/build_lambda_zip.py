import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "build" / "delivery-risk.zip"
UNPACKED_LIMIT = 250 * 1024**2
PACKAGES_WITH_TESTS = {"numpy", "pandas", "scipy", "sklearn"}


def stage_package(work_dir: Path) -> Path:
    package = work_dir / "package"
    requirements = work_dir / "requirements.txt"
    subprocess.run(
        [
            "uv", "export", "--locked", "--no-dev", "--no-hashes",
            "--no-emit-project", "--output-file", str(requirements),
        ],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    subprocess.run(
        [
            "uv", "pip", "install", "--python", sys.executable,
            "--python-platform", "x86_64-manylinux_2_34", "--only-binary", ":all:",
            "--target", str(package), "--requirement", str(requirements),
        ],
        cwd=ROOT,
        check=True,
    )

    shutil.copytree(
        ROOT / "app", package / "app", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    launcher = package / "run.sh"
    launcher.write_text(
        '#!/bin/bash\n'
        'cd "${LAMBDA_TASK_ROOT:-.}"\n'
        'exec python -m uvicorn app.app:app --host 0.0.0.0 --port "${PORT:-8000}"\n',
        encoding="utf-8",
        newline="\n",
    )
    launcher.chmod(0o755)
    return package


def write_archive(package: Path, archive: Path) -> int:
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as zip_file:
        for source in package.rglob("*"):
            relative = source.relative_to(package)
            if relative.parts[0] in PACKAGES_WITH_TESTS and "tests" in relative.parts:
                continue
            if source.is_file():
                zip_file.write(source, relative.as_posix())

        return sum(member.file_size for member in zip_file.infolist())


def main() -> None:
    if sys.platform != "linux" or sys.version_info[:2] != (3, 12):
        raise SystemExit("Run the Lambda ZIP builder with Python 3.12 on Linux or WSL.")

    OUTPUT.parent.mkdir(exist_ok=True)
    with TemporaryDirectory(dir="/tmp") as work_dir, TemporaryDirectory(dir=OUTPUT.parent) as output_dir:
        package = stage_package(Path(work_dir))
        archive = Path(output_dir) / OUTPUT.name
        unpacked_size = write_archive(package, archive)

        remaining = UNPACKED_LIMIT - unpacked_size
        if remaining <= 0:
            raise SystemExit(f"The package leaves no room for the Lambda layer ({unpacked_size:,} bytes).")

        archive.replace(OUTPUT)

    print(f"Saved {OUTPUT} ({OUTPUT.stat().st_size:,} bytes compressed).")
    print(f"Unpacked size: {unpacked_size:,} bytes. Space left for layers: {remaining:,} bytes.")


if __name__ == "__main__":
    main()
