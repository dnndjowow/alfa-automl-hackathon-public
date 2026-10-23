"""Check the project interpreter and exact official package versions; API is opt-in."""

from __future__ import annotations

import argparse
import importlib
from importlib import metadata, util
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
CORE = {
    "numpy": "numpy", "scipy": "scipy", "pandas": "pandas",
    "scikit-learn": "sklearn", "joblib": "joblib", "threadpoolctl": "threadpoolctl",
    "openai": "openai", "httpx": "httpx", "pydantic": "pydantic",
    "psutil": "psutil", "jsonschema": "jsonschema",
}
IMPORTS = CORE | {
    "polars": "polars", "pyarrow": "pyarrow", "catboost": "catboost",
    "lightgbm": "lightgbm", "xgboost": "xgboost", "imbalanced-learn": "imblearn",
    "category-encoders": "category_encoders", "feature-engine": "feature_engine",
    "statsmodels": "statsmodels", "optuna": "optuna", "torch": "torch",
    "autogluon.tabular": "autogluon.tabular", "lightautoml": "lightautoml",
    "flaml": "flaml", "langchain": "langchain", "langchain-openai": "langchain_openai",
    "langgraph": "langgraph", "python-dotenv": "dotenv", "pyyaml": "yaml",
    "orjson": "orjson", "matplotlib": "matplotlib", "seaborn": "seaborn",
    "tqdm": "tqdm", "cloudpickle": "cloudpickle", "tenacity": "tenacity",
}
API_ENV = ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL")


def pins() -> dict[str, tuple[str, str]]:
    result = {}
    for line in (ROOT / "reference" / "requirements.txt").read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([\w.-]+)(?:\[[^\]]+\])?==([^\s]+)", line)
        if not match:
            raise ValueError("Expected exact package pins in reference/requirements.txt")
        result[match[1].lower().replace("_", "-")] = (line, match[2])
    return result


def check_api() -> bool:
    missing = [name for name in API_ENV if not os.environ.get(name)]
    if missing:
        print("API not checked; missing environment variable names: " + ", ".join(missing))
        return False
    try:
        from openai import OpenAI
        with OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ["OPENAI_BASE_URL"],
                    timeout=25.0, max_retries=0) as client:
            response = client.chat.completions.create(
                model=os.environ["OPENAI_MODEL"],
                messages=[{"role": "user", "content": "Ответь одним словом: OK"}],
                max_tokens=32,
            )
        if not response.choices:
            print("API returned no choices")
            return False
        print("API: request completed; response body and credentials are not printed")
        return True
    except Exception as error:
        # Provider error messages can contain request details or credentials.
        print(f"API failed: {type(error).__name__}; HTTP status: {getattr(error, 'status_code', None)}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="Check every package in the official list")
    parser.add_argument("--install-core", action="store_true", help="Install the core subset, constrained by the official list")
    parser.add_argument("--api", action="store_true", help="Make ONE request to the provided API; may consume team budget")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        print("ERROR: Python 3.12 is required")
        return 1
    official = pins()
    if args.install_core:
        if sys.prefix == sys.base_prefix:
            print("ERROR: --install-core must run inside the project virtual environment")
            return 1
        # pip constraint files cannot contain extras (the official file does).
        with tempfile.TemporaryDirectory(prefix="alfa-constraints-") as temporary:
            constraints = Path(temporary) / "constraints.txt"
            constraints.write_text(
                "\n".join(f"{name}=={pin[1]}" for name, pin in official.items()) + "\n",
                encoding="utf-8",
            )
            subprocess.run(
                [sys.executable, "-X", "utf8", "-m", "pip", "install", "--disable-pip-version-check", "--no-input",
                 "--constraint", str(constraints), *[official[name][0] for name in CORE]], check=True,
            )
    print(f"Python: {sys.version.split()[0]}; platform: {sys.platform}")
    print("Profile: " + ("full official list" if args.full else "core subset only"))
    success = True
    for name in (official if args.full else CORE):
        expected = official[name][1]
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError:
            actual = "MISSING"
        ok = actual == expected
        print(f"{'OK' if ok else 'FAIL'}: {name} expected={expected} installed={actual}")
        success &= ok
        if ok and name in IMPORTS:
            try:
                importlib.import_module(IMPORTS[name])
            except Exception as error:
                print(f"FAIL import: {name}: {type(error).__name__}")
                success = False
    print("MCP import available: " + str(util.find_spec("mcp") is not None))
    print("MCP is absent from the official direct dependency list; its platform contract still needs confirmation.")
    for name in API_ENV:
        print(f"{name}: {'set' if os.environ.get(name) else 'not set'} (value hidden)")
    if args.api:
        success = check_api() and success
    else:
        print("API not called; use --api only with the provided credentials.")
    print("This check does not reproduce the Linux platform or test an ML solution.")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
