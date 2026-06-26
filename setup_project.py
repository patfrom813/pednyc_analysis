from pathlib import Path

ROOT = Path.cwd()

folders = [

    # Source code
    "src",

    # Data
    "data",

    # Output
    "outputs",

    # Models
    "models",

    # Documentation
    "docs",

]

for folder in folders:
    (ROOT / folder).mkdir(parents=True, exist_ok=True)

files = [

    "README.md",
    ".gitignore",
    "requirements.txt"

]

for file in files:

    path = ROOT / file

    if not path.exists():
        path.touch()

print("Project initialized successfully.")