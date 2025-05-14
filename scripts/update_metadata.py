import re
from pathlib import Path
import yaml

# List of files to update
FILES_TO_UPDATE = {
    "pyproject.toml": [
        ("version", r'(version\s*=\s*)"\d+\.\d+\.\d+"'),
        ("summary", r'(description\s*=\s*)".*"'),
        ("author", r'(author\s*=\s*)".*"'),
    ],
    "recipe/meta.yaml": [
        ("version", r'(version\s*=\s*)"\d+\.\d+\.\d+"'),
        ("summary", r'(summary:\s*)".*"'),
    ],
    "snpio/docs/source/conf.py": [
        ("version", r'(release\s*=\s*)"\d+\.\d+\.\d+"'),
    ],
    "snpio/docs/HEADER.yaml": [
        ("version", r'(version\s*:\s*)"\d+\.\d+\.\d+"'),
        ("authors", r'(authors\s*:\s*)".*"'),
        ("date", r'(date\s*:\s*)"\d+\-\d+\-\d+"'),
    ],
    "snpio/docs/template.tex": [
        ("date", r"(\{\\Large\s*)\d+\-\d+\-\d+"),
        ("version", r"(\{\\Large Version:\s*)\d+\.\d+\.\d+"),
    ],
    "snpio/scripts/tag_release.sh": [
        ("version", r'(.*v)"\d+\.\d+\.\d+"'),
    ],
}


def update_metadata(metadata_file="metadata.yaml"):
    metadata_file = Path("snpio", "docs", "metadata.yaml")
    if not metadata_file.exists():
        print(f"Metadata file not found: {metadata_file}")
        return

    # Load metadata
    with open(metadata_file, "r") as f:
        metadata = yaml.safe_load(f)

    for infile, patterns in FILES_TO_UPDATE.items():
        try:
            with open(infile, "r") as f:
                content = f.read()

            for key, pattern in patterns:
                if key in metadata:
                    # Preserve the key part and only update the value, ensuring correct formatting

                    if infile == "snpio/docs/template.tex":
                        content = re.sub(
                            pattern, lambda m: f"{m.group(1)}{metadata[key]}", content
                        )
                    else:
                        content = re.sub(
                            pattern, lambda m: f'{m.group(1)}"{metadata[key]}"', content
                        )

            with open(infile, "w") as f:
                f.write(content)
            print(f"Updated {infile}")

        except FileNotFoundError:
            print(f"Skipping {infile}: Not found")


if __name__ == "__main__":
    update_metadata()
