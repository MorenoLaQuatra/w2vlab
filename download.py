"""Download a corpus with an explicit YAML config."""

import sys
from pathlib import Path
from urllib.request import urlopen

from yaml_config_manager import load_config
from w2vlab.wiki import download_wikipedia

ALICE_URL = "https://www.gutenberg.org/cache/epub/11/pg11.txt"


def download_alice(path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} already exists")
    print(f"Downloading {ALICE_URL}", flush=True)
    with urlopen(ALICE_URL, timeout=30) as response:
        text = response.read().decode("utf-8-sig")
    start = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end = "*** END OF THE PROJECT GUTENBERG EBOOK"
    if start not in text or end not in text:
        raise ValueError("Gutenberg text markers not found")
    text = text.split(start, 1)[1].split("***", 1)[1].split(end, 1)[0]
    path.write_text(text.strip() + "\n", encoding="utf-8")
    print(f"Saved {path}")


def main():
    if not any(arg == "--config" or arg.startswith("--config=") for arg in sys.argv[1:]):
        raise SystemExit("Error: --config is required. Example: python download.py --config configs/download.yaml")
    config = load_config()
    if config.source == "wiki":
        if config.articles < -1:
            raise ValueError("articles must be -1, 0, or positive")
        download_wikipedia(config.language, config.output, config.articles)
    elif config.source == "alice":
        download_alice(config.output)
    else:
        raise ValueError("source must be wiki or alice")


if __name__ == "__main__":
    main()
