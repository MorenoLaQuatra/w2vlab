"""Explore vectors with an explicit YAML config."""

import sys

from yaml_config_manager import load_config
from w2vlab.vectors import WordVectors


def show(results):
    for word, score in results:
        print(f"{word:<20} {score:.3f}")


def main():
    if not any(arg == "--config" or arg.startswith("--config=") for arg in sys.argv[1:]):
        raise SystemExit("Error: --config is required. Example: python inference.py --config configs/inference.yaml")
    config = load_config()
    vectors = WordVectors(config.model)
    if not config.interactive:
        try:
            show(vectors.most_similar(config.word, config.top))
        except KeyError as error:
            print(error.args[0])
        return

    print("Commands: similar WORD, similarity WORD WORD, analogy A B C, quit")
    while True:
        try:
            parts = input("> ").lower().split()
            if not parts or parts[0] == "quit":
                break
            if parts[0] == "similar" and len(parts) == 2:
                show(vectors.most_similar(parts[1], config.top))
            elif parts[0] == "similarity" and len(parts) == 3:
                print(f"{vectors.similarity(parts[1], parts[2]):.3f}")
            elif parts[0] == "analogy" and len(parts) == 4:
                show(vectors.analogy(*parts[1:], top=config.top))
            else:
                print("Use: similar WORD | similarity WORD WORD | analogy A B C | quit")
        except (KeyError, ValueError) as error:
            print(error.args[0])
        except (EOFError, KeyboardInterrupt):
            break


if __name__ == "__main__":
    main()
