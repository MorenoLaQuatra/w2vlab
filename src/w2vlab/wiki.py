"""Turn a Wikipedia article dump into training text."""

import bz2
import re
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.request import Request, urlopen

import mwparserfromhell


def download_wikipedia(language, path, max_articles=-1):
    if not re.fullmatch(r"[a-z][a-z0-9-]*", language):
        raise ValueError("use a Wikipedia language code such as scn, it, or en")
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} already exists")
    wiki = f"{language}wiki"
    url = f"https://dumps.wikimedia.org/{wiki}/latest/{wiki}-latest-pages-articles.xml.bz2"
    print(f"Reading {url}", flush=True)
    request = Request(url, headers={"User-Agent":
                      "w2vlab/0.1 (educational; https://github.com/MorenoLaQuatra/word2vec-from-scratch-template)"})

    with tempfile.TemporaryDirectory(dir=path.parent) as temp:
        staged = Path(temp) / path.name
        count = 0
        with urlopen(request, timeout=60) as response, bz2.BZ2File(response) as archive, \
                staged.open("w", encoding="utf-8") as output:
            events = ET.iterparse(archive, events=("start", "end"))
            root = next(events)[1]
            for event, page in events:
                if event != "end" or page.tag.rsplit("}", 1)[-1] != "page":
                    continue
                raw = page.findtext("./{*}revision/{*}text")
                if page.findtext("./{*}ns") == "0" and page.find("./{*}redirect") is None and raw:
                    code = mwparserfromhell.parse(raw)
                    for link in reversed(code.filter_wikilinks()):
                        if ":" in str(link.title):  # file, category, and interwiki links
                            code.remove(link)
                    text = " ".join(code.strip_code().split())
                    if text:
                        output.write(text + "\n")
                        count += 1
                        if count % 100 == 0:
                            print(f"\rExtracted {count:,} articles", end="", flush=True)
                root.clear()  # Do not keep earlier XML pages in memory.
                if max_articles > 0 and count >= max_articles:
                    break
        if not count:
            raise ValueError("no articles found in the dump")
        staged.replace(path)
    print(f"\rSaved {count:,} articles to {path}       ")
