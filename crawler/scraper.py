"""
crawler/scraper.py
Crawlt Schweizer Behördenwebseiten themenbasiert.
Strategie: Verifizierte Seed-URLs + Links nur innerhalb desselben Themenpfads folgen.
"""

import asyncio
import json
import hashlib
from datetime import date
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
from urllib.parse import urlparse, urljoin
from collections import deque

import httpx
from bs4 import BeautifulSoup

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; BehoerdenBot/1.0; Swiss Government Info)"
}

# ── Verifizierte Seeds mit Themenpfad-Filter ──────────────────────────────────
# Format: (start_url, kanton, thema, url_muss_enthalten)
# url_muss_enthalten: Der Crawler folgt nur Links die diesen String enthalten
# So bleibt er beim Thema und schweift nicht ab

SEEDS = [
    # Bund - ch.ch
    ("https://www.ch.ch/de/umzug/",             "bund", "umzug",          "/umzug/"),
    ("https://www.ch.ch/de/familie/heirat/",    "bund", "heirat",         "/familie/heirat"),
    ("https://www.ch.ch/de/familie/geburt/",    "bund", "geburt",         "/familie/geburt"),
    ("https://www.ch.ch/de/familie/todesfall/", "bund", "todesfall",      "/familie/todesfall"),
    ("https://www.ch.ch/de/arbeit/selbstaendigkeit/", "bund", "firmengründung", "/arbeit/selbst"),
    ("https://www.ch.ch/de/alter/pensionierung/",     "bund", "pensionierung",  "/alter/"),
    ("https://www.ch.ch/de/steuern-und-finanzen/steuererklarung/", "bund", "steuern", "/steuern-und-finanzen/"),

    # Admin.ch - Bundesbehörden
    ("https://www.sem.admin.ch/sem/de/home/themen/aufenthalt.html", "bund", "einwanderung", "/aufenthalt"),
    ("https://www.ahv-iv.ch/de/Sozialversicherungen/Alters-und-Hinterlassenenversicherung-AHV", "bund", "ahv", "/AHV"),
    ("https://www.estv.admin.ch/estv/de/home/mehrwertsteuer.html", "bund", "mwst", "/mehrwertsteuer"),

    # Kanton Zürich
    ("https://www.zh.ch/de/steuern-finanzen/steuern/privatpersonen.html", "ZH", "steuern", "/steuern"),
    ("https://www.zh.ch/de/familie/geburt-heirat-tod.html", "ZH", "familie", "/familie/"),

    # Kanton Bern
    ("https://www.be.ch/de/start/themen/steuern-und-finanzen/steuern.html", "BE", "steuern", "/steuern"),
]

MAX_PAGES_PER_SEED = 6
MIN_CONTENT_LENGTH = 300


@dataclass
class CrawledPage:
    url: str
    kanton: str
    thema: str
    titel: str
    inhalt: str
    quelle_name: str
    sprache: str
    gecrawlt_am: str
    chunk_id: str


def _domain(url):
    return urlparse(url).netloc

def _domain_name(url):
    return _domain(url).replace("www.", "")

def _chunk_id(url):
    return hashlib.sha256(url.encode()).hexdigest()[:16]

def _clean_text(soup):
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "form"]):
        tag.decompose()
    main = (
        soup.find("main") or
        soup.find("article") or
        soup.find(id="content") or
        soup.find(class_="content") or
        soup
    )
    text = main.get_text(separator="\n", strip=True)
    lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 30]
    return "\n".join(lines)

def _extract_filtered_links(soup, base_url, must_contain):
    """Gibt nur Links zurück die must_contain im Pfad haben."""
    base_domain = _domain(base_url)
    links = []
    for a in soup.find_all("a", href=True):
        href = urljoin(base_url, a["href"])
        parsed = urlparse(href)
        if (parsed.netloc == base_domain and
            must_contain in parsed.path and
            not href.endswith((".pdf", ".zip", ".docx")) and
            parsed.scheme in ("http", "https")):
            links.append(href.split("?")[0].split("#")[0])
    return list(dict.fromkeys(links))


async def crawl_page(client, url, kanton, thema, must_contain):
    try:
        r = await client.get(url, timeout=15.0, follow_redirects=True)
        r.raise_for_status()
        if "text/html" not in r.headers.get("content-type", ""):
            return None, []
    except Exception as e:
        print(f"    ✗ {str(e)[:80]}")
        return None, []

    soup = BeautifulSoup(r.text, "html.parser")
    inhalt = _clean_text(soup)
    links = _extract_filtered_links(soup, url, must_contain)

    if len(inhalt) < MIN_CONTENT_LENGTH:
        print(f"    ⚠ Zu wenig Inhalt ({len(inhalt)} Zeichen) — JS-rendered?")
        return None, links

    titel = soup.title.string.strip() if soup.title else url

    page = CrawledPage(
        url=url,
        kanton=kanton,
        thema=thema,
        titel=titel[:120],
        inhalt=inhalt,
        quelle_name=_domain_name(url),
        sprache="de",
        gecrawlt_am=date.today().isoformat(),
        chunk_id=_chunk_id(url),
    )
    print(f"    ✓ {len(inhalt):,} Zeichen — {titel[:55]}")
    return page, links


async def crawl_seed(client, seed_url, kanton, thema, must_contain, visited):
    queue = deque([seed_url])
    pages = []
    count = 0

    while queue and count < MAX_PAGES_PER_SEED:
        url = queue.popleft()
        if url in visited:
            continue
        visited.add(url)
        count += 1

        print(f"  [{count}/{MAX_PAGES_PER_SEED}] {url}")
        page, new_links = await crawl_page(client, url, kanton, thema, must_contain)

        if page:
            pages.append(page)
            out = OUTPUT_DIR / f"{page.chunk_id}.json"
            out.write_text(json.dumps(asdict(page), ensure_ascii=False, indent=2))

        for link in new_links[:10]:
            if link not in visited:
                queue.append(link)

        await asyncio.sleep(0.3)

    return pages


async def run_crawler():
    # Alten raw-Ordner leeren
    for f in OUTPUT_DIR.glob("*.json"):
        f.unlink()

    visited = set()
    all_pages = []

    async with httpx.AsyncClient(headers=HEADERS) as client:
        for seed_url, kanton, thema, must_contain in SEEDS:
            print(f"\n{'='*65}")
            print(f"[{kanton}] {thema} | Filter: '{must_contain}'")
            print(f"{'='*65}")
            pages = await crawl_seed(client, seed_url, kanton, thema, must_contain, visited)
            all_pages.extend(pages)
            print(f"  → {len(pages)} Seiten gespeichert")

    summary = OUTPUT_DIR / "_alle_seiten.json"
    summary.write_text(json.dumps([asdict(p) for p in all_pages], ensure_ascii=False, indent=2))

    print(f"\n{'='*65}")
    print(f"FERTIG: {len(all_pages)} Seiten total gecrawlt")
    print(f"Themen: {set(p.thema for p in all_pages)}")
    return [asdict(p) for p in all_pages]


if __name__ == "__main__":
    asyncio.run(run_crawler())