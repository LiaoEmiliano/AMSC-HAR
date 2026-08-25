"""Download a second, title-matched stock-video set for UCF11 classes."""

from __future__ import annotations

import json
import re
import ssl
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "checkAcc_alt"
USED = ROOT / "checkAcc_mixkit"
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
CTX = ssl.create_default_context()
MAX_BYTES = 40 * 1024 * 1024
ID_RE = re.compile(r"assets\.mixkit\.co/videos/(\d+)")
LD_RE = re.compile(
    r'<script type="application/ld\+json"[^>]*>(.*?)</script>', re.S
)

PAGES = {
    "basketball": {
        "pages": [
            "https://mixkit.co/free-stock-video/basketball/",
            "https://mixkit.co/free-stock-video/basketball-game/",
            "https://mixkit.co/free-stock-video/basketball-player/",
        ],
        "want": ["basketball", "dribbl", "shoot", "court", "hoop", "player"],
        "avoid": ["empty", "ball on", "close-up of a basketball"],
    },
    "biking": {
        "pages": [
            "https://mixkit.co/free-stock-video/cycling/",
            "https://mixkit.co/free-stock-video/mountain-biking/",
            "https://mixkit.co/free-stock-video/bicycle/",
        ],
        "want": ["cycl", "bik", "riding", "mountain", "pedal"],
        "avoid": ["parked", "static"],
    },
    "diving": {
        "pages": [
            "https://mixkit.co/free-stock-video/diving/",
            "https://mixkit.co/free-stock-video/swimming-pool/",
            "https://mixkit.co/free-stock-video/jumping-into-water/",
        ],
        "want": ["div", "jump", "pool", "splash", "board"],
        "avoid": ["scuba", "underwater fish", "snorkeling"],
    },
    "golf_swing": {
        "pages": [
            "https://mixkit.co/free-stock-video/golf/",
            "https://mixkit.co/free-stock-video/golf-swing/",
            "https://mixkit.co/free-stock-video/golfer/",
        ],
        "want": ["swing", "golfer", "tee", "drive", "hitting"],
        "avoid": ["course landscape", "empty"],
    },
    "horse_riding": {
        "pages": [
            "https://mixkit.co/free-stock-video/horseback-riding/",
            "https://mixkit.co/free-stock-video/horse-riding/",
            "https://mixkit.co/free-stock-video/horse/",
        ],
        "want": ["rid", "horseback", "jockey", "gallop", "equestrian"],
        "avoid": ["grazing", "standing", "portrait"],
    },
    "soccer_juggling": {
        "pages": [
            "https://mixkit.co/free-stock-video/soccer/",
            "https://mixkit.co/free-stock-video/football/",
            "https://mixkit.co/free-stock-video/soccer-player/",
        ],
        "want": ["jugg", "keepie", "ball control", "freestyle", "solo", "kick"],
        "avoid": ["stadium empty", "crowd"],
    },
    "swing": {
        "pages": [
            "https://mixkit.co/free-stock-video/playground/",
            "https://mixkit.co/free-stock-video/swing/",
            "https://mixkit.co/free-stock-video/children-playing/",
        ],
        "want": ["swing", "playground", "child"],
        "avoid": ["golf", "baseball", "tennis", "dance"],
    },
    "tennis_swing": {
        "pages": [
            "https://mixkit.co/free-stock-video/tennis/",
            "https://mixkit.co/free-stock-video/tennis-player/",
            "https://mixkit.co/free-stock-video/tennis-court/",
        ],
        "want": ["tennis", "forehand", "backhand", "serve", "player", "racket"],
        "avoid": ["empty court"],
    },
    "trampoline_jumping": {
        "pages": [
            "https://mixkit.co/free-stock-video/trampoline/",
            "https://mixkit.co/free-stock-video/jumping/",
        ],
        "want": ["trampoline", "jump"],
        "avoid": ["bungee"],
    },
    "volleyball_spiking": {
        "pages": [
            "https://mixkit.co/free-stock-video/volleyball/",
            "https://mixkit.co/free-stock-video/beach-volleyball/",
        ],
        "want": ["spike", "smash", "hit", "attack", "player", "match"],
        "avoid": ["empty"],
    },
    "walking": {
        "pages": [
            "https://mixkit.co/free-stock-video/walking/",
            "https://mixkit.co/free-stock-video/people-walking/",
            "https://mixkit.co/free-stock-video/pedestrian/",
        ],
        "want": ["walk", "stroll", "pedestrian", "sidewalk"],
        "avoid": ["run", "jog", "dog"],
    },
}


def log(msg: str) -> None:
    sys.stdout.buffer.write((msg + "\n").encode("utf-8", errors="replace"))
    sys.stdout.buffer.flush()


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, context=CTX, timeout=60) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def used_ids() -> set[str]:
    ids: set[str] = set()
    if USED.exists():
        for p in USED.rglob("mixkit_*.mp4"):
            ids.add(p.stem.replace("mixkit_", ""))
    return ids


def walk_ld(obj, items: list[tuple[str, str]]) -> None:
    if isinstance(obj, dict):
        name = str(obj.get("name") or obj.get("headline") or "")
        thumb = str(obj.get("thumbnailUrl") or "")
        content = str(obj.get("contentUrl") or obj.get("url") or "")
        blob = " ".join([thumb, content])
        found = ID_RE.findall(blob)
        if name and found:
            items.append((found[0], name))
        for v in obj.values():
            walk_ld(v, items)
    elif isinstance(obj, list):
        for v in obj:
            walk_ld(v, items)


def videos_from_page(html: str) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for block in LD_RE.findall(html):
        try:
            walk_ld(json.loads(block), items)
        except json.JSONDecodeError:
            continue
    seen: dict[str, str] = {}
    for vid, name in items:
        seen.setdefault(vid, name)
    # fallback: ids without titles
    for vid in ID_RE.findall(html):
        seen.setdefault(vid, "")
    return list(seen.items())


def score(name: str, spec: dict) -> int:
    text = name.lower()
    if not text:
        return 0
    s = 0
    for w in spec["want"]:
        if w in text:
            s += 3
    for w in spec["avoid"]:
        if w in text:
            s -= 4
    return s


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, context=CTX, timeout=120) as resp, tmp.open("wb") as out:
        copied = 0
        while True:
            chunk = resp.read(256 * 1024)
            if not chunk:
                break
            copied += len(chunk)
            if copied > MAX_BYTES:
                raise RuntimeError("file too large")
            out.write(chunk)
    tmp.replace(dest)


def pick_class(label: str, spec: dict, skip: set[str]) -> Path | None:
    dest_dir = OUT / label
    dest_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(
        p
        for p in dest_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".mp4", ".webm", ".avi", ".mpg"}
    )
    if existing:
        log(f"  keep {existing[0].relative_to(ROOT)}")
        return existing[0]

    ranked: list[tuple[int, str, str]] = []
    for page in spec["pages"]:
        try:
            html = fetch(page)
        except Exception as exc:
            log(f"  page failed {page}: {exc}")
            continue
        for vid, name in videos_from_page(html):
            if vid in skip:
                continue
            ranked.append((score(name, spec), vid, name or f"id {vid}"))
        time.sleep(0.3)

    ranked.sort(key=lambda x: (-x[0], x[1]))
    # unique ids, keep best title
    best: dict[str, tuple[int, str]] = {}
    for s, vid, name in ranked:
        if vid not in best:
            best[vid] = (s, name)
    ordered = sorted(
        ((s, vid, name) for vid, (s, name) in best.items()),
        key=lambda x: (-x[0], x[1]),
    )
    log(f"  candidates: {len(ordered)}")
    for s, vid, name in ordered[:8]:
        log(f"    score={s:>2}  {vid}  {name}")

    for s, vid, name in ordered[:10]:
        url = f"https://assets.mixkit.co/videos/{vid}/{vid}-720.mp4"
        dest = dest_dir / f"mixkit_{vid}.mp4"
        try:
            log(f"  downloading {vid} ({name})")
            download(url, dest)
            if dest.stat().st_size < 50_000:
                dest.unlink(missing_ok=True)
                continue
            log(f"  saved {dest.relative_to(ROOT)} ({dest.stat().st_size} bytes)")
            return dest
        except urllib.error.HTTPError as exc:
            dest.unlink(missing_ok=True)
            log(f"  http {exc.code} for {vid}")
        except Exception as exc:
            dest.unlink(missing_ok=True)
            log(f"  failed {vid}: {exc}")
    return None


def main() -> int:
    skip = used_ids()
    log(f"Downloading title-matched Mixkit clips into {OUT}")
    log(f"Skipping {len(skip)} ids from previous Mixkit set")
    missing = []
    for label, spec in PAGES.items():
        log(f"[{label}]")
        if pick_class(label, spec, skip) is None:
            missing.append(label)
    if missing:
        log("Missing classes: " + ", ".join(missing))
        return 1
    log("All 11 classes have a clip.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
