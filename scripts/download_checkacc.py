"""Download one free Mixkit clip per UCF11 class into checkAcc/."""

from __future__ import annotations

import re
import ssl
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "checkAcc"
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
CTX = ssl.create_default_context()
MAX_BYTES = 40 * 1024 * 1024

# Category pages that usually contain a close visual match for UCF11.
PAGES = {
    "basketball": ["https://mixkit.co/free-stock-video/basketball/"],
    "biking": [
        "https://mixkit.co/free-stock-video/cycling/",
        "https://mixkit.co/free-stock-video/mountain-biking/",
    ],
    "diving": [
        "https://mixkit.co/free-stock-video/diving/",
        "https://mixkit.co/free-stock-video/swimming-pool/",
    ],
    "golf_swing": ["https://mixkit.co/free-stock-video/golf/"],
    "horse_riding": [
        "https://mixkit.co/free-stock-video/horseback-riding/",
        "https://mixkit.co/free-stock-video/horse/",
    ],
    "soccer_juggling": [
        "https://mixkit.co/free-stock-video/soccer/",
        "https://mixkit.co/free-stock-video/football/",
    ],
    "swing": [
        "https://mixkit.co/free-stock-video/playground/",
        "https://mixkit.co/free-stock-video/swing/",
    ],
    "tennis_swing": ["https://mixkit.co/free-stock-video/tennis/"],
    "trampoline_jumping": ["https://mixkit.co/free-stock-video/trampoline/"],
    "volleyball_spiking": ["https://mixkit.co/free-stock-video/volleyball/"],
    "walking": ["https://mixkit.co/free-stock-video/walking/"],
}

ID_RE = re.compile(r"assets\.mixkit\.co/videos/(\d+)")


def log(msg: str) -> None:
    sys.stdout.buffer.write((msg + "\n").encode("utf-8", errors="replace"))
    sys.stdout.buffer.flush()


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, context=CTX, timeout=60) as resp:
        return resp.read()


def first_video_ids(html: str) -> list[str]:
    seen: list[str] = []
    for vid in ID_RE.findall(html):
        if vid not in seen:
            seen.append(vid)
    return seen


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


def fetch_class(label: str, pages: list[str]) -> Path | None:
    dest_dir = OUT / label
    dest_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(
        p
        for p in dest_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".mp4", ".webm", ".ogv", ".avi", ".mpg"}
    )
    if existing:
        log(f"  keep {existing[0].relative_to(ROOT)}")
        return existing[0]

    for page in pages:
        try:
            html = fetch(page).decode("utf-8", errors="ignore")
        except Exception as exc:
            log(f"  page failed {page}: {exc}")
            continue
        for vid in first_video_ids(html)[:6]:
            url = f"https://assets.mixkit.co/videos/{vid}/{vid}-720.mp4"
            dest = dest_dir / f"mixkit_{vid}.mp4"
            try:
                log(f"  downloading {url}")
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
        time.sleep(0.4)
    return None


def main() -> int:
    log(f"Downloading Mixkit clips into {OUT}")
    missing = []
    for label, pages in PAGES.items():
        log(f"[{label}]")
        if fetch_class(label, pages) is None:
            missing.append(label)
    if missing:
        log("Missing classes: " + ", ".join(missing))
        return 1
    log("All 11 classes have a clip.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
