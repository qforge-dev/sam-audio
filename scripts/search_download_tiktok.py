#!/usr/bin/env python3
"""Search the web for TikTok videos and download one with yt-dlp."""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

TIKTOK_VIDEO_URL = re.compile(
    r"https://(?:www\.)?tiktok\.com/@[^\s\"'<>/&]+/video/\d+",
    re.IGNORECASE,
)


def find_yt_dlp() -> str:
    configured = os.environ.get("YTDLP")
    if configured:
        return configured
    repository_copy = Path(__file__).resolve().parents[1] / "pipeline/.venv/bin/yt-dlp"
    if repository_copy.is_file():
        return str(repository_copy)
    executable = shutil.which("yt-dlp")
    if executable:
        return executable
    raise RuntimeError(
        "yt-dlp was not found. Run `brew install yt-dlp` or install the "
        "pipeline virtual environment."
    )


def urls_from_page(page: str, *, limit: int) -> list[str]:
    page = html.unescape(page).replace("\\u002F", "/").replace("\\/", "/")
    urls: list[str] = []
    for match in TIKTOK_VIDEO_URL.finditer(page):
        url = match.group(0).split("?", 1)[0]
        if url not in urls:
            urls.append(url)
        if len(urls) >= limit:
            break
    return urls


def search_page(url: str, parameters: dict[str, str], *, limit: int) -> list[str]:
    request = urllib.request.Request(
        f"{url}?{urllib.parse.urlencode(parameters)}",
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 Chrome/126 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            page = response.read().decode("utf-8", errors="replace")
    except (OSError, urllib.error.URLError):  # noqa: UP041
        return []
    return urls_from_page(page, limit=limit)


def search_yahoo(query: str, args: argparse.Namespace, *, limit: int) -> list[str]:
    command = [
        *yt_dlp_arguments(args),
        "--quiet",
        "--no-warnings",
        "--flat-playlist",
        "--dump-single-json",
        f"yvsearch{limit}:site:tiktok.com {query}",
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=45,
        )
        payload = json.loads(result.stdout)
    except (subprocess.SubprocessError, json.JSONDecodeError):  # noqa: UP041
        return []
    urls: list[str] = []
    for item in payload.get("entries") or []:
        if not isinstance(item, dict):
            continue
        for key in ("webpage_url", "original_url", "url"):
            value = str(item.get(key) or "")
            found = urls_from_page(value, limit=1)
            if found and found[0] not in urls:
                urls.append(found[0])
                break
        if len(urls) >= limit:
            break
    return urls


def search_tiktok(query: str, args: argparse.Namespace, *, limit: int) -> list[str]:
    direct = urls_from_page(query, limit=1)
    if direct:
        return direct

    engines: dict[str, Callable[[], list[str]]] = {
        "yahoo": lambda: search_yahoo(query, args, limit=limit),
        "brave": lambda: search_page(
            "https://search.brave.com/search",
            {"q": f"site:tiktok.com/@ {query}", "source": "web"},
            limit=limit,
        ),
        "duckduckgo": lambda: search_page(
            "https://html.duckduckgo.com/html/",
            {"q": f"site:tiktok.com/@ {query}"},
            limit=limit,
        ),
    }
    selected = (
        list(engines) if args.search_provider == "auto" else [args.search_provider]
    )
    urls: list[str] = []
    for name in selected:
        for url in engines[name]():
            if url not in urls:
                urls.append(url)
            if len(urls) >= limit:
                return urls
    return urls


def yt_dlp_arguments(args: argparse.Namespace) -> list[str]:
    result = [args.yt_dlp, "--no-update", "--no-playlist"]
    if args.cookies_from_browser:
        result.extend(["--cookies-from-browser", args.cookies_from_browser])
    return result


def inspect_video(url: str, args: argparse.Namespace) -> dict[str, Any] | None:
    command = [
        *yt_dlp_arguments(args),
        "--quiet",
        "--no-warnings",
        "--skip-download",
        "--dump-single-json",
        url,
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=45,
        )
        payload = json.loads(result.stdout)
    # Parentheses keep this script compatible with the repository's Python 3.12
    # pipeline environment as well as newer Python versions.
    except (subprocess.SubprocessError, json.JSONDecodeError):  # noqa: UP041
        return None
    return {
        "url": str(payload.get("webpage_url") or url),
        "id": str(payload.get("id") or "unknown"),
        "title": str(payload.get("title") or "Untitled TikTok"),
        "uploader": str(payload.get("uploader") or payload.get("creator") or "unknown"),
        "duration": payload.get("duration"),
    }


def download_video(video: dict[str, Any], args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    command = [
        *yt_dlp_arguments(args),
        "--newline",
        "--restrict-filenames",
        "--merge-output-format",
        "mp4",
        "-f",
        "bv*+ba/b",
        "-o",
        str(args.output / "%(id)s_%(title).80B.%(ext)s"),
        "--print",
        "after_move:Downloaded: %(filepath)s",
        str(video["url"]),
    ]
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search public web indexes for TikTok video URLs, inspect them "
            "with yt-dlp, and download the selected match."
        )
    )
    parser.add_argument("query", help="Search text, for example: gameplay dialogue")
    parser.add_argument(
        "--search-provider",
        choices=("auto", "yahoo", "brave", "duckduckgo"),
        default="auto",
        help="Search backend; auto tries all available backends (default: auto)",
    )
    parser.add_argument(
        "--pick",
        type=int,
        default=1,
        help="1-based result number to download (default: 1)",
    )
    parser.add_argument(
        "--results",
        type=int,
        default=10,
        help="Maximum number of search results to inspect (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path.home() / "Downloads/tiktok-search",
        help="Download directory (default: ~/Downloads/tiktok-search)",
    )
    parser.add_argument(
        "--cookies-from-browser",
        metavar="BROWSER",
        help="Pass browser cookies to yt-dlp, for example: chrome or safari",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Print downloadable matches without downloading one",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.pick < 1 or args.results < 1:
        raise SystemExit("--pick and --results must be positive")
    args.yt_dlp = find_yt_dlp()

    urls = search_tiktok(
        args.query,
        args,
        limit=max(args.results * 3, args.results),
    )
    matches: list[dict[str, Any]] = []
    for url in urls:
        video = inspect_video(url, args)
        if video is not None:
            matches.append(video)
        if len(matches) >= args.results:
            break

    if not matches:
        print(
            "No downloadable TikTok videos were found. Try a broader query, "
            "a direct TikTok video URL, or --cookies-from-browser chrome.",
            file=sys.stderr,
        )
        return 1

    print(f"Found {len(matches)} downloadable TikTok video(s):")
    for index, video in enumerate(matches, start=1):
        duration = (
            f"{float(video['duration']):.0f}s"
            if video.get("duration") is not None
            else "unknown duration"
        )
        print(
            f"{index:2}. {video['title']} — @{video['uploader']} — {duration}\n"
            f"    {video['url']}"
        )

    if args.list_only:
        return 0
    if args.pick > len(matches):
        print(
            f"--pick {args.pick} is outside the 1–{len(matches)} result range.",
            file=sys.stderr,
        )
        return 2

    chosen = matches[args.pick - 1]
    print(f"\nDownloading result {args.pick} to {args.output} ...")
    download_video(chosen, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
