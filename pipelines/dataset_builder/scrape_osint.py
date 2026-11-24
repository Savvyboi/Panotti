"""OSINT audio scraper using yt-dlp and ffmpeg to produce 16 kHz, 16-bit, Mono PCM WAVs."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import (
    CLASSES,
    CLASS_OSINT_QUERIES,
    DEFAULT_OSINT_QUERIES,
    EXCLUDED_TITLE_KEYWORDS,
    RAW_DATA_DIR,
    SAMPLE_RATE,
    resolve_ffmpeg_path,
)


def slugify_query(query: str) -> str:
    """Convert a search query into a filesystem-safe slug."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", query.strip().lower()).strip("_")
    return slug or "query"


def transcode_to_16k_mono_wav(
    input_path: Path,
    output_path: Path,
    ffmpeg_bin: str | None = None,
) -> Path:
    """Transcode an audio file to exact 16000 Hz, 16-bit, Mono PCM WAV via ffmpeg."""
    ffmpeg = resolve_ffmpeg_path(ffmpeg_bin)
    if not ffmpeg:
        raise FileNotFoundError(
            "ffmpeg binary not found; install ffmpeg or pass --ffmpeg-location."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(input_path),
        "-ar",
        str(SAMPLE_RATE),
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg transcode failed ({proc.returncode}): {proc.stderr.strip()}"
        )
    return output_path


class _YdlLogCollector:
    """Collects yt-dlp warnings and errors without flooding stdout."""

    def __init__(self) -> None:
        self.warnings: list[str] = []
        self.errors: list[str] = []

    def debug(self, _msg: str) -> None:
        return None

    def info(self, _msg: str) -> None:
        return None

    def warning(self, msg: str) -> None:
        self.warnings.append(str(msg))

    def error(self, msg: str) -> None:
        self.errors.append(str(msg))


def build_duration_filter(
    max_duration_sec: int = 300,
    excluded_keywords: tuple[str, ...] = EXCLUDED_TITLE_KEYWORDS,
    metadata_sink: dict[str, dict[str, Any]] | None = None,
):
    """Return a yt-dlp match_filter skipping livestreams, long clips, and excluded titles."""
    normalized_keywords = tuple(k.lower() for k in excluded_keywords if k)

    def _match_filter(info: dict[str, Any], *, incomplete: bool = False) -> str | None:
        if info.get("is_live"):
            return "Skipping live stream"

        duration = info.get("duration")
        if (
            max_duration_sec > 0
            and isinstance(duration, (int, float))
            and duration > max_duration_sec
        ):
            return f"Duration {duration}s exceeds max_duration_sec={max_duration_sec}s"

        title = str(info.get("title") or "").lower()
        for kw in normalized_keywords:
            if kw in title:
                return f"Title matches excluded keyword '{kw}'"

        vid = info.get("id")
        if metadata_sink is not None and isinstance(vid, str) and vid:
            metadata_sink[vid] = {
                "id": vid,
                "title": info.get("title", ""),
                "webpage_url": info.get("webpage_url", ""),
                "duration": duration,
            }
        return None

    return _match_filter


def build_yt_dlp_options(
    query_dir: Path,
    archive_path: Path,
    ffmpeg_location: str | None = None,
    max_duration_sec: int = 300,
    excluded_keywords: tuple[str, ...] = EXCLUDED_TITLE_KEYWORDS,
    max_downloads: int | None = None,
    logger: Any | None = None,
    metadata_sink: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Construct yt-dlp configuration dict enforcing WAV extraction + archive + filters."""
    opts: dict[str, Any] = {
        "format": "bestaudio/best",
        "outtmpl": str(query_dir / "%(id)s.%(ext)s"),
        "download_archive": str(archive_path),
        "noplaylist": True,
        "ignoreerrors": True,
        "quiet": True,
        "no_warnings": False,
        "match_filter": build_duration_filter(
            max_duration_sec=max_duration_sec,
            excluded_keywords=excluded_keywords,
            metadata_sink=metadata_sink,
        ),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }
        ],
        "postprocessor_args": {
            "extractaudio": [
                "-ar",
                str(SAMPLE_RATE),
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
            ]
        },
    }
    if max_downloads is not None and max_downloads > 0:
        opts["max_downloads"] = int(max_downloads)

    node_bin = shutil.which("node")
    if node_bin:
        opts["js_runtimes"] = {"node": {"path": node_bin}}

    if logger is not None:
        opts["logger"] = logger
    if ffmpeg_location:
        opts["ffmpeg_location"] = (
            str(Path(ffmpeg_location).parent)
            if Path(ffmpeg_location).is_file()
            else ffmpeg_location
        )
    return opts


def scrape_queries(
    queries: list[str] | None = None,
    output_dir: Path | str = RAW_DATA_DIR,
    max_per_query: int = 5,
    dry_run: bool = False,
    archive_file: Path | str | None = None,
    manifest_path: Path | str | None = None,
    ffmpeg_location: str | None = None,
    max_duration_sec: int = 300,
    search_multiplier: int = 3,
    excluded_keywords: tuple[str, ...] = EXCLUDED_TITLE_KEYWORDS,
) -> dict[str, Any]:
    """Download or plan OSINT acoustic downloads and emit a JSON manifest."""
    query_list = list(queries) if queries else list(DEFAULT_OSINT_QUERIES)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    archive = Path(archive_file) if archive_file else (out_dir / "download_archive.txt")
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        archive.touch()

    manifest_file = Path(manifest_path) if manifest_path else (out_dir / "manifest.json")
    manifest_file.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg_bin = resolve_ffmpeg_path(ffmpeg_location)
    candidate_count = max(1, int(max_per_query)) * max(1, int(search_multiplier))

    entries: list[dict[str, Any]] = []
    for query in query_list:
        slug = slugify_query(query)
        query_dir = out_dir / slug
        query_dir.mkdir(parents=True, exist_ok=True)
        search_spec = f"ytsearch{candidate_count}:{query}"

        if dry_run:
            entries.append(
                {
                    "query": query,
                    "slug": slug,
                    "search_spec": search_spec,
                    "max_per_query": max_per_query,
                    "max_duration_sec": max_duration_sec,
                    "output_dir": str(query_dir),
                    "archive_file": str(archive),
                    "status": "dry_run_planned",
                    "files": [],
                    "warnings": [],
                    "errors": [],
                }
            )
            continue

        import yt_dlp  # Lazy import so dry-run & tests stay lightweight

        preexisting_ids = {p.stem for p in query_dir.glob("*.wav")}
        ydl_logger = _YdlLogCollector()
        metadata_sink: dict[str, dict[str, Any]] = {}
        ydl_opts = build_yt_dlp_options(
            query_dir=query_dir,
            archive_path=archive,
            ffmpeg_location=ffmpeg_bin,
            max_duration_sec=max_duration_sec,
            excluded_keywords=excluded_keywords,
            max_downloads=max_per_query,
            logger=ydl_logger,
            metadata_sink=metadata_sink,
        )

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                ydl.extract_info(search_spec, download=True)
            except yt_dlp.utils.MaxDownloadsReached:
                pass

        downloaded_files: list[dict[str, Any]] = []
        for wav_path in sorted(query_dir.glob("*.wav")):
            vid = wav_path.stem
            if vid in preexisting_ids:
                continue
            normalized_tmp = query_dir / f"{vid}.norm.wav"
            transcode_to_16k_mono_wav(wav_path, normalized_tmp, ffmpeg_bin)
            normalized_tmp.replace(wav_path)
            meta = metadata_sink.get(vid, {})
            downloaded_files.append(
                {
                    "id": vid,
                    "title": meta.get("title", ""),
                    "webpage_url": meta.get("webpage_url", ""),
                    "duration": meta.get("duration"),
                    "wav_path": str(wav_path),
                }
            )

        entries.append(
            {
                "query": query,
                "slug": slug,
                "search_spec": search_spec,
                "max_per_query": max_per_query,
                "max_duration_sec": max_duration_sec,
                "output_dir": str(query_dir),
                "archive_file": str(archive),
                "status": "completed" if not ydl_logger.errors else "completed_with_errors",
                "files": downloaded_files,
                "warnings": ydl_logger.warnings,
                "errors": ydl_logger.errors,
            }
        )

    manifest: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "audio_format": {
            "sample_rate": SAMPLE_RATE,
            "channels": 1,
            "subtype": "PCM_16",
            "codec": "pcm_s16le",
        },
        "ffmpeg_path": ffmpeg_bin,
        "archive_file": str(archive),
        "queries": query_list,
        "max_per_query": max_per_query,
        "max_duration_sec": max_duration_sec,
        "search_multiplier": search_multiplier,
        "entries": entries,
    }
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape OSINT drone/piston engine audio into 16kHz 16-bit Mono WAV."
    )
    parser.add_argument(
        "--queries",
        nargs="+",
        default=None,
        help="Search queries passed to yt-dlp ytsearch.",
    )
    parser.add_argument(
        "--class-preset",
        choices=[*CLASSES, "all"],
        default=None,
        help="Use expanded per-class engine query presets and output into <output-dir>/<ClassName>.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RAW_DATA_DIR,
        help="Root output directory for downloaded WAVs and manifest.json.",
    )
    parser.add_argument(
        "--max-per-query",
        type=int,
        default=5,
        help="Maximum videos to download per query.",
    )
    parser.add_argument(
        "--max-duration",
        type=int,
        default=300,
        help="Maximum clip duration in seconds (0 disables duration cap; livestreams always skipped).",
    )
    parser.add_argument(
        "--search-multiplier",
        type=int,
        default=3,
        help="Candidate multiplier searched per query before filtering.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate download plan and manifest without hitting remote endpoints.",
    )
    parser.add_argument(
        "--archive-file",
        type=Path,
        default=None,
        help="Optional path to yt-dlp download archive file.",
    )
    parser.add_argument(
        "--ffmpeg-location",
        type=str,
        default=None,
        help="Optional explicit path to ffmpeg binary.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.class_preset is not None:
        target_classes = list(CLASSES) if args.class_preset == "all" else [args.class_preset]
        default_shared_archive = (
            (Path(args.output_dir).parent if Path(args.output_dir).name in CLASSES else Path(args.output_dir))
            / "global_download_archive.txt"
        )
        effective_archive = args.archive_file if args.archive_file is not None else default_shared_archive
        for cls_name in target_classes:
            cls_dir = (
                Path(args.output_dir)
                if Path(args.output_dir).name == cls_name
                else Path(args.output_dir) / cls_name
            )
            queries = list(args.queries) if args.queries else CLASS_OSINT_QUERIES[cls_name]
            manifest = scrape_queries(
                queries=queries,
                output_dir=cls_dir,
                max_per_query=args.max_per_query,
                dry_run=args.dry_run,
                archive_file=effective_archive,
                ffmpeg_location=args.ffmpeg_location,
                max_duration_sec=args.max_duration,
                search_multiplier=args.search_multiplier,
            )
            mode = "dry-run" if manifest["dry_run"] else "live"
            total_files = sum(len(entry.get("files", [])) for entry in manifest["entries"])
            total_errors = sum(len(entry.get("errors", [])) for entry in manifest["entries"])
            print(
                f"[{mode}][{cls_name}] {len(manifest['entries'])} queries "
                f"({total_files} new files, {total_errors} errors) -> {cls_dir / 'manifest.json'}"
            )
        return 0

    queries = list(args.queries) if args.queries else list(DEFAULT_OSINT_QUERIES)
    manifest = scrape_queries(
        queries=queries,
        output_dir=args.output_dir,
        max_per_query=args.max_per_query,
        dry_run=args.dry_run,
        archive_file=args.archive_file,
        ffmpeg_location=args.ffmpeg_location,
        max_duration_sec=args.max_duration,
        search_multiplier=args.search_multiplier,
    )
    mode = "dry-run" if manifest["dry_run"] else "live"
    total_files = sum(len(entry.get("files", [])) for entry in manifest["entries"])
    total_errors = sum(len(entry.get("errors", [])) for entry in manifest["entries"])
    print(
        f"[{mode}] Wrote manifest for {len(manifest['entries'])} queries "
        f"({total_files} new files, {total_errors} errors) to {Path(args.output_dir) / 'manifest.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
