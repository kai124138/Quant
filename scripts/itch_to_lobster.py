#!/usr/bin/env python3
"""Stream a local gzip ITCH day into per-symbol Parquet (ns, shares, price units)."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", ".cache/mpl")

from lob_latency.itch import reconstruct


def cpu_name() -> str:
    """Read the local CPU description without network or hardware benchmarks."""
    if platform.system() == "Darwin":
        result = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                capture_output=True, text=True, check=False)
        if result.stdout.strip():
            return result.stdout.strip()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def compile_filter() -> Path:
    """Compile C99 with the local cc; cache by source digest, without dependencies."""
    source = ROOT / "tools/itch_filter.c"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    cache = ROOT / ".cache/itch"
    cache.mkdir(parents=True, exist_ok=True)
    binary = cache / f"itch_filter-{digest[:16]}"
    if not binary.exists():
        print("Compiling ITCH filter with cc -std=c99 -O2", flush=True)
        with tempfile.TemporaryDirectory(dir=cache) as temporary:
            target = Path(temporary) / "itch_filter"
            subprocess.run(["cc", "-std=c99", "-O2", "-Wall", "-Wextra",
                            str(source), "-o", str(target)], check=True)
            target.replace(binary)
    return binary


def input_date(path: Path) -> date:
    """Read the calendar date from SMMDDYY-v50.txt.gz or SMMDDYY_head*.gz."""
    match = re.fullmatch(r"S(\d{2})(\d{2})(\d{2})(?:[-_].*)?\.gz", path.name)
    if not match:
        raise ValueError("Expected an SMMDDYY-v50.txt.gz or SMMDDYY_head*.gz filename")
    month, day, year = map(int, match.groups())
    return date(2000 + year, month, day)


def filter_input(binary: Path, source: Path, output: Path, symbols: list[str]) -> dict:
    """Pipe gzip directly into C; accept only explicit premature-gzip-EOF errors."""
    started = time.perf_counter()
    with tempfile.TemporaryFile() as gzip_log, tempfile.TemporaryFile() as filter_log:
        gzip = subprocess.Popen(["gzip", "-dc", str(source)], stdout=subprocess.PIPE, stderr=gzip_log)
        filtered = None
        try:
            filtered = subprocess.Popen([str(binary), str(output), ",".join(symbols)],
                                        stdin=gzip.stdout, stderr=filter_log)
            gzip.stdout.close()
            filter_code = filtered.wait()
            gzip_code = gzip.wait()
        except BaseException:
            if filtered is not None and filtered.poll() is None:
                filtered.kill()
                filtered.wait()
            if gzip.poll() is None:
                gzip.kill()
                gzip.wait()
            raise
        elapsed = time.perf_counter() - started
        gzip_log.seek(0)
        filter_log.seek(0)
        gzip_error = gzip_log.read().decode(errors="replace").strip()
        filter_error = filter_log.read().decode(errors="replace").strip()
    if filter_code:
        raise RuntimeError(f"ITCH filter failed ({filter_code}): {filter_error}")
    diagnostics = [line.lower() for line in gzip_error.splitlines() if line.strip()]
    # Apple's gzip adds a generic "uncompress failed" after the specific EOF
    # diagnostic. Accept it only alongside EOF, never for CRC or format errors.
    truncation = gzip_code in (1, 2) and any(
        "unexpected end of file" in line for line in diagnostics) and all(
        "unexpected end of file" in line or line.endswith(": uncompress failed")
        for line in diagnostics)
    if gzip_code and not truncation:
        raise RuntimeError(f"gzip failed ({gzip_code}): {gzip_error}")
    counts = json.loads((output / "filter_counts.json").read_text())
    if truncation and not counts["total_messages"]:
        raise RuntimeError("Truncated gzip contained no complete ITCH messages")
    return {"counts": counts, "elapsed_seconds": elapsed,
            "messages_per_second": counts["total_messages"] / elapsed,
            "gzip_returncode": gzip_code, "gzip_truncated": truncation,
            "gzip_stderr": gzip_error, "filter_stderr": filter_error}


def convert(source: Path, symbols: list[str], output: Path, *, keep_bin: bool = False) -> dict:
    """Run the deterministic pipeline; record provenance, counts and seconds."""
    started = time.perf_counter()
    day = input_date(source)
    source = source.resolve(strict=True)
    output = output.resolve() / day.isoformat()
    output.mkdir(parents=True, exist_ok=True)
    # A failed rerun must not leave a manifest claiming the new files completed.
    (output / "manifest.json").unlink(missing_ok=True)
    digest = hashlib.sha256()
    hashed_bytes = 0
    with source.open("rb") as stream:
        while hashed_bytes < 64 * 1024**2:
            chunk = stream.read(min(1024**2, 64 * 1024**2 - hashed_bytes))
            if not chunk:
                break
            digest.update(chunk)
            hashed_bytes += len(chunk)
    binary = compile_filter()
    print(f"Filtering {source.name}: {', '.join(symbols)}", flush=True)
    filtered = filter_input(binary, source, output, symbols)
    print(filtered["filter_stderr"], flush=True)
    if filtered["gzip_truncated"]:
        print(f"Accepted truncated gzip prefix: {filtered['gzip_stderr']}", flush=True)
    integrity = {}
    for symbol in symbols:
        print(f"Rebuilding {symbol} ({filtered['counts']['symbols'][symbol]['records']} records)", flush=True)
        integrity[symbol] = reconstruct(output / f"{symbol}.bin", output, symbol,
                                        observation_end_ns=filtered["counts"]["last_timestamp_ns"])
        entry = integrity[symbol]
        print(f"  {entry['emitted_rows']} session rows, {entry['elapsed_seconds']:.3f} s", flush=True)
    rebuild_seconds = sum(entry["elapsed_seconds"] for entry in integrity.values())
    rebuild_messages = sum(entry["input_records"] for entry in integrity.values())
    source_paths = [Path(__file__), ROOT / "tools/itch_filter.c", ROOT / "src/lob_latency/itch.py",
                    ROOT / "src/lob_latency/io_lobster.py"]
    manifest = {
        "input_path": str(source), "input_size_bytes": source.stat().st_size,
        "sha256_first_64_mib": digest.hexdigest(), "sha256_bytes": hashed_bytes,
        "date": day.isoformat(), "weekday": day.strftime("%A"), "symbols": symbols,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "command": "MPLCONFIGDIR=.cache/mpl " + shlex.join([str(ROOT / ".venv/bin/python"), *sys.argv]),
        "hardware": {"machine": platform.machine(), "processor": cpu_name(),
                     "platform": platform.platform(), "python": platform.python_version()},
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in source_paths},
        "seed": None, "filter": filtered, "integrity": integrity,
        "complete_input": not (filtered["gzip_truncated"] or filtered["counts"]["input_ended_mid_message"]),
        "rebuild_seconds": rebuild_seconds, "rebuild_messages": rebuild_messages,
        "rebuild_messages_per_second": rebuild_messages / rebuild_seconds if rebuild_seconds else None,
        "timing_model": {
            "filter_seconds_per_billion_input_messages": filtered["elapsed_seconds"] * 10**9
                                                        / filtered["counts"]["total_messages"]
                                                        if filtered["counts"]["total_messages"] else None,
            "rebuild_seconds_per_million_selected_messages": rebuild_seconds * 10**6 / rebuild_messages
                                                            if rebuild_messages else None,
            "includes_regular_session_parquet_writes": any(v["emitted_rows"] for v in integrity.values()),
            "full_day_extrapolation_verified": False,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "keep_bin": keep_bin,
    }
    if not keep_bin:
        for symbol in symbols:
            (output / f"{symbol}.bin").unlink()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {output / 'manifest.json'} ({manifest['elapsed_seconds']:.3f} s total)", flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--symbols", nargs="+", required=True, help="space- or comma-separated symbols")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--keep-bin", action="store_true")
    args = parser.parse_args()
    symbols = [symbol for group in args.symbols for symbol in group.split(',')]
    if (not all(re.fullmatch(r"[A-Za-z0-9._-]{1,8}", symbol) for symbol in symbols)
            or len(set(symbols)) != len(symbols)):
        parser.error("Symbols must be unique, nonempty ITCH symbols of at most eight characters")
    try:
        convert(args.input, symbols, args.out, keep_bin=args.keep_bin)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"ITCH conversion failed: {error}\n")


if __name__ == "__main__":
    main()
