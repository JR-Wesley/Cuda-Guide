#!/usr/bin/env python3
"""Summarize CUPTI copy/kernel overlap inside this lab's NVTX case ranges."""
import sqlite3
import sys
from pathlib import Path


def merged(intervals):
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


def intersection(a, b):
    i = j = total = 0
    while i < len(a) and j < len(b):
        total += max(0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


if len(sys.argv) != 2:
    raise SystemExit("Usage: python3 analyze_trace.py timeline.sqlite")
connection = sqlite3.connect(Path(sys.argv[1]).resolve().as_uri() + "?mode=ro", uri=True)
ranges = connection.execute("""
    SELECT n.start,n.end,COALESCE(n.text,s.value)
    FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id
    WHERE n.end IS NOT NULL
""").fetchall()
found = set()
print("case,kernels,copies,kernel_streams,kernel_union_ms,copy_union_ms,copy_kernel_overlap_ms")
for start, end, name in ranges:
    if name not in {"pinned_one_stream", "pinned_three_streams", "pageable_three_streams"}:
        continue
    found.add(name)
    kernels = connection.execute(
        "SELECT start,end,streamId FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start>=? AND end<=?",
        (start, end)).fetchall()
    copies = connection.execute(
        "SELECT start,end FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE start>=? AND end<=?",
        (start, end)).fetchall()
    if not kernels or len(copies) != 2 * len(kernels):
        raise SystemExit(f"FAIL incomplete/unexpected trace for {name}: {len(kernels)} kernels, {len(copies)} copies")
    k = merged([(a, b) for a, b, _ in kernels])
    c = merged(copies)
    print(f"{name},{len(kernels)},{len(copies)},{len({s for _, _, s in kernels})},"
          f"{sum(b-a for a,b in k)/1e6:.6f},{sum(b-a for a,b in c)/1e6:.6f},"
          f"{intersection(k,c)/1e6:.6f}")
if len(found) != 3:
    raise SystemExit("FAIL missing pipeline NVTX case ranges")
print("PASS complete case ranges; unions avoid double counting; warmup included")
