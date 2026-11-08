import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

"""Offline inspection of existing learning-run artifacts (stdlib only)."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys


BACKENDS = ("triton", "gluon", "cute", "cutile")
MAX_CUBINS = 20
TOOL = os.environ.get("CUOBJDUMP", "cuobjdump")
TIMEOUT = 60
POLICY = (
    "PASS cases only; Triton gemm/transpose/fusion; Gluon transpose/copy; "
    "CuTe transpose/shared_copy/gemm; cuTile all ops. Group by backend/op/config "
    "(excluding plan_ids), across supplied runs. Largest shape product per group; "
    "ties use run path/evidence lexically. Keep every artifact of each selected case. "
    "Reject selections above 20 cubins; never truncate a fusion pair."
)
LIMITS = [
    "Offline only: no CUDA imports, compilation, JIT, GPU queries, launches or profiling.",
    "Static instruction occurrences are not dynamic execution counts or performance measurements.",
    "Missing PTX/SASS/resources are unavailable, not zero. No occupancy/cache/DRAM claims.",
    "Entry/function counts describe exported code, not observed runtime launches.",
    "cuTile TileIR is a separate frontend export, not captured JIT machine code; identity unverified.",
    "Selected shapes/configurations only; no exhaustive boundary or synchronization proof.",
    "nvdisasm is not invoked; SASS and resource evidence use cuobjdump only.",
]
PTX_INTEREST = ("mma.", "ldmatrix.", "cp.async", "ld.global", "st.global",
                "ld.shared", "st.shared", "bar.", "shfl.")
SASS_INTEREST = ("HMMA", "IMMA", "MMA", "LDG", "STG", "LDS", "STS", "LDSM",
                 "LDGSTS", "LDGDEPBAR", "DEPBAR", "BAR", "SHFL")


def utc():
    return datetime.now(timezone.utc).isoformat()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def relative(path, base):
    return Path(os.path.relpath(path, base)).as_posix()


def contained(base, name):
    path = (base / name).resolve()
    if not path.is_relative_to(base.resolve()):
        raise ValueError(f"Path escapes evidence root: {name}")
    return path


def wanted(backend, case):
    op = case["op"]
    return (backend == "cutile"
            or backend == "triton" and op in ("gemm", "transpose", "fusion")
            or backend == "gluon" and op in ("transpose", "copy")
            or backend == "cute" and (op in ("transpose", "gemm")
                                      or case["config"].get("kind") == "shared_copy"))


def discover(roots):
    manifests = {}
    for root in roots:
        root = root.resolve(strict=True)
        found = [root / b / "results.json" for b in BACKENDS
                 if (root / b / "results.json").is_file()]
        if not found:
            raise ValueError(f"No backend results.json under run root: {root}")
        for path in found:
            manifests[path] = json.loads(path.read_text(encoding="utf-8"))
    groups, inventory = {}, []
    for path, result in sorted(manifests.items()):
        backend = path.parent.name
        for case in result["cases"]:
            config = {k: v for k, v in case["config"].items() if k != "plan_ids"}
            key = (backend, case["op"], json.dumps(config, sort_keys=True))
            item = {"manifest": path, "backend": backend, "case": case, "key": key}
            inventory.append(item)
            if case["status"] == "PASS" and wanted(backend, case):
                groups.setdefault(key, []).append(item)
    selected = []
    for key, items in sorted(groups.items()):
        items.sort(key=lambda x: (-math.prod(x["case"]["shape"]),
                                 str(x["manifest"]), x["case"]["evidence"]))
        selected.append(items[0])
    count = 0
    for item in selected:
        case_dir = contained(item["manifest"].parent, item["case"]["evidence"])
        artifacts = case_dir / "artifacts"
        if not artifacts.is_dir():
            raise ValueError(f"Missing case artifacts directory: {artifacts}")
        item["artifacts"] = artifacts
        count += len(list(artifacts.rglob("*.cubin")))
    if count > MAX_CUBINS:
        raise ValueError(f"Selection requires {count} cubins; hard limit is {MAX_CUBINS}")
    if not selected:
        raise ValueError("No representative PASS cases found")
    return manifests, inventory, selected


def inspect_text(text, kind):
    """Extract actual instruction lines, excluding comments/directives and encodings."""
    counts, hits, waits = Counter(), [], Counter()
    if kind == "ptx":
        clean = re.sub(r"/\*.*?\*/", lambda m: "\n" * m[0].count("\n"), text, flags=re.S)
        clean = re.sub(r"//[^\n]*", "", clean)
        # Inline PTX may put the opcode alone on a line and operands on the next.
        pattern = re.compile(r"^\s*(?:@!?%\w+\s+)?([a-z][a-z0-9_.]*)(?=[\s;]|$)")
        entries = re.findall(r"\.entry\s+([^\s(]+)", clean)
        targets = re.findall(r"^\s*\.target\s+(.+)$", clean, re.M)
    else:
        clean = text
        pattern = re.compile(r"^\s*/\*[0-9a-fA-F]+\*/\s*(?:@!?[A-Z0-9]+\s+)?([A-Z][A-Z0-9_.]*)\b")
        entries = re.findall(r"Function\s*:\s*(\S+)", text)
        targets = re.findall(r"^\s*arch\s*=\s*(\S+)", text, re.M)
    originals = text.splitlines()
    for number, line in enumerate(clean.splitlines(), 1):
        match = pattern.match(line)
        if not match:
            continue
        opcode = match[1]
        counts[opcode] += 1
        interesting = (opcode.startswith(PTX_INTEREST) if kind == "ptx"
                       else opcode.split(".")[0] in SASS_INTEREST)
        if interesting:
            hits.append({"line": number, "opcode": opcode, "text": originals[number - 1].strip()})
        if kind == "ptx" and opcode == "cp.async.wait_group":
            operand = re.search(r"cp\.async\.wait_group\s+(\d+)\s*;", line)
            if operand:
                waits[operand[1]] += 1
    return {"entries_or_functions": entries, "targets": sorted(set(targets)),
            "instruction_counts": dict(sorted(counts.items())),
            "interesting_lines": hits, "cp_async_wait_group_operands": dict(waits)}


class Evidence:
    def __init__(self, output):
        self.output = output
        self.files = []
        self.commands = []
        self.errors = []

    def input_file(self, source, original_base):
        record = {"original_root": str(original_base),
                  "original_relative_path": relative(source, original_base),
                  "input_path": str(source.resolve()), "bytes": source.stat().st_size}
        self.files.append(record)
        return record

    def command(self, flags, input_record=None):
        index = len(self.commands)
        prefix = f"commands/{index:03d}"
        argv = [TOOL, *flags]
        if input_record:
            argv.append(input_record["input_path"])
        record = {"argv": argv, "cwd": str(self.output), "started_utc": utc(),
                  "timeout_seconds": TIMEOUT, "input": input_record,
                  "raw_log": prefix + ".log", "returncode": None, "status": "started"}
        path = self.output / (prefix + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        dump(path, record)
        with (self.output / record["raw_log"]).open("wb") as log:
            try:
                process = subprocess.Popen(argv, cwd=self.output, stdout=log, stderr=subprocess.STDOUT)
                try:
                    record["returncode"] = process.wait(timeout=TIMEOUT)
                    record["status"] = "ok" if process.returncode == 0 else "failed"
                except subprocess.TimeoutExpired:
                    process.kill()
                    record["returncode"] = process.wait()
                    record["status"] = "timeout"
            except OSError as exc:
                record["status"] = "unavailable"
                record["error"] = str(exc)
                log.write((str(exc) + "\n").encode())
        record["finished_utc"] = utc()
        dump(path, record)
        self.commands.append(record)
        if record["status"] != "ok":
            self.errors.append(f"Command {index}: {record['status']}; see {record['raw_log']}")
        return record


def markdown(report):
    lines = ["# Offline static evidence", "", report["selection_policy"], "",
             "Counts: `" + json.dumps(report["counts"], sort_keys=True) + "`", "",
             "[Full summary](summary.json) · [Input paths](files.json) · "
             "[Commands](commands.json)", "",
             "## Limits", "", *["- " + x for x in LIMITS], "",
             "## Selected evidence", ""]
    for case in report["cases"]:
        lines += [f"### {case['label']}: {case['id']}", "",
                  f"- Backend/op/shape: `{case['backend']}/{case['op']}` `{case['shape']}`",
                  f"- Config: `{json.dumps(case['config'], sort_keys=True)}`",
                  f"- Original evidence (relative to input run): `{case['original_evidence']}`",
                  f"- Input run: `{case['run_root']}`",
                  f"- PTX: **{case['ptx_status']}**; SASS: **{case['sass_status']}**.",
                  f"- Exported PTX entries / SASS functions: {case['ptx_entries']} / {case['sass_functions']}."]
        if case["backend"] == "cutile":
            lines += ["- Separate TileIR export only; PTX/SASS/resources missing. No machine-instruction inference."]
        for artifact in case["artifacts"]:
            path = artifact["file"]["input_path"]
            lines.append(f"- [{Path(path).name}]({path})")
            for kind in ("ptx", "sass"):
                if kind not in artifact:
                    continue
                facts = artifact[kind]
                counts = Counter(hit["opcode"] for hit in facts["interesting_lines"])
                lines.append(f"  - {kind.upper()} matched instruction occurrences: `{dict(counts)}`")
                if facts["cp_async_wait_group_operands"]:
                    lines.append(f"  - PTX wait_group operands: `{facts['cp_async_wait_group_operands']}`")
            for command in artifact.get("commands", []):
                log = command["raw_log"]
                lines.append(f"  - [{command['argv'][1]} raw log]({log}), rc={command['returncode']}, {command['status']}")
            if artifact.get("resource_lines"):
                lines += ["  - Actual cuobjdump resource output:", "", "```text",
                          *[x["text"] for x in artifact["resource_lines"]], "```", ""]
        lines.append("")
    lines += ["## Errors", "", *report["errors"]] if report["errors"] else ["## Errors", "", "None."]
    return "\n".join(lines) + "\n"


def collect(roots, output, argv):
    manifests, inventory, selected = discover(roots)
    ev = Evidence(output)
    run_ids = {path.parent.parent for path in manifests}
    ev.command(["--version"])
    cases = []
    for index, item in enumerate(selected):
        original = item["case"]
        root = item["manifest"].parent.parent
        label = f"{index:02d}-{item['backend']}-{original['op']}"
        case = {"label": label, "backend": item["backend"],
                **{k: original[k] for k in ("id", "op", "shape", "config")},
                "run_root": str(root), "original_evidence": relative(item["artifacts"].parent, root),
                "historical_launches_per_call": original.get("metadata", {}).get("launches_per_call"),
                "artifacts": []}
        for source in sorted(item["artifacts"].rglob("*")):
            if not source.is_file():
                continue
            record = ev.input_file(source, root)
            artifact = {"file": record}
            if source.suffix == ".ptx":
                artifact["ptx"] = inspect_text(source.read_text(), "ptx")
                if not artifact["ptx"]["entries_or_functions"] or not artifact["ptx"]["instruction_counts"]:
                    ev.errors.append(f"PTX parse produced no entries/instructions: {record['input_path']}")
            elif source.suffix == ".cubin":
                sass = ev.command(["--dump-sass"], record)
                resources = ev.command(["--dump-resource-usage"], record)
                artifact["commands"] = [sass, resources]
                if sass["status"] == "ok":
                    artifact["sass"] = inspect_text((output / sass["raw_log"]).read_text(errors="replace"), "sass")
                    if not artifact["sass"]["instruction_counts"] or not artifact["sass"]["entries_or_functions"]:
                        ev.errors.append(f"SASS parse produced no functions/instructions: {sass['raw_log']}")
                if resources["status"] == "ok":
                    artifact["resource_lines"] = [
                        {"line": i, "text": line.strip()}
                        for i, line in enumerate((output / resources["raw_log"]).read_text(errors="replace").splitlines(), 1)
                        if line.strip()]
            case["artifacts"].append(artifact)
        for kind in ("ptx", "sass"):
            parsed = [a[kind] for a in case["artifacts"] if kind in a]
            case[kind + "_status"] = "available" if parsed else "missing"
            case["ptx_entries" if kind == "ptx" else "sass_functions"] = (
                sum(len(p["entries_or_functions"]) for p in parsed) if parsed else None)
            if not parsed and case["backend"] != "cutile":
                ev.errors.append(f"{label}: {kind.upper()} missing")
        cases.append(case)
    selected_keys = {(x["manifest"], x["case"]["evidence"]) for x in selected}
    selection = [{"run_root": str(x["manifest"].parent.parent), "backend": x["backend"],
                  "evidence": x["case"]["evidence"], "id": x["case"]["id"],
                  "selected": (x["manifest"], x["case"]["evidence"]) in selected_keys,
                  "reason": ("representative" if (x["manifest"], x["case"]["evidence"]) in selected_keys
                             else "non-PASS, outside focus, or another shape in same config group")}
                 for x in inventory]
    counts = {"input_runs": len(run_ids), "input_cases": len(inventory), "selected_cases": len(cases),
              "cubins": sum(Path(f["input_path"]).suffix == ".cubin" for f in ev.files),
              "ptx": sum(Path(f["input_path"]).suffix == ".ptx" for f in ev.files),
              "tileir": sum(Path(f["input_path"]).suffix == ".tileirbc" for f in ev.files),
              "commands": len(ev.commands), "failed_commands": sum(c["status"] != "ok" for c in ev.commands),
              "input_files": len(ev.files), "errors": len(ev.errors)}
    report = {"schema_version": 1, "mode": "offline-static", "finished_utc": utc(),
              "argv": argv, "cwd": str(Path.cwd()), "output": str(output),
              "selection_policy": POLICY, "cubins_limit": MAX_CUBINS, "counts": counts,
              "limits": LIMITS, "selection": selection, "cases": cases, "errors": ev.errors}
    dump(output / "files.json", ev.files)
    dump(output / "commands.json", ev.commands)
    dump(output / "summary.json", report)
    (output / "SUMMARY.md").write_text(markdown(report), encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="existing run roots containing backend/results.json")
    args = parser.parse_args(argv)
    from command_support import Record
    output = Record("static").path
    invocation = [sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)]
    dump(output / "invocation.json", {"argv": invocation, "cwd": str(Path.cwd()), "started_utc": utc()})
    try:
        report = collect(args.runs, output, invocation)
        rc = 1 if report["errors"] else 0
        print(json.dumps({"output": str(output), "counts": report["counts"], "returncode": rc}))
    except Exception as exc:
        dump(output / "failure.json", {"type": type(exc).__name__, "message": str(exc)})
        print(f"Static collection failed; evidence preserved at {output}: {exc}", file=sys.stderr)
        rc = 1
    dump(output / "exit.json", {"returncode": rc, "finished_utc": utc()})
    return rc


if __name__ == "__main__":
    sys.exit(main())
