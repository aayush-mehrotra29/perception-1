"""Sequential, bounded-memory target qualification. No model fitting or uploads."""
import argparse
import json
from pathlib import Path

from .edugage_targets import build_viewing, guard
from .worker_contract import file_sha256


def qualify(corpus: Path, output: Path):
    guard(output)
    output.mkdir(parents=True, exist_ok=False)
    paths = sorted(corpus.glob("participant-*/*.json"))
    paths = [p for p in paths if "arrays" in json.loads(p.read_text())]
    if not paths:
        raise ValueError("no recorded viewing artifacts found")
    rows = []
    for path in paths:
        guard(output)
        name = f"{path.parent.name}-{path.stem}"
        try:
            result = build_viewing(path, output / name, verify_raw=True)
            row = {k: result[k] for k in ("participant", "lecture", "supported_windows", "scheduled_windows", "raw_verification")}
            row.update(status="verified_aggregation" if result["raw_verification"]["performed"] else "no_supported_targets",
                       report=f"{name}/report.json", report_sha256=file_sha256(output / name / "report.json"))
        except (ValueError, KeyError, OSError, AssertionError) as error:
            row = {"status": "failed", "source": str(path), "error_type": type(error).__name__, "error": str(error)}
        rows.append(row)
        report = {"schema": "perception-corpus-target-qualification-v1", "expected_viewings": len(paths),
                  "completed_viewings": len(rows), "complete": len(rows) == len(paths), "viewings": rows,
                  "signal_quality_qualification_complete": False,
                  "physical_acquisition_timing_established": False,
                  "training_eligibility_established": False}
        temporary = output / "summary.tmp"
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(output / "summary.json")
        print(json.dumps(row, allow_nan=False), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    qualify(args.corpus, args.output)
