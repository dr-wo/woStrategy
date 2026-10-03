"""Read local R11 inputs; reproduce old behavior only in temporary directories.

Usage: python reproduce.py --data-root /path/to/woData --output /tmp/result.json
No source caches, saved models or historical evidence are changed.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import pandas as pd
from wostrategy.analysis import fp_tyre_evidence as current

ROOT = Path(__file__).resolve().parents[3]
BASE = "bf5e12cabf61135c3ce438659b3c7af879d793a1"


def frozen(name, relative):
    source = subprocess.check_output(["git", "show", f"{BASE}:{relative}"], cwd=ROOT, text=True)
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(source, str(ROOT / relative) + " (before)", "exec"), module.__dict__)
    return module


def measure(module, frames):
    try:
        result = module.analyse_fp_tyre_evidence_sessions(frames, year=2026, round_number=11)
        return {"accepted": True, "support": result["longrun_support"],
                "contrasts": len(result["degradation_contrasts"])}
    except ValueError as exc:
        return {"accepted": False, "error": str(exc)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    old_loader = frozen("wostrategy.core._fp_loader_before", "src/wostrategy/core/pre_race_session_data.py")
    old = frozen("wostrategy.analysis._fp_evidence_before", "src/wostrategy/analysis/fp_tyre_evidence.py")
    old.load_cached_weekend_sessions = old_loader.load_cached_weekend_sessions
    paths = {s: args.data_root / f"fastf1/session_laps/schema_v1/year=2026/round=11/session={s}/laps.pkl"
             for s in ("FP1", "FP2", "FP3")}
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths.values()}
    originals = {s: pd.read_pickle(p) for s, p in paths.items()}
    result = {"baseline_commit": BASE, "input_sha256": hashes,
              "input_attrs": {s: f.attrs for s, f in originals.items()}, "materialized": {}, "loader": {}}
    for version in ("unified-traffic-v2", "unified-traffic-v3", None):
        frames = {s: f.copy(deep=True) for s, f in originals.items()}
        for frame in frames.values():
            if version is None:
                frame.attrs.clear()
            else:
                frame.attrs["traffic_evaluator_version"] = version
        result["materialized"][str(version)] = {
            "before": measure(old, frames), "after": measure(current, frames),
        }
        if version is None:
            continue
        outcomes = {}
        for label, module in (("before", old), ("after", current)):
            with tempfile.TemporaryDirectory(prefix="fp-contract-") as directory:
                root = Path(directory)
                inputs = {s: f.copy(deep=True) for s, f in frames.items()}
                try:
                    calculated = module.calculate_fp_tyre_evidence(
                        year=2026, round_number=11, data_root=root,
                        session_loader=lambda session, **_: inputs[session],
                    )
                    outcomes[label] = {"accepted": True, "support": calculated["longrun_support"]}
                except ValueError as exc:
                    outcomes[label] = {"accepted": False, "error": str(exc)}
                cached = sorted(root.rglob("laps.pkl"))
                outcomes[label]["cache_count"] = len(cached)
                outcomes[label]["cached_attrs"] = [pd.read_pickle(p).attrs for p in cached]
                for path in cached:
                    session = path.parent.name.split("=")[1]
                    pd.testing.assert_frame_equal(pd.read_pickle(path), frames[session])
                outcomes[label]["cached_quantitative_contents_unchanged"] = bool(cached)
                outcomes[label]["evidence_written"] = bool(list(root.rglob("fp_tyre_evidence_v1.json")))
                if cached:
                    recheck = measure(current, {p.parent.name.split("=")[1]: pd.read_pickle(p) for p in cached})
                    assert not recheck["accepted"]
                    outcomes[label]["current_rejects_previously_mislabelled_cache"] = recheck
        result["loader"][version] = outcomes

    # A constructed current-contract fixture, never a relabelled historical cache.
    spec = importlib.util.spec_from_file_location("_contract_fixture", ROOT / "tests/test_fp_materialized_contract.py")
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    frames = {"FP2": fixture.current_laps()}
    kwargs = dict(year=2026, round_number=11, calculated_at=datetime(2026, 1, 1, tzinfo=timezone.utc))
    before = old.analyse_fp_tyre_evidence_sessions(frames, **kwargs)
    after = current.analyse_fp_tyre_evidence_sessions(frames, **kwargs)
    metadata = {"analysis_id", "input_fingerprint", "session_traffic_contracts"}
    assert {k: v for k, v in before.items() if k not in metadata} == {
        k: v for k, v in after.items() if k not in metadata}
    result["valid_v4"] = {"all_non_identity_payload_fields_equal": True,
                          "support": after["longrun_support"], "contrasts": len(after["degradation_contrasts"])}
    assert hashes == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths.values()}
    result["source_files_unchanged"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
