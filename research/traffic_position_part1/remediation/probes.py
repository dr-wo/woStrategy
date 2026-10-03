"""Replay review regressions against original Part 1 or working remediation.

The before run uses immutable author-baseline commits; it never checks them out.
"""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import pytest

root=Path(__file__).resolve().parents[3]
base=root.parents[2]
repos={name:base/name/'.worktrees/traffic-position-part1' for name in ['woStrategy','woData','woPlanner']}
if '--before' in sys.argv:
    commits=json.loads(Path(__file__).with_name('baseline_commits.json').read_text())
    for repo,name,path in [
        ('woData','wodata.live.reducer','src/wodata/live/reducer.py'),
        ('woStrategy','wostrategy.analysis.traffic','src/wostrategy/analysis/traffic.py'),
        ('woPlanner','woplanner.analysis.service','src/woplanner/analysis/service.py'),
        ('woPlanner','woplanner.analysis.gui','src/woplanner/analysis/gui.py'),
    ]:
        importlib.import_module(name)
        source=subprocess.check_output(['git','show',commits[repo]+':'+path],cwd=repos[repo],text=True)
        spec=importlib.util.spec_from_file_location(name,repos[repo]/path)
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module
        exec(compile(source,str(repos[repo]/path)+' (author baseline)','exec'),module.__dict__)
        parent, attribute = name.rsplit('.', 1)
        setattr(importlib.import_module(parent), attribute, module)
        if name == 'wodata.live.reducer':
            import wodata.live
            wodata.live.LiveSessionStore = module.LiveSessionStore
    # Service imports may have loaded these before the frozen traffic module.
    for name in ['wostrategy.core.pre_race_session_data', 'wostrategy.core.telemetry_loader']:
        if name in sys.modules:
            importlib.reload(sys.modules[name])
args=[
 str(repos['woStrategy']/'tests/test_traffic_review_remediation.py'),
 str(repos['woStrategy']/'tests/test_position_traffic.py')+'::test_reducer_malformed_status_cannot_certify_complete_field',
 str(repos['woData']/'tests/test_live_telemetry_buffers.py')+'::test_unrecognised_position_status_remains_unknown',
 str(repos['woStrategy']/'tests/test_telemetry_loader.py')+'::test_load_or_cache_session_telemetry_refreshes_old_traffic_cache',
 str(repos['woStrategy']/'tests/test_pre_race_session_data.py')+'::test_loader_refreshes_old_enriched_traffic_cache',
 str(repos['woPlanner']/'tests/test_analysis_service.py')+'::test_quarantine_repeats_explicitly_and_retains_model_and_history',
 str(repos['woPlanner']/'tests/test_analysis_service.py')+'::test_quarantine_first_update_without_a_model_is_explicit',
 str(repos['woPlanner']/'tests/test_analysis_service.py')+'::test_quarantine_ui_explains_retention_without_publishing_history',
 str(repos['woPlanner']/'tests/test_analysis_service.py')+'::test_quarantine_rejects_prepared_automatic_traffic_rows',
 '-q', '--tb=short']
raise SystemExit(pytest.main(args))
