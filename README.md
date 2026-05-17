# woStrategy

Formula 1 strategy analysis tools built on FastF1.

This repository contains the strategy-analysis code that was previously living
inside a Fast-F1 fork: session wrappers, long-stint preparation, pre-season test
analysis, and plotting utilities.

## Install

```bash
python -m pip install -e .
```

The package depends on `fastf1`. Your installed FastF1 version determines which
events and telemetry formats are supported.

## Usage

```python
from wostrategy import Session, run_two_day_benchmark_race_sim

session = Session(2026, 2, 3, test=True)

result = run_two_day_benchmark_race_sim(
    year=2026,
    round_number=2,
    benchmark_session=2,
    comparison_session=3,
    output_prefix="temp/r2s2_r2s3",
    min_laps=30,
    reference_laps=57,
    test=True,
)
```

You can also run the pre-season analysis example with:

```bash
python -m wostrategy.script.pre_season_analysis
```
