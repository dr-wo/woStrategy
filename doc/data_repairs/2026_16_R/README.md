# Reviewed physical tyre-life source repair: 2026 R16 race

The overlay fills **14** missing physical `TyreLife` cells and leaves **8 VER cells unresolved**. It runs at normal producer persistence (including cached producer reuse), after canonical numerical calculations. It never edits raw timing caches, derives physical prior use from model `TyreAgeLaps`, overrides existing non-null values, or changes the frozen BSF method. The JSON packaged at `src/wostrategy/analysis/known_clean_lap_repairs.json` contains exact driver/lap keys, physical values, derivations, source URLs and checksums. Each persisted clean CSV receives a `.repairs.json` sidecar. The captured files here were read independently from public services on 2026-10-06.

## Event and boundary checks

OpenF1 session 11731 / meeting 1308 and driver identities match the local 2026 round16 race: Bahrain GP relocated to Kuala Lumpur/Sepang on Oct4. Local raw `SessionInfo` has the same keys and F1 API path. Scheduled07UTC in OpenF1 is not the actual delayed race start; event identity was checked using the keys, official identity, drivers and race/pit boundaries rather than scheduled clock alone.

OpenF1 was investigated first. Its returned stint boundaries conflict with the local race and official F1 pit-stop summary, so its zeros were **not** accepted as physical anchors:

| Driver | OpenF1 returned affected stint | Actual replacement interval / official pit lap |
| --- | --- | --- |
| ALB23 | stint4 SOFT laps17–41, age0 | SOFT laps30–41 / pit29 |
| COL43 | stint3 HARD laps3–9; next MEDIUM10–55 | HARD laps32–42 / pit31 |
| LAW30 | stint3 HARD laps10–45, age0 | HARD laps31–45 / pit30 |
| VER3 | stint4 SOFT laps10–55, age0 | SOFT laps34–43 / pit33 |

Retained exact responses: `openf1_session.json`, `openf1_drivers.json`, `openf1_stints.json`. The public F1 `TimingAppData.jsonStream` and final `TimingAppData.json` are also retained verbatim (UTF-8 BOM included). The stream shows reindexing of stint entries around03:19–03:21; blindly joining indices or accepting a stale `StartLaps` would produce the wrong physical anchor.

## Independent physical anchors

[Pirelli's official race release](https://press.pirelli.com/sepang-debuts-the-intermediates-and-crowns-verstappen/) links the captured [pit-stop graphic](https://content.presspage.com/uploads/2363/fa648611-1484-4754-9b2a-30129e9af641/16-bh26-pitstop-en.jpg?10000). Its legend distinguishes solid NEW from striped USED sets. ALB's lap30 SOFT, COL's lap32 HARD and LAW's lap31 HARD are NEW. The final public F1 records independently agree: these three replacement records have `New=true, StartLaps=0`; TotalLaps respectively12,11,15 match the stint intervals. The [official F1 pit-stop summary](https://www.formula1.com/en/results/2026/races/1308/bahrain/pit-stop-summary) supplies pit29,31,30. Alpine's [race release](https://media.alpinecars.com/2026-formula-one-bahrain-grand-prix-in-malaysia-sunday/?lang=eng) explicitly identifies COL's lap31 stop for new hards.

With prior physical use **independently established as zero**, completed physical tyre life at race lap L is `0 + L - first_replacement_lap + 1`:

| Driver / local stint | Clean reference race laps | Repaired physical TyreLife |
| --- | --- | --- |
| ALB /4 SOFT |33,34,35,36,37,38 |4,5,6,7,8,9 |
| COL /3 HARD |38,39,40,41 |7,8,9,10 |
| LAW /3 HARD |32,33,34,35 |2,3,4,5 |
| VER /3 SOFT |35,36,37,38,39,40,41,42 |unresolved, unchanged null |

The local `FreshTyre=False` for ALB/COL does not establish physical prior use and is intentionally preserved. All other source values remain intact. `TyreAgeLaps` and `StintLapNumber` are context only, never the independent physical source.

## Why VER remains unresolved

Pirelli explicitly marks the lap34 replacement soft set USED. The final F1 entry instead reports `New=false, StartLaps=0, TotalLaps=10`; this zero conflicts with the independent USED evidence. The public stream introduces the relevant index2 record at03:29:53.749 (UNKNOWN, New=false, TotalLaps0, TyresNotChanged1), then SOFT/TyresNotChanged0 at03:30:28.767, without supplying a new explicit `StartLaps` for that set after reindexing. Therefore the inherited/final zero cannot establish prior physical use. At03:47:18.871 the **later** lap44 replacement, index3, explicitly supplies `StartLaps=3`; this belongs to a different tyre set and cannot repair lap35–42. No prior physical age for the target VER set was established.

## Reproducible audit and limited regeneration

`research/r16_tyre_life_repair_20261006/verify_r16.py` preserves the original170-row CSV, records the exact22 missing rows and existing surrounding known lives, checks all existing physical lives, all non-TyreLife raw CSV tokens and parsed columns, and compares the canonical saved-bank full-fit prediction/baselines/counts byte-for-byte before/after. It hashes all other canonical output files and the original Weekend run. Only14 null CSV cells changed; canonical numeric arrays and every other clean column are exactly identical.

The fresh diagnostic BSF output uses a separate cache and copies approved R1–R15 artifacts without regeneration. Current BSF remains **DEGRADED**: existing incomplete-input policy excludes8 unresolved VER rows (162/170 retained). That policy and numerical kernel were not changed by this repair. This is partial source recovery, not the requested fully supported BSF figure. Its graph has2 P /9 F /0 N,8 disconnected components,0 junction transfers, with only Mercedes/Williams projected support. The generated report states these restrictions. The original Weekend run and original BSF cache remain untouched.
