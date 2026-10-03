# FP Performance R&D — canonical re-entry index

核查日期：2026-10-03。这是研究入口与关闭状态索引；不替代原 REPORT、protocol 或冻结结果，不授权新实验或生产接入。结论以原报告的 target、support 和信息集为准。

## Current working takeaway

- 普通周末 FP2 raw long-run 可作为粗粒度 relative-performance **幅值证据**；相同事件/队对上的 FP2→Race gap 比 FP1 更接近。[Transfer §2–3][transfer]
- **近邻精确排序未建立**；区分悬殊对手的 ordering 不能代替 close-rival precision。[Transfer §5][transfer]
- Physical pit segmentation 与 interruption-tolerant S1 恢复了真实-looking programmes；label 仍不是已知 intent、等载油或 race simulation 的证明。[Selector §1–7][selector]
- 已测 universal correction、历史 programme habit 和 field-scale 路线没有建立可迁移增益；暂停通用修正，不把局部收益扩大成规则。[Correction §4–8][correction]、[Discovery §2–4][discovery]
- Fuel/load physical inference 尚未建立：Race fuel 是剩余圈 proxy 坐标，不是独立 s/kg；common-load 重建是条件性 retro 情景。[Common-load][common]
- Sprint 结束后的 raw pace 在 2025/2026 均提供 Race gap 的增量信息；固定 blend 未一致胜过 Sprint-only。[Transfer §4][transfer]
- “2025 普遍比 2026 更稳定”没有稳健支持，更没有识别规则成熟度或 setup 的因果作用。[Transfer §6–8][transfer]
- FP fast-lap + 过去 Q1 是值得保留的经验对照；target 审计后仍有方向性证据，尚非独立 prospective 确认。[Expanded §5][expanded]
- FP3→Q1 的已测天气中心/区间规则未提供重复增益；无天气 rolling blend 仅保留为对照。[Environment §4–8][environment]
- Race retro 是有误差的模型目标；改进其质量与可比性是 reopen correction 的重要依赖，不能把其他方法输给 raw 当作 raw 已足够准确。

## Research lines

状态：**RETAINED**＝保留有边界的用途/基准；**PAUSED**＝当前不继续路线；**CLOSED_FOR_NOW**＝已测批次结束；**NEEDS_NEW_EVIDENCE**＝尚不足以成立。触发条件由已有报告的限制/后续条件整理，不构成新 protocol；同线多个 branches 合并。

Entry point 的路径按对应 commit 的 repo root 解释；REPORT 链接指向本机保留的工作区。前八份研究 REPORT 已逐项核对与所列冻结 commit 字节一致。

| Research line | Status | Current strongest conclusion | Important boundary | Revisit trigger | Entry point |
|---|---|---|---|---|---|
| Historical Team Tyre Prior（team-specific RD1/RD2/RD3 来源待恢复） | NEEDS_NEW_EVIDENCE | 本次未定位到该既往 team×compound 历史 prior 研究的可恢复 REPORT、commit 或 artifacts，不能确认它们已持久化，也不能裁定其科学结果。另有 global P0/D0 baseline 文档，但不能替代团队 prior 研究。 | Global baseline 明确排除 team variation；找不到材料不等于未做过研究或研究失败。 | 先恢复原 RD1/RD2/RD3 报告、season-window/support/walk-forward 产物及冻结身份；随后按原报告触发条件复入，不在此新设验证结论。 | Team-specific commit/REPORT：未定位；`5bda17f` · [doc/CROSS_EVENT_TYRE_PREDICTION_V1.md][tyre] 仅供 global baseline 语义参考 |
| FP long-run programme / selector / repeatability | RETAINED | Physical containers + S1/S2 恢复遗漏长跑；共同支持的 run pace 相对稳定。 | 人工 programme judgement 不是 ground truth；共同 run 稳定不等于跨 session gap 或 fuel 稳定。 | 新 FP programme 审阅样本，或相容 compound/age/time 的跨 session 与双车重复证据。 | `0901138` · [research/fp_longrun_selector_exploration/REPORT.md][selector]；repeatability 延伸 `c23392a` · [REPORT][repeatability] |
| FP long-run→Race universal correction / fuel-related work | PAUSED | Reference-specific pace 可追溯；局部进度对齐未转化成稳定历史预测增益，已测 discovery 候选未建立通用规则。 | Canonical BSL 不是无误差真值；残差/fuel proxy 不识别 starting load，oracle correction 不等于预测。 | Materially improved Race retro 或独立 FP tyre/thermal/track reference；物理 fuel 解释另需独立 mass sensitivity、burn calibration 和 programme/setup 证据。 | `c6d31a5` · [research/fp_programme_corrections_rnd/REPORT.md][correction]；`87f65c6` · [research/fp_race_correction_discovery/REPORT.md][discovery] |
| Common-load scenario / performance reconstruction | CLOSED_FOR_NOW | 同 programme 共同起始载油项相消；已算 tyre/time/burn 情景未稳定胜 raw，仅保留条件性进度线索。 | 使用同站 Race joint 参数，属 retro/oracle-assisted；长短 programme 的绝对 pace 不可直接拼接。 | 新 event block，且 FP 当时可用的参数/独立 physical calibration 足以检验冻结线索。 | `4c1c974` · [research/fp_common_load_scenario/REPORT.md][common]；`inputs/bank_registry.json` |
| Raw FP long-run→Race evidence | RETAINED | 早期局部 ordering evidence 与后续 FP2 粗幅值证据可保留；未建立精确 Race gap。 | 旧 geometry/BSL 表与新 whole-Race matched target/support 不同；near-rival precision 不成立。 | 新 prospective ordinary GP：先固定 raw/FP2 用法，随后在同 target、同 pairs 与覆盖账本上评分。 | `532cfc2` · [research/fp_race_2025_transfer/REPORT.md][transfer] §2–3、5；旧几何 `afd4213` · [REPORT][geometry]、expanded `33b9e72` · [REPORT][expanded] |
| FP fast-lap / Q1 evidence | RETAINED | 当次 FP 快圈与历史 Q1 融合有经验增量方向；FP3 sector shape 仅为较弱线索。 | Q1 target 审计是事后敏感性，不升级为新确认；fresh/低胎龄不保证 push。 | 新普通周末的冻结 FP/Q1 预测，配可靠的 Q1 validity/deletion/driver support；sector 需独立新站。 | `33b9e72` · [research/fp_performance_expanded/REPORT.md][expanded] §5–6；关键 `q_sector/FINDINGS.md` |
| FP3→Q1 environmental exploration | CLOSED_FOR_NOW | 已测 signed/absolute-weather 中心及天气区间规则没有重复收益；raw 与无天气 rolling blend 保留。 | Target-selected Q1 实测天气是 retrospective；三站不确认 blend 稳定优胜或区间校准。 | 真正未参与选择的新 FP3/Q1 周末；天气预测主张另需有发布时间的预测天气，而非后来的实测天气。 | `b251bbd` · [research/fp3_q1_environment/REPORT.md][environment]；`operational.py` |
| Sprint→Race relationship | RETAINED | 共同三节点支持上，2025/2026 Sprint 相对 FP1 重复提供 Race gap 幅值增量。 | 仅适用于 Sprint weekend；跨节点不保证同胎/同 driver/free air，不能推出 setup 因果或精确近邻排名。 | 新 Sprint event 可对冻结 raw/Sprint 用法评分，并核对 compound/phase/driver 支持。 | `532cfc2` · [research/fp_race_2025_transfer/REPORT.md][transfer] §4；原五站 `16526ce` · [research/fp_sprint_race/REPORT.md][sprint] |
| 2025 full-season FP→Race transfer | RETAINED | 完整 acquisition 与适用事件复验支持普通 FP2 幅值用途，并重复 Sprint 增量。 | 24 events 不等于每节可评分；2026-derived 方法用于历史 2025，不是 2025 as-run 或自动 blind。 | 新未参与开发的 ordinary/Sprint event block，沿用已冻结 selector、target 与 abstention。 | `532cfc2` · [research/fp_race_2025_transfer/REPORT.md][transfer]；`run_analysis.py`、`outputs/scores.csv.gz` |
| 2025 vs 2026 comparison | NEEDS_NEW_EVIDENCE | 没有稳健的普遍 2025 稳定性优势；ordering 与 magnitude 不同向，matched 差异依赖 Monaco/赛季阶段。 | 2026 仅前15站；同 C-number 不代表跨年度同物理轮胎。 | 新未参与开发的 matched event blocks，具可比 format、赛季阶段、队对与 tyre/reference context。 | `532cfc2` · [research/fp_race_2025_transfer/REPORT.md][transfer] §6–8；`outputs/year_context_cohorts.csv` |

表面冲突的读取顺序：先核对 **target → support → reference/summary → information set**。例如原 Sprint 的 BSL 是较早 bank；最新研究统一重算 B，并与 observed A 分列。旧 geometry 的局部排序比例不能与全季 observed target 的排序比例直接相减。不要平均这些数字或重新裁决旧冻结结论。旧 STOP gates 只约束其原 protocol，不自动成为所有后续研究的门槛。

## Reopen map

- IF improved Race retro / independent FP tyre-history reference → reopen **universal correction** → `c6d31a5` [Correction][correction]；候选失败记录从 `87f65c6` [Discovery][discovery] 恢复。
- IF new Sprint weekend → reopen **frozen Sprint validation** → `532cfc2` [Transfer §4][transfer]。
- IF new ordinary GP → reopen **prospective FP2 raw evaluation** → `532cfc2` [Transfer §2–3][transfer]。
- IF 原 team-specific RD1/RD2/RD3 报告/冻结产物被恢复 → reopen **Historical Team Tyre Prior** 的入口核对 → 从恢复出的真实 commit/REPORT 开始；`5bda17f` [global baseline][tyre] 不能替代它。
- IF new FP3/Q1 weekend (+ archived forecast for weather claims) → reopen **fast-lap/environment validation** → `b251bbd` [Environment][environment]，保留 `33b9e72` 的 Q target 审计。
- IF new comparable cross-year blocks → reopen **year comparison** → `532cfc2` [Transfer §6–8][transfer]。

Historical Team Tyre Prior 搜索记录：检查 woStrategy/woData/woPlanner/woWeekend 的可访问 Git refs、reflogs 与历史文件名，以及研究目录、现存 worktrees、本地 temp 材料。找到的 `rd3_design_gates_20261001`、`rd3-prior-semantics` 和 `rd3_shadow` 涉及 Race MC/net-slope/prior 语义，不是所指 team×compound 历史 tyre-prior RD1/RD2/RD3。上述团队研究的报告/提交/产物在本次搜索范围内未定位；未据此推断其未曾存在，也未借用无关 RD3 commit。

## Technical archive

本次重新核查三条研究 branch 的 HEAD 与 `git status --short`：均为空，tracked/staged diff 均为空。主 checkout 的 README/BSF 等并发改动未处理。只建议归档聊天，不删除 worktree/cache，不 push/merge。

以下 `W`＝`/Users/zhxutong/dr-wo/woStrategy/research/.worktrees`，`D`＝`/Users/zhxutong/dr-wo/woData`；路径均为本机实际持久位置。

| Agent/research branch | Final commit | Worktree clean? | Non-git persistent artifacts | Archive recommendation |
|---|---|---|---|---|
| `codex/fp-programme-corrections-rnd` | `c6d31a5350f0dfd4b605af7c74b626270cb2a2c8` | Yes | 无必须另外保留的本轮产物；`joint_v1/`、`convergence_v1/` 已提交。保留 COMMANDS 中外部来源。 | SAFE_TO_ARCHIVE |
| `codex/fp-common-load-scenario-rnd` | `4c1c97405bf3206679e81b7be82e7e56247aa3a6` | Yes | `W/fp-common-load-scenario-rnd/research/fp_common_load_scenario/` 的 `inputs/`（45 MiB）、`outputs/`（148 MiB）、`outputs_2048/` + `outputs_bank80k/`（约61 MiB）、两份 pair draws（约71 MiB）；保留 bank_registry 指向的 expanded worktree。 | SAFE_TO_ARCHIVE；上述目录不可随工作区清理 |
| `codex/fp-race-2025-transfer-rnd` | `532cfc204ac8c0b2e194b3b6d418e8cf53286662` | Yes | `W/fp-race-2025-transfer-rnd/research/fp_race_2025_transfer/banks/`（382 MiB，含修前 Miami）、`cache/`（1.2 GiB）；`D/wodata/research_session_inputs/schema_v1/fp_race_2025_transfer_20261003/`（54 MiB，含 supplements）。 | SAFE_TO_ARCHIVE；上述目录不可随工作区清理 |

依赖保全：common-load 的 bank_registry 实际指向 `W/fp-performance-expanded-exploration/`；其 joint banks 必须保留。Sprint 独立输入/缓存/banks 的路径与可恢复紧凑证据见 [DATA_CHANGES][sprint-data]。环境探索新增18个 telemetry parser 文件（约446 MB）位于 `D/fastf1/raw_cache/schema_v1/`；当时修改共享 HTTP SQLite 前未留 before-image，事实已记录于 [DATA_CHANGES][environment-data]。最新 transfer 也记录了独立 HTTP 副本被清空、只有事后 snapshot、随后从完整共享源重新播种的历史；见 [REPRODUCE][reproduce]。不在本次补救或重跑。

三个 closure handoff 均报告 `No material session-only knowledge remains.`；其重要边界与保全历史可在原文档定位。归档聊天无已发现的技术阻碍，但 Git 不能单独恢复上述忽略产物。本索引以包含它的 Git commit 为恢复入口；归档建议不授权删除或清理任何含非 Git 产物的工作区。

[tyre]: ../doc/CROSS_EVENT_TYRE_PREDICTION_V1.md
[selector]: .worktrees/fp-race-2025-transfer-rnd/research/fp_longrun_selector_exploration/REPORT.md
[repeatability]: .worktrees/fp-race-2025-transfer-rnd/research/fp_longrun_same_session_repeatability_rnd/REPORT.md
[correction]: .worktrees/fp-programme-corrections-rnd/research/fp_programme_corrections_rnd/REPORT.md
[discovery]: .worktrees/fp-race-correction-discovery/research/fp_race_correction_discovery/REPORT.md
[common]: .worktrees/fp-common-load-scenario-rnd/research/fp_common_load_scenario/REPORT.md
[geometry]: .worktrees/fp-race-2025-transfer-rnd/research/fp_race_relative_geometry_rnd/REPORT.md
[expanded]: .worktrees/fp-performance-expanded-exploration/research/fp_performance_expanded/REPORT.md
[environment]: .worktrees/fp3-q1-environment-rnd/research/fp3_q1_environment/REPORT.md
[sprint]: .worktrees/fp-sprint-race-relationship-rnd/research/fp_sprint_race/REPORT.md
[transfer]: .worktrees/fp-race-2025-transfer-rnd/research/fp_race_2025_transfer/REPORT.md
[sprint-data]: .worktrees/fp-sprint-race-relationship-rnd/research/fp_sprint_race/DATA_CHANGES.md
[environment-data]: .worktrees/fp3-q1-environment-rnd/research/fp3_q1_environment/DATA_CHANGES.md
[reproduce]: .worktrees/fp-race-2025-transfer-rnd/research/fp_race_2025_transfer/REPRODUCE.md
