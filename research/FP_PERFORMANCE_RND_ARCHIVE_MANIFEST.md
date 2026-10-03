# FP Performance R&D archive manifest

2026-10-03；只做仓库保全/清理，不重跑实验、不改 frozen commits/REPORT、不改 production、不 push/merge/gc。此文件与 canonical index 是唯一 cleanup docs。

## Permanent Git preservation

- Annotated `archive/fp-rnd/transfer-20261003` → `532cfc204ac8c0b2e194b3b6d418e8cf53286662`。
- Annotated `archive/fp-rnd/race-quali-20261003` → `d9da699e32a79a371225f5e0deb691b6608cd4f5`。
- 表中每个 HEAD 均经 `cat-file` 与 ancestor gate 核验。索引引用的 `5bda17f`、`0901138`、`c23392a`、`c6d31a5`、`87f65c6`、`33b9e72`、`afd4213`、`b251bbd`、`4c1c974`、`16526ce`、`532cfc2` 均由 transfer tag 到达；不为每个 branch 另造 tag。

## Safety inventory and removal record

每行 HEAD 是该 branch 最后 scientific checkpoint；clean 表示 tracked、staged、untracked 均为空。RETAINED 科学条目仍保留，其已完成的旧 research worktree 不代表 active development。

| Candidate branch | Exact HEAD | Before state / ignored evidence | Preserving ref | Cleanup outcome |
|---|---|---|---|---|
| `codex/fp-common-load-scenario-rnd` | `4c1c97405bf3206679e81b7be82e7e56247aa3a6` | clean；134 ignored，132 material / 368530738 bytes | `archive/fp-rnd/transfer-20261003` | 正常 remove 成功；local branch deleted -D |
| `codex/fp-performance-expanded-exploration` | `33b9e72c8b6c34099b7f15c3087f652969c0482b` | clean；140 ignored，26 material / 132422189 bytes | `archive/fp-rnd/transfer-20261003` | 正常 remove 成功；local branch deleted -D |
| `codex/fp-programme-corrections-rnd` | `c6d31a5350f0dfd4b605af7c74b626270cb2a2c8` | clean；4 ignored，0 material / 0 bytes | `archive/fp-rnd/transfer-20261003` | 正常 remove 成功；local branch deleted -D |
| `codex/fp-race-2025-transfer-rnd` | `532cfc204ac8c0b2e194b3b6d418e8cf53286662` | clean；1396 ignored，1320 material / 1730254804 bytes | `archive/fp-rnd/transfer-20261003` | 正常 remove 成功；local branch deleted -D |
| `codex/fp-race-correction-discovery` | `87f65c612d12440d54c3fe0d049fded408b0a756` | clean；3 ignored，0 material / 0 bytes | `archive/fp-rnd/transfer-20261003` | 正常 remove 成功；local branch deleted -D |
| `codex/fp-sprint-race-relationship-rnd` | `16526ce7904f7f4b2e690089ff49237c5a9819d8` | clean；257 ignored，195 material / 893408205 bytes | `archive/fp-rnd/transfer-20261003` | 正常 remove 成功；local branch deleted -D |
| `codex/fp3-q1-environment-rnd` | `b251bbdafe2da6a7b38871a63b5e645acb31ceaa` | clean；19 ignored，8 material / 55459727 bytes | `archive/fp-rnd/transfer-20261003` | 正常 remove 成功；local branch deleted -D |
| `codex/fp-race-quali-evidence-rnd` | `d9da699e32a79a371225f5e0deb691b6608cd4f5` | clean；63 ignored，0 material / 0 bytes | `archive/fp-rnd/race-quali-20261003` | 正常 remove 成功；local branch deleted -D |
| `codex/fp-longrun-relative-fuel-rnd` | `18829d39445447d07d74a01d22290d8ee4acb64b` | registration 的 .git 缺失；relative-fuel/geometry 残留空目录，bias/mass 残留87 files / 8900964 bytes | `archive/fp-rnd/transfer-20261003` | KEEP：normal remove 拒绝；worktree registration 与 branch 保留，未 force |
| `codex/fp-longrun-same-session-repeatability-rnd` | `c23392ac809495c7c9e49e5dade046e0da883605` | 清理前实际目录不存在；无当前文件可盘点；陈旧 registration 已正常移除 | `archive/fp-rnd/transfer-20261003` | 正常 remove 成功；local branch deleted -D |
| `codex/fp-longrun-selector-exploration` | `0901138d67f56c0db564c2cf7488f3cbcea6c0ed` | 清理前实际目录不存在；无当前文件可盘点；陈旧 registration 已正常移除 | `archive/fp-rnd/transfer-20261003` | 正常 remove 成功；local branch deleted -D |
| `codex/fp-programme-bias-mass-rnd` | `dd74e459bdb8242185921e84fbedd2662aba44bb` | registration 的 .git 缺失；relative-fuel/geometry 残留空目录，bias/mass 残留87 files / 8900964 bytes | `archive/fp-rnd/transfer-20261003` | KEEP：normal remove 拒绝；worktree registration 与 branch 保留，未 force |
| `codex/fp-race-field-scale-rnd` | `12a4f47f14193cd50577e3dc097be14fe0ec221e` | 清理前实际目录不存在；无当前文件可盘点；陈旧 registration 已正常移除 | `archive/fp-rnd/transfer-20261003` | 正常 remove 成功；local branch deleted -D |
| `codex/fp-race-relative-geometry-rnd` | `afd4213e464eb3ddcb6aefe9a026720694adb486` | registration 的 .git 缺失；relative-fuel/geometry 残留空目录，bias/mass 残留87 files / 8900964 bytes | `archive/fp-rnd/transfer-20261003` | KEEP：normal remove 拒绝；worktree registration 与 branch 保留，未 force |

## Durable evidence and old-path mappings

`A` = `/Users/zhxutong/dr-wo/woData/research_archive/fp_performance_rnd_20261003/`；`W` = `/Users/zhxutong/dr-wo/woStrategy/research/.worktrees/`。未找到已有适用 research_archive convention，本次建立 A。

- `A/frozen_git/research/`：从 transfer tag 提取的完整 tracked research 快照（REPORT/protocol/scripts/tests/committed banks/results）。
- `A/race_quali_git/doc/fp_race_quali_evidence_rnd/`：独立 FP/Quali tag 的 tracked 证据。
- Material ignored 文件：`W/<candidate>/<relative-file>` → `A/non_git/<candidate>/<relative-file>`。逐文件复制与 SHA-256/size 相等后才允许正常 worktree remove；未使用逐文件批量删除脚本。
- REPORT/committed research 路径：`W/<candidate>/research/<line>/<file>` → `A/frozen_git/research/<line>/<file>`；精确旧代码仍以表中 checkpoint + preserving tag 恢复。
- Common-load `bank_registry.json` 的15条依赖：原 `W/fp-performance-expanded-exploration/research/.../bank.npz` → Git-tracked banks 的 `A/frozen_git/research/...` 或 ignored banks 的 `A/non_git/fp-performance-expanded-exploration/research/...`。每条映射及 SHA-256 保存在 `A/archive_inventory.json` 的 `dependency_mappings`。
- 可执行的迁移副本：`A/relocated_bank_registry.json`；其15个新目标均实际存在并与旧 bank 字节相同。原 registry/冻结 commit 不改，原路径仅作为来源身份，不再作运行入口。
- 所有 non-Git 文件、文件数、总字节、逐文件 hash、原始 inventory 及主 repo 基线保存于 `A/archive_inventory.json`、`A/main_before.json`。它们是技术 evidence，不是科学报告。

保全 non-Git：1681 files / 3180075663 bytes；tracked frozen copies：1903 files / 421993945 bytes。

## Cache classification / external state

- A / must preserve：common-load inputs/outputs/draws、expanded joint/traffic evidence、environment prediction tables、Sprint 与 transfer banks/独立 HTTP/parser caches。没有足够证据证明历史 HTTP/parser 状态可 bit-for-bit 重建，因此未将 GB 级缓存当作可删下载缓存。
- B / rebuildable：仅 `__pycache__`、`.pytest_cache`、`.DS_Store`；不含 inputs、banks、draws、network sources。精确清单在 inventory，正常 remove 时才删。
- C / durably elsewhere：原位保留 `/Users/zhxutong/dr-wo/woData/wodata/research_session_inputs/schema_v1/fp_race_2025_transfer_20261003/` 与 `fp_sprint_race_20261002/`；共享 `/Users/zhxutong/dr-wo/woData/fastf1/raw_cache/schema_v1/` 未修改或删除。
- 环境研究18个 telemetry parser 文件约446 MB仍在共享 raw_cache；其历史 HTTP before-image 缺口保持记录。Transfer 的独立 HTTP cache 清空/事后 snapshot/reseed 历史不补修，精确保留 archive 中副本。

## Intentionally retained / final verification

主 checkout、traffic-position-part1、r15-historical-interruptions、integration checkout、fp3-q-historical detached checkout、r15-fp-temperature-context，以及 qualifying-track-evolution、rd3-continuation-shadow 等非候选 branch 不动；未从本任务证据推断它们全部 active。剩余 worktree 不保证全部 active，具体未确认 scope 者保守保留。

初始批量“copy then unlink originals”脚本被 automatic approval review 拒绝且未执行；实际采用 copy-only + 全量 hash 验证，再逐个正常 git worktree remove。若 normal remove 拒绝，记录拒绝并保留，不 force，不绕过。

Final verification passed：8个存在的 worktrees 正常移除、3个已无目录的陈旧 registrations 正常移除；11个 local branches 在 -d 因未 merge 拒绝后，按已授权的保全例外 -D 删除。另3个 invalid registrations/branches 因正常 remove 拒绝而保留。只提交本 manifest 与 canonical index。三个旧 chats 的归档不依赖保留它们的旧 worktrees；必须长期保留两个 archive tags、A 和原位 woData evidence。

已分类 rebuildable 总量：2890605 bytes；实际移除/释放量以最终 verification 更新为准。

## Final cleanup outcome

- Removed real worktrees：common-load、expanded、correction-first、transfer、discovery、Sprint、environment、early race/quali（8）。
- Removed stale registrations：same-session-repeatability、selector、field-scale（3，清理前目录已不存在）。
- Retained invalid registrations + local branches：`codex/fp-longrun-relative-fuel-rnd`、`codex/fp-programme-bias-mass-rnd`、`codex/fp-race-relative-geometry-rnd`；正常 remove 报 `.git does not exist`。不 prune/force，也不删除相应 branch。它们不是 active work，故“剩余全 active”目标尚未完全达到。
- Bias/mass 87个残留文件约8.49 MiB额外复制并 hash 校验到 `A/residual_worktrees/fp-programme-bias-mass-rnd/`；原件未动，详细记录 `A/residual_inventory.json`。另外两个异常目录为空。
- 删除的 rebuildable 文件仅为 inventory 所列 bytecode/pytest/系统元数据，共2890605 bytes（约2.76 MiB），随正常 worktree remove 移除。GB级 network caches 未归为 rebuildable。
- Archive 实际占用约3.38 GiB；净释放约2.1 GiB。8个 worktrees 原占用约5.47 GiB，其中首个 correction-first 只有327 MiB的取整 before 记录；其余7个的精确 before 合计5399820 KiB，记录于 `A/pre_remove_disk.json`。空间数字包含冻结副本及保全副本，未将“迁移”误算为全部净释放。
- 三个旧 chats 均 SAFE_TO_ARCHIVE；无新 evidence 损失。必须保留 A、两个 annotated tags、原位 woData research inputs/shared source state；无 frozen history rewrite、production changes、remote-ref changes、push/merge/gc。
