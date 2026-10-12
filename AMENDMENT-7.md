# AMENDMENT-7 — PREREG-22 第 1c 點：`M2_g0` 的設定補算 CIL 與 t < 4（EXT-3 的格 2）

日期：2026-10-12。分支 `ext-3`（自 main @ 264e038）。**PREREG-22.md、`ABLATION_full.csv` 不修改。**
登記時間：EXT-3 的 run（864f72b，2026-10-12 23:51）之後、REPORT_ext3 產出之前 commit。

## 問題

PREREG-22 第 1c 點（承 EXT-1 的 PI 裁決；`ext2/DECISIONS.md` D3）：M2 只報 WP，CIL 與 t < 4 的欄填「—」，不補算。

EXT-3 的格 2（`B_TXT`：AR ＋ s = s0 四輪各 16 的 v0 ＋ 對 τ̂ 兩類文字的 cosine）與 `ABLATION_full.csv` 的 `M2_g0` 列是同一個設定：
`M2_g0` 是 M2 表裡「不用 head（g = 0）」的那一列，只有告訴任務的 t = 4 WP。兩者 t = 4 WP 的逐折值最大絕對差 0（兩序、十折；
`ext1/c/M2_g0/` 對 `lad1/cells/B_TXT/`），十折平均 0.8966。格 2 另外算了 CIL ACC、Masked ACC（Table 1 定義）、Forgetting、BWT 與 t = 1…3，
等於替 `M2_g0` 這個設定補算了第 1c 點說不補算的欄位。

補充指令寫的條件是「格 1 若是 M2」。格 1（0.903）不是 M2：它不在 `ABLATION_full.csv`，只在 REPORT_moe3 F5 的「全部 patch × S-R0」。
對到 M2 的是格 2；依同一個理由登記本檔。若 PI 認為格 2 不需要 AMENDMENT，本檔作廢，數字不受影響。

## 修正

- 範圍只限 `M2_g0` 的設定，以新列名 `B_TXT` 寫在 `EXT3_perfold.csv` 與 REPORT_ext3；`ABLATION_full.csv` 的 M2 各列維持原狀（CIL 與 t < 4 仍為空）。
- `M2_head_*` 四列（跨器官共用 head）不補算，第 1c 點對它們仍然有效。
- 補算用的向量是 `moe2` 向量快取的 g0（test 每張有四個任務文字的版本，可取 τ̂ 的版本）；EXT-1 當時 M2 用的 NC-8 快取只有真實任務文字的版本，
  這是當時填「—」的原因。

## 理由

論文的消融階梯需要這一格。
