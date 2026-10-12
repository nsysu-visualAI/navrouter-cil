# PREREG-23 — EXT-3 登記（論文 ladder 的兩格：整片平均的兩階段讀出、FINAL-B 換成文字 cosine 讀出）

登記時間：2026-10-12 23:59。本檔 commit 之後不得修改。
**本檔寫於 run 開始之後。** 兩格的 run 在 2026-10-12 23:51 執行完畢（程式 commit 864f72b、結果 commit 36f3224），
其數字在本檔 commit 之前已在聊天回報過一次（當時配對用 paired t-test）。第一部分的判讀規則由 PI 在 run 開始前寫定，
於 run 之後才送達並登記；時序照實記在 `outputs/navcil/mac/ext3/DECISIONS.md` D1。
機器：Mac（MacBookPro18,3，Apple M1 Pro，16 GB），`--device cpu`、`torch.set_num_threads(8)`、torch 2.11.0、closed-form float64；
四列（兩格與兩個對照）來自同一台、同一次執行。
分支：`ext-3`（自 main @ 264e038）。不合併；不修改既有程式、既有報告與既有 PREREG；不動 `paper/`。
報告：`outputs/navcil/mac/REPORT_ext3.md`；判斷紀錄 `outputs/navcil/mac/ext3/DECISIONS.md`。

---

## 第一部分：指令原文

### 目的

補論文 ladder 的兩格。只跑這兩格，其他設定一律不動。B（FINAL-B）與 direct 8-class（LIN8）在同一台、同一批重算當對照；
不同機器的數字不能相減。

- **格 1**：subtype 用 whole-slide mean（草稿裡的 0.903，來源 MOE-3）。先確認設定是：analytic router（whole-slide mean）＋
  subtype ridge readout 也吃 whole-slide mean；沒有 text-guided evidence，沒有 learned head。設定不符就停下來回報。十折 × 兩序。
- **格 2**：no-head ＋ cosine readout。跟 B 完全相同（analytic router、text-guided 64 patches），只把 subtype readout 從 ridge 換成：
  evidence 平均向量對 τ̂ 兩個 class text 的 cosine，取較大者。十折 × 兩序。

### 判讀規則（PI 於 run 開始前寫定）

- **顯著的定義**：十折逐折配對，exact binomial 雙尾符號檢定 p < 0.05（即贏 9 或 10 折）。
- **格 1 預期**：約 0.903，對 LIN8 顯著。若對 LIN8 不顯著：論文把第一步寫成「分解本身增益有限」，方法不變。
- **格 2 預期**：比 FINAL-B 低約 3 pp 且顯著。若與 FINAL-B 差不到 1 pp 且不顯著：方法不變，論文加一句「讀出換成 cosine 差異不大」。

### 整理規則

1. 分支 `ext-3`（從 main 開），完成後不自己 merge。
2. 格 1 先對到既有的列：查 `ABLATION_full.csv` 與 `REPORT_moe3.md`，0.903 是哪一列、是否就是 M2。若是 M2，補算 CIL 就改了
   PREREG-22 第 1c 點，寫一份 AMENDMENT（下一個空號），理由：論文的消融階梯需要這一格。
3. 統計照既有規範：逐折配對報 mean diff、贏的折數／10、exact binomial 雙尾 p、slide 層級 bootstrap 95% CI（1000 次，seed 0）。
   不用 paired t-test。配對三組：格 1 vs LIN8、格 1 vs FINAL-B、格 2 vs FINAL-B。
4. 指標定義照 FINAL_RESULTS：Masked ACC 用 Table 1 定義，Forgetting 用既有定義；兩序叫 reverse 與 paper。
5. 輸出：`REPORT_ext3.md`、`ext3/DECISIONS.md`、逐折 CSV（欄位同 `ABLATION_full.csv`）進版控；pytest 通過。
6. 已經跑完的不重跑。每階段完成在聊天回報一次，全部完成回報總摘要；不問問題；失敗記錄後繼續其他階段。

---

## 第二部分：操作化細則（不改變第一部分）

未列於此、執行中才遇到的判斷，寫入 `outputs/navcil/mac/ext3/DECISIONS.md`，並印在報告內。

1. **資料、類別順序、序、指標**：同 PREREG-21 細則 1、3 與 PREREG-22 細則 1。ACC_t、Forgetting、BWT 由 `nc5_report.cil_full`；
   Masked ACC（Table 1 定義）＝向量取自 τ̂ 的版本、在真實任務兩類內 argmax；WP ＝ Masked ACC（oracle 定義）。
2. **列**（`row` 欄的鍵；沒有 seed）：
   - `MV_RDG`（格 1）：TP ＝ AR（[mean_vec; 1]，γ = 1e-3）；讀出 ＝ [mean_vec; 1] 累加式 8 類 ridge（γ = 1e-3，沿用 MOE-3 F5 的值、不重選），τ̂ 兩類內 argmax。
   - `B_TXT`（格 2）：TP 與向量同 FINAL-B（s = s0、四輪各 16、λ = 1.5 的 v0）；讀出 ＝ v0 對 τ̂ 兩類文字的 cosine，第一類 ≥ 第二類判第一類。
   - `FINALB`：PREREG-22 的 FINAL-B（γ_B = 1e-3）。`LIN8`：mean_vec／8 類 ridge γ = 0.01、無 TP（`nc8_report.stage_row`）。
3. **來源**：四列的逐折值取自 `outputs/navcil/mac/lad1/cells/<row>/<order>_fold<f>.json`（`scripts/lad1_cells.py` @ 864f72b 的輸出），不重跑。
   該次執行內的查核：FINALB、LIN8 對 `ext2/a/` 逐折差 ≤ 1e-9；MV_RDG 的 t = 4 CIL、WP、Ā 對 `moe3/f5.json` 的 S-R0 × mv 差 ≤ 1e-9。
4. **逐折配對**（PREREG-21 細則 4；函式 `ext1_d.sign_pair`）：每序、每個指標，D_f = 前者_f − 後者_f；報兩者的十折 mean ± sd、D 的平均、
   贏（D > 1e-12）／輸（D < −1e-12）／平手折數、`scipy.stats.binomtest(贏, 贏 + 輸, 0.5, alternative="two-sided").pvalue`（平手不計）。
   指標：t = 4 的 ACC、WP、Masked ACC（Table 1 定義）、Forgetting、BWT、Ā。「贏」一律指數值較大（Forgetting 的贏＝遺忘較多），與既有報告相同。
5. **bootstrap**（PREREG-21 細則 23；函式 `ext1_d.bootstrap`）：只做 t = 4 的 CIL ACC；單位 = test slide；(折, 任務) 分層、配對重抽；
   統計量 = 十折平均的四任務等權 CIL ACC 差；1000 次；`torch.Generator().manual_seed(0)`；95% CI = 2.5 與 97.5 百分位。
6. **判讀**：「顯著」只以 t = 4 CIL ACC 的符號檢定判定（p < 0.05），兩序各判一次；bootstrap CI 並列、不參與判定。
   格 1 的「約 0.903」以十折平均四捨五入到三位小數是否為 0.903 記錄。格 2 的「差不到 1 pp」＝ |平均差| < 0.01。
   格 1 對 FINAL-B 沒有預期，只報數字。
7. **逐折 CSV**：`outputs/navcil/mac/EXT3_perfold.csv`，欄位同 `ABLATION_full.csv`（row, order, fold, t, CIL_ACC, WP, MaskedACC_table1,
   MaskedACC_oracle, Forgetting, BWT, storage_bytes）。
8. commit 前跑 pytest；不 merge、不 push。
