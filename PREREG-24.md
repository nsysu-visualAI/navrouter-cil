# PREREG-24 — EXT-3 第二段登記（消融表的 γ 統一用同一條驗證規則選：LIN8 與格 1）

登記時間：2026-10-13，本段任何程式執行前、任何數字計算前 commit。本檔 commit 之後不得修改。
機器：Mac（MacBookPro18,3，Apple M1 Pro，16 GB），`--device cpu`、`torch.set_num_threads(8)`、torch 2.11.0、closed-form float64；
本段的列（LIN8、格 1、FINAL-B）在同一台、同一次執行重算。
分支：`ext-3`。不合併；不修改既有批次的程式、報告與 PREREG；不動 `paper/`。
報告：`outputs/navcil/mac/REPORT_ext3.md` 新的一節；判斷紀錄 `outputs/navcil/mac/ext3/DECISIONS.md`。

登記時已知的數字（照實列出）：LIN8（γ = 0.01）、格 1（γ = 1e-3）、FINAL-B（γ = 1e-3）的 test 結果與三組配對（REPORT_ext3 第 2–5 節）；
FINAL-B 的 validation 目標值與 test 的 γ 敏感度（REPORT_ext2 0-4、A-9）。LIN8 與格 1 在本規則下的 validation 目標值尚未計算過。

---

## 第一部分：指令原文

0. **原則**：消融表每一列的 γ 都用同一條驗證規則（PREREG-20 的 trajectory 驗證）在同一個 grid 上選，和 EXT-2 對 FINAL-B 的做法相同；不看 test。
   **判讀**：
   a. 選到的 γ 與現用值相同：消融表照現有數字。
   b. 有列的 γ 改變：主表用驗證選到的數字，現用值的數字放補充；兩者都寫進 REPORT。
   c. 第二階（格 1 → FINAL-B）在新 γ 下小於 +1 pp，或 slide 層級 bootstrap 95% CI 含 0：在 REPORT 標明，不加實驗。
1. LIN8 與格 1 各選一次 γ；先查 LIN8 現用的 0.01 是不是同一條規則選的，寫明出處。FINAL-B 直接引用 EXT-2 的選擇結果。
2. 用選到的 γ 重算 LIN8 與格 1：十折 t = 4 的 CIL ACC、WP、Masked ACC（Table 1 定義）、兩序 Forgetting。
3. 配對：格 1 − LIN8、格 1 − FINAL-B；指標 CIL ACC 與 WP；報平均差、贏／輸／平手、符號檢定 p、slide 層級 bootstrap 95% CI（1000 次，seed 0）。
4. PREREG-24 commit 之前，不在聊天裡回報任何數字。
5. 結果寫進 REPORT_ext3.md 新的一節與 ext3/DECISIONS.md；逐折 CSV 進版控；pytest 通過；完成回報一次。

---

## 第二部分：操作化細則（不改變第一部分）

未列於此、執行中才遇到的判斷，寫入 `ext3/DECISIONS.md`，並印在報告內。

1. **規則**（同 EXT-2 D4、PREREG-22 細則 5）：PREREG-19 細則 5 的目標——每個候選、每折，對兩序 × t = 1…4 共 8 個
   「validation 告訴任務 WP_t（已學任務等權）」取平均，再取十折平均；取最大者；完全相等才算同分，同分取較大的 γ。
   grid ＝ {1e-5, 1e-4, 1e-3, 1e-2, 1e-1}（`moe3_common.RDG_G`）。選到 grid 邊界就標示，不延伸。只用 validation。
2. **格 1**：`moe3_common.val_objective(run, f, "mv", "RDG")` 與 `moe3_common.select`（與 `ext2_a.reselect` 對 FINAL-B 用的是同兩個函式）。
3. **LIN8**：同一個目標、同一個 grid，權重用 LIN8 自己的程式（`nc8_report.B8.W(..., lin8=True)`，經 `moe1_common.lin8_stage`）；
   「告訴任務」＝在真實任務兩欄內判（第一類 − 第二類 ≥ 0 判第一類），即 LIN8 的 Masked ACC 定義。
4. **登記時由程式已知的結構**：LIN8 與格 1 的讀出是同一個累加式 ridge（同樣的 [mean_vec; 1]、同樣每類一欄的 B），兩者只差判定方式
   （LIN8：已學類別內 8 類 argmax、無 TP；格 1：AR 判 τ̂、τ̂ 兩類內 argmax）。告訴任務的目標看不到判定方式的差別，
   所以兩次選擇預期得到同一組目標值與同一個 γ。仍各算一次，報兩組目標值的最大絕對差；若不同，各用各的。
5. **限制與描述列**：本規則的目標不含跨任務的錯誤，而那正是 LIN8 的 8 類 argmax 會犯的錯。另報 LIN8 在同一個 grid 上的
   「validation 8 類 CIL ACC（兩序 × t = 1…4 平均、已學任務等權、十折平均）」，**只作描述**：不用於選擇、不進主表、不據以加算任何 test 數字。
6. **LIN8 現用值的出處**：寫明 0.01 當初的 grid、目標、同分規則與是否在邊界（出處：程式行號與報告行號），並與本規則逐項對照。
7. **FINAL-B**：γ_B = 1e-3，引用 `ext2/a.json` 的 `gamma`（EXT-2 0-4），不重選。
8. **重算**（一次執行內）：LIN8(γ)、格 1(γ) 各算「現用值」與「選到的值」（相同就只有一組）；FINAL-B(1e-3) 同批重算。t = 1…4、兩序、十折。
   - 格 1、FINAL-B：`ext2_a.first_of` ＋ `ext1_c.cl_eval`（與 run 相同）。
   - LIN8(γ)：`moe1_common.lin8_stage` 的 logits，已學類別內 argmax（CIL）、真實任務兩欄內判（Masked）；指標由 `nc5_report.cil_full`。
   - 查核（不過即停，不組表）：**KD** LIN8(0.01)、格 1(1e-3)、FINAL-B 的逐折值（acc_t、wp_t、mk1_t、Forgetting、BWT）對 `lad1/cells/` 最大絕對差 ≤ 1e-9，
     且 t = 4 逐張 CIL 判定對錯相同。**KE** 每列 t = 4 的逐張告訴任務對錯，依任務等權平均後與 wp_t[3] 差 ≤ 1e-9。
9. **報的指標**：t = 4 的 CIL ACC、WP、Masked ACC（Table 1 定義）、Forgetting（兩序）、BWT、Ā；十折 mean ± sd。
10. **配對**：格 1 − LIN8、格 1 − FINAL-B；各列用選到的 γ（主），γ 有改變時現用值的同兩組配對放補充。指標 CIL ACC 與 WP（t = 4）。
    `ext1_d.sign_pair`（贏 = 前者較大；exact binomial 雙尾，平手不計）；`ext1_d.bootstrap`（slide 層級、(折, 任務) 分層、配對、1000 次、seed 0）。
    WP 的 bootstrap 用同一個函式，逐張值取「告訴任務時是否判對」（細則 8 的 KE 那一組）；統計量 = 十折平均的四任務等權 WP 差。
11. **判讀的操作化**：
    - a／b：逐列比較選到的 γ 與現用值（LIN8 0.01、格 1 1e-3、FINAL-B 1e-3）。
    - c：「第二階」＝ FINAL-B − 格 1 的 t = 4 CIL ACC（各用選到的 γ）。「小於 +1 pp」＝ 十折平均差 < 0.01；
      「CI 含 0」＝ CIL ACC 的 bootstrap 95% CI 下界 ≤ 0 ≤ 上界。任一成立就在 REPORT 標明；不加實驗。
      符號檢定是否顯著（PREREG-23 的定義）並列，不屬於 c 的條件。
12. **逐折 CSV**：`outputs/navcil/mac/EXT3_gamma_perfold.csv`，欄位同 `ABLATION_full.csv`；`row` 帶 γ（例：`LIN8_g0.01`）。
13. 短時間指令以 `caffeinate -ims` 在前景執行（同 EXT-2 D9）；commit 前跑 pytest；不 merge、不 push。
