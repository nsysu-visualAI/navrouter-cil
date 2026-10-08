# docs/team-handoff：給組員（與組員的 AI）的接手包

目的：不用讀整個 codebase，十分鐘內知道這條線做了什麼設計、跑了什麼實驗、數字是多少、現在卡在哪。所有數字出自 `main` 的 tag `ext2-20261003`（commit 7ad6cd1）。

## 一句話

不保存任何舊切片、推論時不知道任務，用凍結的 CONCH 做四個器官（食道、腎、乳房、肺）的亞型持續學習。判器官與判亞型都是閉式 ridge（累加統計量），中間用類別文字相似度挑 64 個 patch 當證據，每任務可選一個 1,033 參數的修正頭。十折 CIL 正確率 0.925；同一組切分重跑的 QPMIL-VL 為 0.868／0.884，十折贏 9 折。

## 建議閱讀順序

| 順序 | 檔案 | 讀完知道什麼 | 時間 |
|---|---|---|---|
| 1 | `01_工作匯報_HackMD_2026-10-08_含上週0930報告.md` | 現況總覽：設定、系統三步、依老師路線做了什麼、主表、逐折配對、分數階梯、逐類別、負結果、要討論的事。後半是 9/30 的上一版報告，含 TP 的手算例子與 Table 1 | 15 分鐘 |
| 2 | `02_架構圖_v3_2026-10-08.png` | 系統圖（待定稿，已知三處小筆誤見 `05` 第 2 節） | 2 分鐘 |
| 3 | `05_給老師的報告_兩套並排_2026-10-03.md` | 兩套（有修正頭／不訓練）並排、同折對照、MoE 負結果、MergeSlide 為何不比、請老師定的四件事、老師可能的追問 | 15 分鐘 |
| 4 | `03_論文草稿模擬_有修正頭版_2026-10-03.md`、`04_論文草稿模擬_不訓練版_2026-10-03.md` | 兩個版本的論文骨架：貢獻三條、方法、實驗、claim ↔ 表、自評。用來討論論文方向 | 各 10 分鐘 |
| 5 | `06_代號對照與白話整理_讀舊報告必備_2026-09-25.md` | 讀 `outputs/navcil/mac/REPORT_*.md` 與 `history/` 之前先看：R0–R8、L0–L2、I6、AR、LIN8、NC-x、MOE-x、PREREG 都在這裡 | 查閱用 |
| 6 | `history/`（依日期排序） | 每一輪的設計、預註冊判準、結果、判讀。想知道「為什麼不做 X」就翻這裡 | 選讀 |

## 現在的系統（對照架構圖）

| 步 | 做什麼 | 要存什麼 | 程式代號 |
|---|---|---|---|
| 1 判器官 | 整片 patch 平均 → L2 → 補 1 → 閉式 ridge → 器官 | G（513×513 共用）、C（每任務一欄） | AR（TP） |
| 2 挑證據 | patch 對該器官兩類文字的最大 cosine（切片內 z-score）＋修正頭 g → 四輪各挑 16 個、避開相似 → 64 個平均 → L2 → 補 1 | 每任務兩個文字向量、一個修正頭（1,033 參數） | I6（head）；λ = 1.5 |
| 3 判亞型 | 證據向量 → 閉式 ridge → 在該器官兩欄內取大 | A（513×513 共用）、B（每任務兩欄） | LR／RDG（閉式讀出） |

- 主系統（有修正頭）：CIL 0.925 ± 0.019，五 seed。程式代號 FINAL-A。
- 拿掉修正頭（不訓練）：0.921 ± 0.022，無 seed。程式代號 FINAL-B。差 0.4 pp，十折 7 比 3，不顯著。
- 10/1 報的舊版（判亞型改和文字比 cosine）：0.913。

## 時間線（詳細在 history/）

| 日期 | 批次 | 做了什麼 | 結果 |
|---|---|---|---|
| 9/24–9/25 | NC-0～NC-1 | 建乾淨 repo（不含 QPMIL）；每任務一個 expert（TIL）；七種判任務方法；組成 CIL | TIL 0.940；CIL 最好 0.890（prototype router） |
| 9/25 | NC-2～NC-4 | 六種 router；expert 換 LoRA（rank 2 夠）；三種新 router；容量依需求 | router 改法全沒贏；LoRA 省參數不掉分 |
| 9/25–9/26 | NC-5～NC-6 | 背景分派（失敗）；router 改閉式 ridge（AR）；TSP 文字投影（失敗） | 任務辨識 0.956 → 0.986；CIL 0.918 |
| 9/26 | NC-7～NC-8 | expert 改成 zero-shot 分數＋1,033 參數修正頭（I6）；主表同批重算 | CIL 0.913（兩序相同），expert 儲存 3% |
| 10/1–10/3 | MOE-0～MOE-4 | 八種 MoE 設計；發現把判亞型改成閉式 ridge 讀出才是有效的 | 0.913 → 0.925，贏 8–9 折；MoE 全沒過判準 |
| 10/3 | MOE-5、EXT-1、EXT-2 | 定案；QPMIL-VL 同折逐折匯入；全量重算零差；不訓練版全套 | 主表定稿 |

## 程式與輸出在哪

| 要找 | 位置 |
|---|---|
| 定案數字與消融 | `outputs/navcil/mac/FINAL_RESULTS.md`、`ABLATION_full.csv`、`FINALB_RESULTS.md` |
| 一張 slide 的逐步數值 | `outputs/navcil/mac/FINAL_EXAMPLE.md`、`FINALB_EXAMPLE.md` |
| 同折對照與統計 | `outputs/navcil/mac/REPORT_ext1.md`、`REPORT_ext2.md`、`outputs/external/qpmil_vl/` |
| 每一批的報告與判準 | `outputs/navcil/mac/REPORT_moe0.md`…`REPORT_moe4.md`、`REPORT_stage*.md`；`PREREG-*.md`、`AMENDMENT-*.md`、各批 `DECISIONS.md` |
| 讀出 ridge 的累加程式 | `scripts/moe3_common.py`（第 242–263 行） |
| MergeSlide 為何不比 | `outputs/external/mergeslide/DECISION.md`、`PORT.md` |
| 執行規範 | `AGENTS.md`、`RUNBOOK_navcil.md` |

## 現在要討論的事

1. 修正頭的定位：留在方法裡、消融寫明不顯著（建議）。
2. 貢獻定位能不能投 ISBI 2027（10/26）：「分析 + 一個簡單但強的系統」。
3. 論文名稱：沿用 PathSelect 或改。
4. MergeSlide 同折對照：ISBI 之後在公司 DGX Spark 做（要 2.6 TB 原始切片）。

## 給 AI 助手的提示

讀這個資料夾時，`06_代號對照` 是解碼表；`01` 與 `05` 是最新事實；`history/` 裡早期文件的數字（0.899、0.913）是當時的主系統，已被後來的 0.925 取代，不要混用。文件裡的「PI」指 Aaron，不是指導老師。
