EXT-3 的判斷紀錄（2026-10-12）。每條：做了什麼判斷、為什麼。D1–D3 在 PREREG-23 commit 時寫入；之後為整理中新增。

| # | 判斷 | 為什麼 |
|---|---|---|
| D1 | PREREG-23 寫於 run 開始之後；判讀規則在 run 開始前已由 PI 寫定。時序：兩格的 run 在 2026-10-12 23:51 執行（程式 commit 864f72b、結果 commit 36f3224）；逐折值、十折平均與 paired t-test 的配對結果在 PREREG-23 commit 之前已在聊天回報過一次；補充指令（含判讀規則、改用符號檢定與 bootstrap）在那之後才送達。因此「回報任何數字之前先 commit PREREG-23」這一條沒有做到，照實記錄；符號檢定與 bootstrap 的數字則是在 PREREG-23 commit 之後才計算與回報。 | 指令 2。先前聊天裡的 paired t-test 數字不屬於既有規範，由 REPORT_ext3 的符號檢定與 bootstrap 取代，不引用。 |
| D2 | 先前的兩個 commit 在分支 `lad-1`（自 main @ 264e038），不在 main；把 `lad-1` 改名為 `ext-3` 繼續工作。沒有改寫歷史，main 沒有變動。 | 指令 1 的「停下來」條件是「已經 commit 在 main」，不成立。`lad-1` 本來就是從 main 開的，改名後即為「從 main 開的 ext-3」。 |
| D3 | 不重跑。四列的逐折 JSON 留在原來的 `outputs/navcil/mac/lad1/cells/`（連同 `lad1/LADDER_perfold.csv`、`lad1/paired.json`、`lad1/REPORT_lad1.md` 與 log）；EXT-3 只讀這些 JSON 做配對、bootstrap 與報告。`lad1/REPORT_lad1.md`、`lad1/paired.json` 裡的 paired t-test 與 t 分布 CI 不引用。 | 指令「已經跑完的不重跑，只照下面整理」。搬動目錄會讓 864f72b 的程式與其輸出路徑對不上。 |
