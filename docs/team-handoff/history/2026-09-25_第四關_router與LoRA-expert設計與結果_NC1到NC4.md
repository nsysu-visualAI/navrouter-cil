# NavRouter-CIL：第一到三關判讀與第四關計畫（2026-09-25）

與 `2026-09-24_全局計畫` 的關係：那份是整體計畫；本頁接在 NC-1（navrouter-cil 4fe7ac3）之後，記錄結果判讀、T-Hard 原始程式查核、9/25 的六項決定、第四關（router 線與 LoRA 線）的預先註冊、第五關 router 機制、NC-4 容量依需求與兩種 router。文件裡的「PI」指 Aaron。

---

## 結論

| # | 問題 | 結論 |
|---|---|---|
| 1 | 乾淨頭做得起來嗎 | 做得起來。每任務一個 expert 的 TIL（2 類）十折 0.9399 ± 0.0163，十折都贏 zero-shot top-64（0.9005），不需要 QPMIL |
| 2 | CIL 卡在哪 | 全部卡在任務辨識（TP）。實測 CIL 與每折 TP × WP 平均差不到 0.003，老師的乘法在我們的設定裡成立；oracle 分派的 CIL 就是 0.9399 |
| 3 | 現在最好的 CIL | proto 分派＋Hard：0.8900（兩序相同）。每任務只存一個 512 維平均向量（2,048 bytes），不存任何 slide 或特徵，零 replay |
| 4 | 只用文字判任務（T-Hard 類）夠嗎 | 不夠。text-class＋Hard 0.8142，比完全不訓練的 zero-shot 8 類 top-64（0.8270）還低。主要錯在 ESCA 被判成 Lung：食道鱗狀細胞癌與肺鱗狀細胞癌在文字與形態上都相近 |
| 5 | N1（expert 的證據判任務） | 照原定義不成立：nav 的 TP 0.8523，低於 text-class。依 PREREG 落在「其他」：N1 列為消融，第四關主做 N2 |
| 6 | 同學的 T-Hard 是什麼 | 是 PathSelect（含 replay，M128／M512）裡的查詢向量條件化，不是 expert 的分派器。十折結果對 baseline 是 +0.10 pp（M128）與 −0.85 pp（M512），信賴區間都跨 0 |
| 7 | 老師問的「T-Hard 好的話是不是就不用 replay」 | 在我們的架構裡答案是對的：expert 各自獨立、不會忘，所以不需要 replay；剩下的問題全在分派準不準。文字分派 0.814，prototype 分派 0.890 |
| 8 | 第四關做什麼 | 兩條線各只動一個變數：R 線換 router（只推論）；L 線把 expert 換成共享底座＋LoRA（N2，省參數） |

---

## 1. 第一到三關結果（NC-1，navrouter-cil 4fe7ac3，Mac CPU，同一批；PREREG e06a8e5 開跑前 commit）

### 1.1 第一關：乾淨 WP（test Masked ACC，四任務平均，十折 mean ± sd，λ* = 1.5）

| 方法 | 結果 |
|---|---|
| (a) zero-shot 全部 patch 平均 | 0.7763 ± 0.0288 |
| (b) zero-shot top-64 | 0.9005 ± 0.0162 |
| (c) random-64 | 0.7760 ± 0.0195 |
| (d) expert 一次 top-64 | 0.9370 ± 0.0183（softmax 聚合 0.9397） |
| (e) expert 四輪 | **0.9399 ± 0.0163**（softmax 聚合 0.9392），贏 (b) 10/10 折 |
| (d)／(e) 在 oracle 分派下改用 8 類判 | 0.8280／0.8264 |

### 1.2 第二關：TP（t = 4，兩序十折平均）

| 變體 | 要存什麼 | TP |
|---|---|---|
| proto | 每任務 2,048 bytes | 0.9415 |
| patch-vote | 不存 | 0.9379 |
| text-class | 不存 | 0.9224 |
| fuse（text-class＋nav） | 不存 | 0.9008 |
| nav-cal | 每任務 2 個數 | 0.8613 |
| nav | 不存 | 0.8523 |
| text-organ | 不存（文字 key） | 0.7116 |

| 主要錯誤 | 張數（十折合計） |
|---|---|
| text-class：ESCA → Lung | 40 |
| nav：Lung → 其他任務 | 294 |
| text-organ：Lung → BRCA | 528 |

### 1.3 第三關：CIL（t = 4；兩序的 ACC 相同，Forgetting 不同）

| 分派 | ACC | Forgetting（reverse／paper） |
|---|---|---|
| oracle＋Hard | 0.9399 | 0／0 |
| **proto＋Hard** | **0.8900** | 0.018／0.032 |
| text-class＋Hard | 0.8142 | 0.120／0.037 |
| nav＋Hard | 0.7668 | 0.113／0.097 |
| 無分派：zero-shot 8 類 top-64 | 0.8270 | 0.070／0.025 |
| 無分派：E-max | 0.6918 | 0.190／0.113 |
| 外部參考：QPMIL-VL 已發表值（非 paired） | 0.859（reverse）／0.890（paper） | — |

| 判讀 | 內容 |
|---|---|
| 分解成立 | 每折 TP × WP 與實測 CIL 平均差不到 0.003 |
| 分派本身也消掉分類頭的跨任務混淆 | oracle 分派＋8 類只有 0.826，分派後在 2 類裡判是 0.940 |
| 順序不變 | expert 與 proto key 都只用各自任務的資料，t = 4 時兩序 ACC 相同；只有中途的 Forgetting 不同 |
| Soft ≈ Hard | logit_scale 約 100，任務內 softmax 接近 one-hot，結果由分派權重決定 |

儲存與時間：每任務 expert 132,097 參數（528,388 bytes）；proto key 2,048 bytes／任務；零 replay。

---

## 2. T-Hard 原始程式查核（AaronHung/T-hard，fork 自 Russell512/T-hard，3dfb128）

| 項目 | 內容 |
|---|---|
| 分派規則 | slide 全部 patch 特徵平均正規化，對「已見任務的文字 prototype」取 cosine 最大者；任務文字 prototype = 該任務 2 個類別文字 embedding 的平均再正規化。不存任何影像資料 |
| 用途 | 把選到的任務 prototype 當成 PathSelect selector 的查詢向量，不是用來選 expert |
| 實驗設定 | PathSelect 全流程，含 replay（M128／M512），8 張 H200 |
| 十折正式結果 | M128：T-Hard 80.84 對 Q0 80.74（+0.10 pp，95% CI [−0.89, +1.03]）；M512：81.76 對 82.60（−0.85 pp）；Task-IL 約 91.0–91.3% |
| 和我們的 text-class 的差別 | 我們取「最像的類別」所屬的任務；T-Hard 取「最像的任務平均文字」。R 線把 T-Hard 的規則照原樣加進去（R0），並核對兩邊 f_txt cache 的 SHA256 相同 |

---

## 3. 9/25 的決定

| # | 項目 | 決定 |
|---|---|---|
| 1 | 期中報告與 QPMIL | 先用乾淨頭把結果做清楚，再決定怎麼和老師溝通 |
| 2 | 什麼算 replay | 不存 slide、不存 patch 特徵，就不算 replay；每任務存少量統計量照實揭露。與 exemplar-free CIL 的慣例一致（FeCAM 只存類別統計量） |
| 3 | expert 的遺忘 | expert 各自獨立，WP 不會忘；CIL 的遺忘改用「任務增加後分派正確率下降多少」來衡量 |
| 4 | TP 的主張 | 不存資料為第一優先 |
| 5 | 場次 | ISBI 2027 |
| 6 | 命名 | 每任務選片器稱 expert（程式裡 selector），分派到 expert 的模組稱 router |

研發方式：機制一個一個加，每加一個都記錄「介入了什麼 → 內部狀態怎麼變 → 行為怎麼變 → 指標怎麼反映」。

---

## 4. 第四關設計（PREREG-2，開跑前 commit，之後不改）

共同設定：expert 用第一關的 bank（十折），四輪選片（K = 64、每輪 16、λ* = 1.5），Hard 組合；Mac CPU，同一批。

### 4.1 R 線：換 router（只推論）

| 代號 | router | 每任務儲存 |
|---|---|---|
| R0 | T-Hard 原規則：slide 平均對「任務文字 prototype（2 個類別文字平均）」取最大 | 不存 |
| R1 | patch-vote | 不存 |
| R2 | proto（第三關的 0.8900，當參照） | 2 KB |
| R3 | multi-proto：每任務訓練 slides 的平均向量做 k-means，k ∈ {2, 4, 8} | k × 2 KB |
| R4 | proto 與 patch-vote 分數各自 z-normalize 後相加 | 2 KB |
| R5 | evidence-proto（N1 修正版）：每任務存「expert τ 在 τ 自己訓練 slides 上選中 64 張的平均」 | 2 KB |

| 判準 | 內容 |
|---|---|
| R-pass | validation 選出的 router，儲存 ≤ 16 KB／任務、不存任何 slide 資料，test CIL ACC ≥ 0.890 |
| R-改進 | 相對 R2 的每折 paired 差 ≥ +0.01，且贏 ≥ 7/10 折 |
| 無儲存 router | R0、R1 中較好者比 zero-shot 8 類 top-64（0.8270）高 ≥ 0.02，且贏 ≥ 7/10 折 |

### 4.2 L 線：LoRA expert（N2，需要訓練）

| 代號 | expert | 每任務參數 |
|---|---|---|
| L0 | 完整 expert（第一關 bank） | 132,097 |
| L1(r) | 第一個任務的完整 expert 當底座並凍結；之後每任務只學第一層的 rank-r 增量 BA、第一層 bias 與第二層；r ∈ {1, 2, 4, 8} | 770r + 513 |
| L2 | 各任務增量相加成一個合併 expert，不用 router | 一份 |

| 判準 | 內容 |
|---|---|
| L-pass | 最小的 r 使四任務平均 WP ≥ L0 − 0.01（十折、兩序都要），且 r ≤ 4 |

### 4.3 每個介入要記錄什麼

| 介入 | 內部狀態 | 行為 | 指標 |
|---|---|---|---|
| 換 router | 每張 slide 對各候選任務的分數與 margin | 被派到哪個 expert（4×4 混淆矩陣） | TP、CIL |
| 完整 expert → LoRA r | 與 L0 的 patch 分數相關係數；增量 BA 的 Frobenius norm | 選中 64 張與 L0 的 Jaccard | WP、每任務參數 |
| 合併增量、拿掉 router（L2） | 合併後分數分布 | 選片與各任務 expert 的 Jaccard | CIL |

---

## 7. 第四關結果判讀（NC-2，navrouter-cil 3a9e622；PREREG-2 f79ce66 開跑前 commit）

### 7.1 判準落點

| 判準 | 數值 | 結果 |
|---|---|---|
| 主 router（validation 選） | R3(k=8)，validation ACC 0.8942 | — |
| R-pass | 16,384 bytes／任務；test ACC 0.8987 ≥ 0.890 | 通過 |
| R-改進（對 R2） | +0.0086，贏 7/10 | 未通過（差不到 +0.01） |
| 無儲存 router（R0 對 zero-shot 8 類 top-64） | +0.0348，贏 9/10 | 通過 |
| L-pass | r = 2、4、8 都達標；r = 1 出現 NaN | r* = 2（r = 1 修正重跑後定案） |

### 7.2 R 線（test，t = 4）

| 變體 | 每任務儲存 | TP macro | CIL ACC | 對 R2（贏折數） |
|---|---|---|---|---|
| R0 T-Hard 原規則 | 0 | 0.9154 | 0.8617 | −0.0283（0） |
| R1 patch-vote | 0 | 0.9070 | 0.8540 | −0.0361（0） |
| R2 proto | 2,048 | 0.9454 | 0.8900 | — |
| R3(k=4) | 8,192 | 0.9545 | 0.8990 | +0.0090（7） |
| R3(k=8)，主 router | 16,384 | 0.9527 | 0.8987 | +0.0086（7） |
| R4 proto＋patch-vote | 2,048 | 0.9241 | 0.8686 | −0.0214（2） |
| R5 evidence-proto | 2,048 | 0.8696 | 0.8164 | −0.0737（0） |

### 7.3 L 線（L0 的 WP = 0.9399，門檻 0.9299）

| r | 非底座任務參數 | WP reverse／paper | L2（合併、無 router）ACC reverse／paper |
|---|---|---|---|
| 1 | 1,283 | 無效（NaN） | 無效 |
| 2 | 2,053 | 0.9401／0.9339 | 0.6951／0.6574 |
| 4 | 3,593 | 0.9409／0.9358 | 0.7004／0.6694 |
| 8 | 6,673 | 0.9410／0.9330 | 0.7056／0.6813 |

與 L0 比：patch 分數的 Pearson 相關約 0.73–0.93，四輪選中 64 張的 Jaccard 只有約 0.10–0.29。

### 7.4 完整系統列：R3(k=8)＋L1(r=2)

| 序 | ACC | Masked ACC | Forgetting | QPMIL-VL 已發表（非 paired） |
|---|---|---|---|---|
| reverse | 0.8987 | 0.9297 | 0.0227 | 0.859／0.925／0.064 |
| paper（forward） | 0.8940 | 0.9228 | 0.0287 | 0.890／0.930／0.027 |

### 7.5 判讀

| # | 發現 | 對論文的意義 |
|---|---|---|
| 1 | 老師的計畫（TIL → CIL → LoRA／MoE）第一輪完整跑通：reverse 0.8987、forward 0.8940，零 replay | reverse 高於 QPMIL-VL 已發表值 4 pp；forward 持平 |
| 2 | 順序穩健：兩序只差 0.005 | expert 各自獨立、router key 只用各自任務的資料 |
| 3 | T-Hard 規則是最好的無儲存 router（0.8617，贏 zero-shot 9/10 折） | 老師提的想法在這個架構裡成立 |
| 4 | 合併 expert（L2）只有 0.66–0.71 | router 是必要的；和 MergeSlide 這類合併路線的區隔 |
| 5 | LoRA r = 2 用完整 expert 1.6% 的參數，WP 不降 | 每任務的選片差異可以用很低的秩表示 |
| 6 | LoRA 與完整 expert 選的 patch 大部分不同（Jaccard 0.10–0.29），WP 卻相同 | 有很多張 patch 組合一樣有效；選片重疊率不是可靠的代理指標 |
| 8 | expert 證據當 router（R5、nav）兩次都失敗 | expert 挑的是分類證據（腫瘤），分辨器官要看組織背景 |
| 9 | 剩下的差距：oracle 0.9399 對 0.8987，約 4 pp，全部在分派 | 第五關的機制都針對這一段，特別是 ESCA↔Lung |

---

## 8. r = 1 的 NaN（AMENDMENT-1）

| 項目 | 內容 |
|---|---|
| 現象 | r = 1 的 60 個增量有 51 個變成 NaN；只在 A 梯度的第 512、513 欄；與 CPU 執行緒數有關 |
| 判斷 | rank-1 反向傳播的數值運算問題（BLAS gemv），不是訓練發散 |
| 修法 | r = 1 改用廣播乘法；所有訓練每步檢查 NaN／Inf；固定 8 條執行緒；整張 L 表重跑（v2） |

---

## 9. 第五關（PREREG-3）：只動 router，目標 ESCA↔Lung

| 代號 | 假設 | 做法 | 每任務儲存 |
|---|---|---|---|
| R6 patch-codebook vote | slide 平均把組織背景平均掉了；看每個 patch 屬於哪個任務的組織 | 每任務從訓練 patch 取樣做 k-means（k = 8），測試時 patch 投票 | 16 KB |
| R7 patch 對角高斯 | 同上，但用分布 | 每任務存 patch 特徵的平均與變異數，patch 依對角 Mahalanobis 距離投票 | 4 KB |
| R8 top-2 會診（MoE top-k） | router 沒把握時，讓兩個 expert 各自看、再比較 | margin 小於門檻時跑前兩名 expert，在 4 個類別中取最大 | 0 |

## 10. 第五關結果判讀（NC-3，navrouter-cil bee5519）

| 文件 | 判準 | 結果 |
|---|---|---|
| AMENDMENT-1 | r = 1 在 1／3／8 條執行緒下都沒有 NaN；r = 2 新舊寫法 50 步參數差 0 | 通過 |
| PREREG-2（v2） | L-pass | 通過，**r* = 2 定案** |
| PREREG-3 | R-機制 | 未通過：validation 選出的是基準 R3(k=8) 本身 |

| 變體 | 每任務儲存 | ACC | 對基準（贏折數） | ESCA→Lung／Lung→ESCA |
|---|---|---|---|---|
| 基準 R3(k=8) | 16,384 | 0.8987 | — | 5／39 |
| R6 codebook 投票 | 16,384 | 0.8905 | −0.0082（3） | 3／45 |
| R7 patch 對角高斯 | 4,096 | 0.8808 | −0.0178（1） | 2／102 |
| R8 top-2 會診（q* = 0） | 16,384 | 0.8986 | −0.0000（2） | 5／39 |

| # | 發現 | 意義 |
|---|---|---|
| 1 | 用 expert 的證據決定任務，三種做法都失敗：nav、R5、R8 | expert 挑的是診斷證據，各任務的類別文字相似度不能跨任務比較 |
| 2 | patch 層級的投票（R1、R6、R7）都不如整張片子的平均向量 | 任務身分主要在整張片子的組織組成；局部組織（鱗狀上皮區）反而讓肺與食道更像 |
| 3 | 基準 R3 的主要錯誤其實是 Lung→ESCA（39 張），不是 ESCA→Lung（5 張） | 報告與論文要照各 router 分開寫 |
| 5 | 聯合訓練的線性 probe（非 CL，只當上限）在同樣的 slide 平均特徵上 micro TP 0.982 | 最近原型 router 還沒到這個上限（之後 NC-5 的閉式 ridge 分派器 AR 達 0.986） |

---

## 11. NC-4 結果判讀（navrouter-cil f77488a；PREREG-4 88532d6）

| 判準 | 結果 | 數字 |
|---|---|---|
| I7 容量依需求（每任務用 validation 在 {zero-shot、底座、r=1,2,4,8} 挑最省的） | 未通過 | test WP reverse 0.9335、paper 0.9279，比固定 r = 2 低 0.0066、0.0060 |
| B1 撞車加 key | 未通過 | CIL reverse −0.0228（1/10）、paper −0.0077（2/10） |
| B2 假特徵線性 router | 未通過 | CIL reverse −0.0020（6/10）、paper −0.0008（4/10） |

| # | 發現 | 意義 |
|---|---|---|
| 1 | I7 的 validation 常選「不訓練」：ESCA（paper 序）10 折有 9 折選 zero-shot | 有些任務 VLM 本身就夠用；這個觀察後來導出 I6（zero-shot 分數＋低秩修正，即修正頭） |
| 2 | 到目前為止，9 種 router 改法都沒贏 R3(k=8) | 「平均向量＋最近原型」的 router 已接近瓶頸；要換比法（NC-5 的閉式 ridge） |
| 3 | LoRA 那一側下一個候選改為 I6 | 同時處理「底座來自第一個任務、與順序有關」的問題 |

---

## 本頁代號對照

| 代號 | 意思 |
|---|---|
| TIL／WP、CIL、TP | 告訴任務時的 2 類正確率；不告訴任務時的 8 類正確率；任務辨識正確率 |
| Hard、Hard-8、Soft | router 選一個任務後在 2 類裡判；在 8 類裡判；依機率加權 |
| R0～R8 | 各種 router：R0 = T-Hard 規則；R2 = 每任務 1 個平均向量；R3 = 每任務 k 個平均向量；R5、nav、R8 = 用 expert 的證據；R1、R6、R7 = patch 投票 |
| L0、L1(r)、L2 | 完整 expert；共用底座＋每任務 rank r 的 LoRA；全部合併、不用 router |
| r、r\* | LoRA 的秩；事先訂的「不掉超過 0.01 的最小 r」，結果為 2 |
| N1、N2 | 用 expert 證據判任務（沒成立）；共享底座＋LoRA（成立） |
| I4、I5、I7 | 假特徵訓練 router；只在撞車處加 key；容量依需求 |
| PREREG、AMENDMENT-1 | 開跑前寫死的判準；修 r = 1 數值錯誤的紀錄 |
| PI | 文件中指 Aaron，不是老師 |

Sources:
- T-hard（fork）: https://github.com/AaronHung/T-hard
- FeCAM: https://arxiv.org/abs/2309.14062
- Kim et al., NeurIPS 2022: https://arxiv.org/abs/2211.02633
