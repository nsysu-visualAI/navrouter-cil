#!/usr/bin/env python3
"""EXT-3（PREREG-23 細則 3–7）：ladder 兩格的逐折配對、bootstrap 與報告。只讀 lad1_cells.py 的逐折輸出（lad1/cells/），不重跑、不讀快取。

  配對  MV_RDG − LIN8、MV_RDG − FINALB、B_TXT − FINALB：ext1_d.sign_pair（符號檢定）＋ t = 4 CIL ACC 的 ext1_d.bootstrap
  判讀  PREREG-23 第一部分的規則（顯著 = t = 4 CIL ACC 的符號檢定 p < 0.05）

    NAVCIL_MACHINE=mac python scripts/ext3_report.py
輸出：outputs/navcil/<machine>/ext3/stats.json、EXT3_perfold.csv、REPORT_ext3.md
"""
from __future__ import annotations

import csv
import json
import os
import sys
import traceback
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import ext1_d as DD                                                       # noqa: E402
from ext1_c import D64, M                                                 # noqa: E402

ORDERS, FOLDS = DD.ORDERS, DD.FOLDS
BASE = REPO_ROOT / "outputs" / "navcil" / os.environ["NAVCIL_MACHINE"]
CELLS, OUT = BASE / "lad1" / "cells", BASE / "ext3"
ROWS = ["LIN8", "MV_RDG", "FINALB", "B_TXT"]
WHAT = {"LIN8": "direct 8-class：[mean_vec; 1]／8 類 ridge（γ = 0.01）、8 類 argmax、無 TP",
        "MV_RDG": "格 1：TP = AR；[mean_vec; 1]／8 類 ridge（γ = 1e-3）、τ̂ 兩類內 argmax",
        "FINALB": "FINAL-B：TP = AR；v0（s = s0、四輪各 16）；[v0; 1]／8 類 ridge（γ = 1e-3）、τ̂ 兩類內 argmax",
        "B_TXT": "格 2：TP = AR；v0 同 FINAL-B；v0 對 τ̂ 兩類文字的 cosine、取較大者"}
PAIRS = [("MV_RDG", "LIN8"), ("MV_RDG", "FINALB"), ("B_TXT", "FINALB")]
METRICS = [("ACC", "ACC"), ("WP", "MaskedACC_oracle"), ("MaskedACC_table1", "MaskedACC_table1"), ("Forgetting", "Forgetting"),
           ("BWT", "BWT"), ("Abar", "Abar")]
HEAD = {"ACC": "CIL ACC", "WP": "WP", "MaskedACC_table1": "Masked ACC（Table 1）", "Forgetting": "Forgetting", "BWT": "BWT", "Abar": "Ā"}
# fp32 儲存（共用 S、每任務 p）；DECISIONS D11
STORAGE = {"LIN8": (1_052_676, 4_104), "FINALB": (2_105_352, 10_252), "MV_RDG": (1_052_676, 6_156), "B_TXT": (1_052_676, 6_148)}
ALPHA, NEAR = 0.05, 0.01


def row(name: str, o: str, f: int) -> dict:
    return json.loads((CELLS / name / f"{o}_fold{f}.json").read_text())


def series(name: str, o: str, key: str) -> list[float]:
    return [DD.METRICS[key](row(name, o, f)) for f in FOLDS]


def ok(name: str, o: str) -> list[torch.Tensor]:
    return [torch.tensor(row(name, o, f)["ok4"], dtype=D64) for f in FOLDS]


def tasks_per_fold() -> list[torch.Tensor]:
    """D8：每張 test slide 的任務取自 ext2/a/FINALB 的逐折 JSON；先核對其 ok4 與本批 FINALB 逐張相同。"""
    out = []
    for f in FOLDS:
        ref = json.loads((BASE / "ext2" / "a" / "FINALB" / f"{ORDERS[0]}_fold{f}.json").read_text())
        if ref["ok4"] != row("FINALB", ORDERS[0], f)["ok4"] or any(len(row(r, ORDERS[0], f)["ok4"]) != len(ref["task"]) for r in ROWS):
            raise M.CheckFailed(f"fold {f}: ext2/a 的 FINALB 與 lad1/cells 的逐張判定或張數不一致")
        out.append(torch.tensor(ref["task"]))
    return out


def run() -> dict:
    tasks_pf = tasks_per_fold()
    summary = {o: {r: {mk: DD.ms(series(r, o, key)) for mk, key in METRICS} for r in ROWS} for o in ORDERS}
    pairs = {}
    for a, b in PAIRS:
        pairs[f"{a} − {b}"] = {}
        for o in ORDERS:
            cell = {mk: DD.sign_pair(series(a, o, key), series(b, o, key)) for mk, key in METRICS}
            cell["ACC"]["boot_ci95"] = DD.bootstrap(None, tasks_pf, ok(a, o), ok(b, o))
            pairs[f"{a} − {b}"][o] = cell
    same = {r: max(abs(x - y) for key in ("ACC", "MaskedACC_oracle", "MaskedACC_table1")
                   for x, y in zip(series(r, ORDERS[0], key), series(r, ORDERS[1], key))) for r in ROWS}

    def sig(c):
        return c["p_sign"] is not None and c["p_sign"] < ALPHA

    c1 = {o: pairs["MV_RDG − LIN8"][o]["ACC"] for o in ORDERS}
    c2 = {o: pairs["B_TXT − FINALB"][o]["ACC"] for o in ORDERS}
    verdict = {
        "cell1": {o: {"mean": summary[o]["MV_RDG"]["ACC"][0], "is_0.903": round(summary[o]["MV_RDG"]["ACC"][0], 3) == 0.903,
                      "sig_vs_LIN8": sig(c1[o]), "fallback": not sig(c1[o])} for o in ORDERS},
        "cell2": {o: {"mean_diff": c2[o]["mean_diff"], "sig_vs_FINALB": sig(c2[o]), "lower": c2[o]["mean_diff"] < 0,
                      "fallback": abs(c2[o]["mean_diff"]) < NEAR and not sig(c2[o])} for o in ORDERS},
        "cell1_vs_FINALB": {o: {"mean_diff": pairs["MV_RDG − FINALB"][o]["ACC"]["mean_diff"], "sig": sig(pairs["MV_RDG − FINALB"][o]["ACC"])}
                            for o in ORDERS}}
    return {"rows": WHAT, "summary": summary, "pairs": pairs, "order_t4_max_abs_diff": same, "verdict": verdict,
            "n_boot": DD.N_BOOT, "boot_seed": DD.BOOT_SEED, "alpha": ALPHA}


def write_csv() -> None:
    def fm(x):
        return "" if x is None else f"{x:.10f}"

    with (BASE / "EXT3_perfold.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["row", "order", "fold", "t", "CIL_ACC", "WP", "MaskedACC_table1", "MaskedACC_oracle", "Forgetting", "BWT", "storage_bytes"])
        for name in ROWS:
            sp = STORAGE[name]
            for o in ORDERS:
                for f in FOLDS:
                    r = row(name, o, f)
                    for t in range(4):
                        w.writerow([name, o, f, t + 1, fm(r["acc_t"][t]), fm(r["wp_t"][t]), fm(r["mk1_t"][t]), fm(r["wp_t"][t]),
                                    fm(r["forgetting"]) if t == 3 else "", fm(r["bwt"]) if t == 3 else "", sp[0] + (t + 1) * sp[1]])


def pm(v) -> str:
    return f"{v[0]:.4f} ± {v[1]:.4f}"


def yn(x: bool) -> str:
    return "是" if x else "否"


def write_report(res: dict) -> None:
    pv = json.loads((BASE / "lad1" / "paired.json").read_text())
    p, ka, kb = pv["provenance"], pv["KA"], pv["KB"]
    S, PR, V = res["summary"], res["pairs"], res["verdict"]
    L = ["# REPORT EXT-3 — 論文 ladder 的兩格（十折 × 兩序；PREREG-23）", "",
         "只給數字與表格。兩格與兩個對照列出自同一台、同一次執行；本報告只讀該次執行的逐折輸出，不重跑。", "",
         "## 狀態", "",
         "| 項目 | 狀態 |", "|---|---|",
         "| 兩格的 run（十折 × 兩序，t = 1…4） | 完成，無失敗 |",
         "| 格 1 對到既有的列 | 完成：不在 `ABLATION_full.csv`；＝ REPORT_moe3 F5「全部 patch × S-R0」；不是 M2（D4） |",
         "| 格 2 對到既有的列 | 完成：＝ `M2_g0` 補上 CIL 與 t < 4；AMENDMENT-7（D5） |",
         "| 逐折配對、bootstrap、判讀 | 完成 |",
         "| PREREG-23 先於任何數字回報 | **未做到**：PREREG-23 寫於 run 之後，且先前已在聊天回報過一次數字（D1） |", "",
         "## 0　執行", "",
         "| 項目 | 值 |", "|---|---|",
         f"| 機器 | {p['hw_model']}（{p['cpu']}，{int(p['mem_bytes']) // 2**30} GB）；{p['os']} |",
         f"| 環境 | Python {p['python']}、torch {p['torch']}、`--device {p['device']}`、threads {p['threads']} |",
         f"| run 的程式 commit | `{p['commit']}`（scripts／selector／data／configs 無未 commit 變更：{yn(not p['dirty'])}） |",
         "| run 的指令 | `NAVCIL_MACHINE=mac scripts/run_stage.sh lad1_cells /Users/aaron/venvs/navcil/bin/python scripts/lad1_cells.py --device cpu`"
         "（tmux ＋ `caffeinate -ims`） |",
         f"| run 完成時間、耗時 | {pv['finished']}，{pv['wall_s']} 秒；log `outputs/navcil/mac/logs/lad1_cells.log`（exit=0） |",
         "| 本報告的指令 | `NAVCIL_MACHINE=mac /Users/aaron/venvs/navcil/bin/python scripts/ext3_report.py` |",
         f"| run 內查核 KA | FINALB、LIN8 對 `ext2/a/` 逐折最大絕對差 {max(ka['per_row_max_abs'].values()):.1e}（容許 {ka['tol']:.0e}）：{'通過' if ka['pass'] else '不過'} |",
         f"| run 內查核 KB | MV_RDG 的 t = 4 CIL、WP、Ā 對 `moe3/f5.json` 的 S-R0 × mv 最大絕對差 {kb['max_abs']:.1e}（容許 {kb['tol']:.0e}）：{'通過' if kb['pass'] else '不過'} |",
         "", "## 1　列的定義與對到的既有列", "",
         "| row | 定義 | 既有的列 |", "|---|---|---|",
         f"| LIN8 | {WHAT['LIN8']} | `ABLATION_full.csv` 的 `LIN8`；本批重算，差 0 |",
         f"| MV_RDG | {WHAT['MV_RDG']} | 不在 `ABLATION_full.csv`；REPORT_moe3 F5「全部 patch × S-R0」（只有 t = 4 的 WP、CIL、Ā）；差 0 |",
         f"| FINALB | {WHAT['FINALB']} | `FINALB_perfold.csv` 的 `FINALB`（＝ `ABLATION_full.csv` 的 `NOHEAD`）；本批重算，差 0 |",
         f"| B_TXT | {WHAT['B_TXT']} | `ABLATION_full.csv` 的 `M2_g0`（只有 t = 4 的 WP）；WP 逐折差 0；CIL 與 t < 4 為本批補算（AMENDMENT-7） |",
         "", "格 1 的設定確認：TP 與讀出都只吃 mean_vec（全部 patch 平均後 L2 正規化）；沒有文字選片、沒有 head。"
         "格 1 的讀出 γ = 1e-3 沿用 MOE-3 F5（u_64 選到的值，未對 mean_vec 重選）；LIN8 的 γ = 0.01（D6）。", "",
         "## 2　t = 4（十折 mean ± sd；test）", ""]
    for o in ORDERS:
        L += [f"### 序 {o}", "", "| row | " + " | ".join(HEAD[mk] for mk, _ in METRICS) + " |", "|---|" + "---|" * len(METRICS)]
        L += [f"| {r} | " + " | ".join(pm(S[o][r][mk]) for mk, _ in METRICS) + " |" for r in ROWS] + [""]
    L += [f"兩序在 t = 4 的 CIL ACC、WP、Masked ACC 逐折相同（四列最大絕對差 {max(res['order_t4_max_abs_diff'].values()):.1e}）；"
          "Forgetting、BWT、Ā 兩序不同（D10）。", "", "## 3　逐折（t = 4）", ""]
    for r in ROWS:
        L += [f"### {r}", "", "| fold | CIL ACC | Masked ACC（Table 1） | Forgetting（reverse） | BWT（reverse） | Forgetting（paper） | BWT（paper） |",
              "|---|---|---|---|---|---|---|"]
        for i, f in enumerate(FOLDS):
            a, b = row(r, "reverse", f), row(r, "paper", f)
            L.append(f"| {f} | {a['acc_t'][3]:.4f} | {a['mk1_t'][3]:.4f} | {a['forgetting']:.4f} | {a['bwt']:.4f} | {b['forgetting']:.4f} | {b['bwt']:.4f} |")
        L.append(f"| mean ± sd | {pm(S['reverse'][r]['ACC'])} | {pm(S['reverse'][r]['MaskedACC_table1'])} | {pm(S['reverse'][r]['Forgetting'])} | "
                 f"{pm(S['reverse'][r]['BWT'])} | {pm(S['paper'][r]['Forgetting'])} | {pm(S['paper'][r]['BWT'])} |")
        L.append("")
    L += ["CIL ACC 與 Masked ACC 兩序逐折相同，只列一欄。t = 1…3 的逐折值在 `EXT3_perfold.csv`。", "",
          "## 4　逐折配對（t = 4；同折、同序；差 = 前者 − 後者）", "",
          "贏／輸／平手 = 前者數值較大／較小／相同的折數（Forgetting 的「贏」＝遺忘較多）；p = exact binomial 雙尾符號檢定（平手不計）；"
          f"bootstrap = slide 層級、(折, 任務) 分層、配對、{res['n_boot']} 次、seed {res['boot_seed']}，只做 CIL ACC。", ""]
    for name, per in PR.items():
        for o in ORDERS:
            L += [f"### {name}（序 {o}）", "", "| 指標 | 前者 | 後者 | 平均差 | 贏／輸／平手 | p（符號檢定） | bootstrap 95% CI |", "|---|---|---|---|---|---|---|"]
            for mk, _ in METRICS:
                c = per[o][mk]
                ci = f"[{c['boot_ci95'][0]:+.4f}, {c['boot_ci95'][1]:+.4f}]" if "boot_ci95" in c else "—"
                pv_ = "不適用" if c["p_sign"] is None else f"{c['p_sign']:.4f}"
                L.append(f"| {HEAD[mk]} | {pm(c['a'])} | {pm(c['b'])} | {c['mean_diff']:+.4f} | {c['wins']}／{c['losses']}／{c['ties']} | {pv_} | {ci} |")
            L.append("")
    L += ["## 5　PREREG-23 的判讀", "", f"顯著 = t = 4 CIL ACC 的逐折符號檢定 p < {res['alpha']}。兩序的 t = 4 CIL ACC 逐折相同，判讀結果相同，下表列 reverse。", "",
          "| 項目 | 預期 | 結果 | 是否觸發備案 |", "|---|---|---|---|"]
    o = ORDERS[0]
    a, b, c = PR["MV_RDG − LIN8"][o]["ACC"], PR["B_TXT − FINALB"][o]["ACC"], PR["MV_RDG − FINALB"][o]["ACC"]
    L += [f"| 格 1 的 CIL ACC | 約 0.903 | {V['cell1'][o]['mean']:.4f}（三位小數為 0.903：{yn(V['cell1'][o]['is_0.903'])}） | — |",
          f"| 格 1 − LIN8 | 顯著 | {a['mean_diff']:+.4f}；贏／輸／平手 {a['wins']}／{a['losses']}／{a['ties']}；p = {a['p_sign']:.4f}；"
          f"顯著：{yn(V['cell1'][o]['sig_vs_LIN8'])} | {yn(V['cell1'][o]['fallback'])}（備案：寫成「分解本身增益有限」） |",
          f"| 格 2 − FINAL-B | 低約 3 pp 且顯著 | {b['mean_diff']:+.4f}（{b['mean_diff'] * 100:+.2f} pp）；贏／輸／平手 {b['wins']}／{b['losses']}／{b['ties']}；"
          f"p = {b['p_sign']:.4f}；顯著：{yn(V['cell2'][o]['sig_vs_FINALB'])} | {yn(V['cell2'][o]['fallback'])}（備案：差不到 1 pp 且不顯著） |",
          f"| 格 1 − FINAL-B | 無預期 | {c['mean_diff']:+.4f}；贏／輸／平手 {c['wins']}／{c['losses']}／{c['ties']}；p = {c['p_sign']:.4f}；"
          f"顯著：{yn(V['cell1_vs_FINALB'][o]['sig'])}；bootstrap 95% CI [{c['boot_ci95'][0]:+.4f}, {c['boot_ci95'][1]:+.4f}] | — |",
          "", "## 未完成項與原因", "",
          "- 「回報任何數字之前先 commit PREREG-23」：未做到（D1）。", "- 其餘無。", "",
          "## 失敗", "", "無。", "",
          "## DECISIONS.md（`ext3/DECISIONS.md` 全文）", "", (OUT / "DECISIONS.md").read_text().rstrip(), ""]
    (BASE / "REPORT_ext3.md").write_text("\n".join(L))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        res = run()
        res.update({"stage": "ext3", "finished": M.now(), "source": "lad1/cells"})
        (OUT / "stats.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=M._default))
        write_csv()
        write_report(res)
    except Exception:
        (OUT / "FAILED_report.txt").write_text(f"[{M.now()}] 階段 report 失敗\n{traceback.format_exc()}")
        traceback.print_exc()
        return 3
    for name, per in res["pairs"].items():
        for o in ORDERS:
            r = per[o]["ACC"]
            print(o, name, f"{r['mean_diff']:+.4f}", f"{r['wins']}／{r['losses']}／{r['ties']}", r["p_sign"], r["boot_ci95"])
    print("EXT-3：完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
