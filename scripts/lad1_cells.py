#!/usr/bin/env python3
"""LAD-1：論文 ladder 的兩格（十折、兩序、t = 1…4；只讀既有快取，不訓練、不載入 head 權重來算這兩格）。

  MV_RDG  格 1：TP = AR（[mean_vec; 1]、γ = 1e-3）；判讀 = [mean_vec; 1] 累加式 8 類 ridge（γ = 1e-3）、τ̂ 兩類內 argmax
          （= MOE-3 F5「全部 patch」的 S-R0；沒有文字選片、沒有 head）
  B_TXT   格 2：與 FINAL-B 相同（AR、s = s0、四輪各 16、λ = 1.5 的 v0），只把判讀換成 v0 對 τ̂ 兩類文字的 cosine、取較大者
  對照    FINALB（v0／RDG γ = 1e-3）、LIN8（mean_vec／8 類 ridge γ = 0.01、無 TP）在同一次執行重算

查核（不過即停，不組表）：
  KA  FINALB、LIN8 的逐折值對 ext2/a（差 ≤ 1e-9）
  KB  MV_RDG 的 t = 4 CIL、WP 與 Ā 對 moe3/f5.json 的 S-R0 × mv（差 ≤ 1e-9）
配對：格 1 − LIN8、格 1 − FINALB、格 2 − FINALB；每序十折的平均差、95% CI（t 分布，自由度 9）、勝場數、paired t-test p。

    NAVCIL_MACHINE=mac python scripts/lad1_cells.py --device cpu [--folds 1-10]
輸出：outputs/navcil/<machine>/lad1/cells/<row>/<order>_fold<f>.json（＋ .done）、lad1/LADDER_perfold.csv、lad1/paired.json、
      lad1/REPORT_lad1.md、lad1/cells.done
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch
from scipy import stats as sps

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import ext1_c as CC                                                       # noqa: E402
import ext2_a as A2                                                       # noqa: E402
from ext1_c import C, M, P, T, log                                        # noqa: E402

ORDERS = CC.ORDERS
G = 1e-3
ROWS = ["LIN8", "MV_RDG", "FINALB", "B_TXT"]
WHAT = {"LIN8": "direct 8-class：mean_vec／8 類 ridge（γ = 0.01）、無 TP",
        "MV_RDG": "格 1：AR ＋ mean_vec／ridge（γ = 1e-3）、τ̂ 兩類內",
        "FINALB": "B：AR ＋ v0（四輪 64）／ridge（γ = 1e-3）、τ̂ 兩類內",
        "B_TXT": "格 2：AR ＋ v0（四輪 64）／對 τ̂ 兩類文字的 cosine"}
PAIRS = [("MV_RDG", "LIN8"), ("MV_RDG", "FINALB"), ("B_TXT", "FINALB")]
# (鍵, 表頭, 數值越大越好)
METRICS = [("acc4", "CIL ACC（t = 4）", True), ("mk4", "Masked ACC（t = 4；Table 1 定義）", True), ("forgetting", "Forgetting", False)]
TOL = 1e-9


class Ctx(CC.Ctx):
    """ext1_c.Ctx 的唯讀來源不變；輸出寫到 lad1/cells/。"""

    def __init__(self):
        super().__init__()
        self.out = self.base / "lad1"
        self.root = self.out / "cells"


def txt_first(cx: Ctx, D: dict, kind: str):
    """TXT：pv 所指任務版本的向量對任務 pc 兩類文字的 cosine，第一類 ≥ 第二類即判第一類（同 ext1_c.do_fold 的 txt_first）。"""
    n = torch.arange(len(D["task"]))

    def first(t, pv, pc):
        return C.pair_diff(T.cos8(cx.r3, D, kind)[n, pv], pc) >= 0
    return first


def do_fold(cx: Ctx, f: int, o: str) -> None:
    CC.do_fold(cx, f, o, ["LIN8"])
    D = T.data(cx.r3, "test", f)
    task, label = D["task"], D["label"]
    ar = T.ar_stages(cx.r3, f, o, D)
    firsts = {"MV_RDG": A2.first_of(cx, f, o, D, "mv", G),
              "FINALB": A2.first_of(cx, f, o, D, A2.KIND_B, G),
              "B_TXT": txt_first(cx, D, A2.KIND_B)}
    for row, first in firsts.items():
        cx.save(row, o, f, CC.cl_eval(first, ar, task, label, o, cx.tasks))


def load(cx: Ctx, row: str, o: str, f: int) -> dict:
    return json.loads(cx.path(row, o, f).read_text())


def check_ka(cx: Ctx, folds: list[int]) -> dict:
    ref, per = cx.base / "ext2" / "a", {}
    for row in ("FINALB", "LIN8"):
        m = 0.0
        for o in ORDERS:
            for f in folds:
                a, b = load(cx, row, o, f), json.loads((ref / row / f"{o}_fold{f}.json").read_text())
                for k in A2.K9_KEYS:
                    m = max([m] + [abs(x - y) for x, y in zip(A2.flat(a[k]), A2.flat(b[k]))])
        per[row] = m
    return {"per_row_max_abs": per, "tol": TOL, "pass": bool(max(per.values()) <= TOL)}


def check_kb(cx: Ctx, folds: list[int]) -> dict:
    j = json.loads((cx.base / "moe3" / "f5.json").read_text())
    m = 0.0
    for o in ORDERS:
        for f in folds:
            a, b = load(cx, "MV_RDG", o, f), j["orders"][o]["S-R0"]["mv"][j["folds"].index(f)]
            m = max(m, abs(a["acc_t"][3] - b["cil"]), abs(a["wp_t"][3] - b["wp"]), abs(C.mean(a["acc_t"]) - b["abar"]))
    return {"reader": j["readers"]["S-R0"], "max_abs": m, "tol": TOL, "pass": bool(m <= TOL and j["readers"]["S-R0"] == ["RDG", G])}


def values(cx: Ctx, folds: list[int]) -> dict:
    """{row: {order: {metric: 逐折值}}}"""
    out = {}
    for row in ROWS:
        out[row] = {}
        for o in ORDERS:
            rs = [load(cx, row, o, f) for f in folds]
            out[row][o] = {"acc4": [r["acc_t"][3] for r in rs], "mk4": [r["mk1_t"][3] for r in rs],
                           "wp4": [r["wp_t"][3] for r in rs], "forgetting": [r["forgetting"] for r in rs],
                           "bwt": [r["bwt"] for r in rs]}
    return out


def paired(a: list[float], b: list[float], higher: bool) -> dict:
    """逐折差 D = a − b：平均、樣本標準差、95% CI（t 分布）、a 較好／較差／平手的折數、paired t-test（雙尾）。"""
    d = [x - y for x, y in zip(a, b)]
    n, m = len(d), C.mean(d)
    sd = statistics.stdev(d) if n > 1 else 0.0
    better = sum((x > 1e-12) if higher else (x < -1e-12) for x in d)
    worse = sum((x < -1e-12) if higher else (x > 1e-12) for x in d)
    if n > 1 and sd > 0:
        h = sps.t.ppf(0.975, n - 1) * sd / math.sqrt(n)
        p = float(sps.ttest_rel(a, b).pvalue)
    else:
        h, p = 0.0, None
    return {"diff": d, "mean": m, "sd": sd, "ci95": [m - h, m + h], "better": better, "worse": worse, "ties": n - better - worse,
            "p_ttest": p, "n": n}


def provenance() -> dict:
    def sh(*cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()
        except OSError:
            return ""
    return {"commit": sh("git", "rev-parse", "HEAD"), "dirty": sh("git", "status", "--porcelain", "--", "scripts", "selector", "data", "configs"),
            "argv": sys.argv, "python": sys.version.split()[0], "torch": torch.__version__, "threads": torch.get_num_threads(),
            "hw_model": sh("sysctl", "-n", "hw.model"), "cpu": sh("sysctl", "-n", "machdep.cpu.brand_string"),
            "mem_bytes": sh("sysctl", "-n", "hw.memsize"), "os": platform.platform(), "device": "cpu"}


def ms(x: list[float]) -> str:
    return f"{C.mean(x):.4f} ± {statistics.stdev(x):.4f}"


def report(cx: Ctx, folds: list[int], V: dict, res: dict) -> None:
    pv = res["provenance"]
    L = ["# REPORT LAD-1 — ladder 兩格（十折 × 兩序）", "",
         f"機器：{pv['hw_model']}（{pv['cpu']}，{int(pv['mem_bytes']) // 2**30} GB）；`--device cpu`、threads {pv['threads']}、torch {pv['torch']}；"
         f"程式 commit {pv['commit']}；完成 {res['finished']}。四列出自同一次執行。", "",
         "| 列 | 定義 |", "|---|---|"] + [f"| {r} | {WHAT[r]} |" for r in ROWS] + [
         "", f"查核：KA（FINALB、LIN8 對 ext2/a）最大絕對差 {max(res['KA']['per_row_max_abs'].values()):.1e}；"
         f"KB（MV_RDG 對 moe3/f5 的 S-R0 × mv）最大絕對差 {res['KB']['max_abs']:.1e}。", ""]
    for row in ROWS:
        L += [f"## {row} 逐折", "", "| 序 | fold | " + " | ".join(h for _, h, _ in METRICS) + " |", "|---|---|---|---|---|"]
        for o in ORDERS:
            v = V[row][o]
            L += [f"| {o} | {f} | " + " | ".join(f"{v[k][i]:.4f}" for k, _, _ in METRICS) + " |" for i, f in enumerate(folds)]
            L.append(f"| {o} | mean ± sd | " + " | ".join(ms(v[k]) for k, _, _ in METRICS) + " |")
        L.append("")
    L += ["## 逐折配對", "", "差 = 前者 − 後者；「較好」在 Forgetting 指數值較小。p = paired t-test（雙尾）；差全為 0 時 p 不適用。", "",
          "| 比較 | 序 | 指標 | 平均差 | 95% CI | 前者較好／10 | 較差 | 平手 | p |", "|---|---|---|---|---|---|---|---|---|"]
    for a, b in PAIRS:
        for o in ORDERS:
            for k, h, _ in METRICS:
                s = res["paired"][f"{a}-{b}"][o][k]
                p = "不適用" if s["p_ttest"] is None else f"{s['p_ttest']:.4g}"
                L.append(f"| {a} − {b} | {o} | {h} | {s['mean']:+.4f} | [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}] | "
                         f"{s['better']} | {s['worse']} | {s['ties']} | {p} |")
    (cx.out / "REPORT_lad1.md").write_text("\n".join(L) + "\n")


def assemble(cx: Ctx, folds: list[int]) -> None:
    with (cx.out / "LADDER_perfold.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["row", "order", "fold", "t", "CIL_ACC", "WP", "MaskedACC_table1", "MaskedACC_oracle", "Forgetting", "BWT"])
        for row in ROWS:
            for o in ORDERS:
                for f in folds:
                    r = load(cx, row, o, f)
                    for t in range(4):
                        w.writerow([row, o, f, t + 1] + [f"{r[k][t]:.10f}" for k in ("acc_t", "wp_t", "mk1_t", "wp_t")]
                                   + ([f"{r['forgetting']:.10f}", f"{r['bwt']:.10f}"] if t == 3 else ["", ""]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--folds", default="1-10")
    a = ap.parse_args()
    if a.device != "cpu":
        raise SystemExit("LAD-1 規定 --device cpu（與 FINAL-B、LIN8 的既有批次相同）")
    folds = P.parse_folds(a.folds)
    pv = provenance()
    sys.argv = sys.argv[:1]
    cx = Ctx()
    done, fail = cx.out / "cells.done", cx.out / "FAILED_cells.txt"
    if done.exists():
        log("cells: done, skip")
        return 0
    cx.root.mkdir(parents=True, exist_ok=True)
    pv["threads"] = torch.get_num_threads()
    log(f"LAD-1 cells folds={folds} threads={pv['threads']} commit={pv['commit'][:7]}")
    t0 = time.perf_counter()
    try:
        for f in folds:
            for o in ORDERS:
                t1 = time.perf_counter()
                do_fold(cx, f, o)
                log(f"fold {f} {o}: {len(ROWS)} 列 {time.perf_counter() - t1:.1f}s")
            cx.r3.drop(); cx.r4.drop(); cx.r3.v2._vec.clear()
        ka, kb = check_ka(cx, folds), check_kb(cx, folds)
        log(f"KA {ka['per_row_max_abs']}（{'通過' if ka['pass'] else '不過'}）；KB max_abs = {kb['max_abs']:.2e}（{'通過' if kb['pass'] else '不過'}）")
        if not (ka["pass"] and kb["pass"]):
            raise M.CheckFailed(f"查核不過：KA = {ka}；KB = {kb}")
        V = values(cx, folds)
        pr = {f"{x}-{y}": {o: {k: paired(V[x][o][k], V[y][o][k], hi) for k, _, hi in METRICS} for o in ORDERS} for x, y in PAIRS}
        same = {row: max(abs(p - q) for p, q in zip(V[row][ORDERS[0]]["acc4"], V[row][ORDERS[1]]["acc4"])) for row in ROWS}
        res = {"stage": "cells", "folds": folds, "wall_s": round(time.perf_counter() - t0, 1), "finished": M.now(), "provenance": pv,
               "rows": WHAT, "KA": ka, "KB": kb, "values": V, "paired": pr, "order_acc4_max_abs_diff": same}
        assemble(cx, folds)
        report(cx, folds, V, res)
        (cx.out / "paired.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=M._default))
        done.write_text(M.now() + "\n")
        log(f"cells: 完成（{res['wall_s']:.0f}s）")
        return 0
    except Exception:
        fail.write_text(f"[{M.now()}] 階段 cells 失敗（folds={folds}）\n{traceback.format_exc()}")
        traceback.print_exc()
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
