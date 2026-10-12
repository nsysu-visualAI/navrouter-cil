#!/usr/bin/env python3
"""EXT-3 第二段（PREREG-24）：LIN8 與格 1（MV_RDG）的 γ 用 PREREG-19 細則 5 的逐階段 validation 規則重選，再用選到的 γ 重算。
只讀既有快取，不訓練。FINAL-B 的 γ 引用 ext2/a.json，不重選；FINAL-B 在同一次執行重算。

  選擇   格 1：moe3_common.val_objective(kind = mv)／select；LIN8：同一個目標，權重用 moe1_common.lin8_stage（nc8_report.B8.W(lin8=True)）
  描述   LIN8 的 validation 8 類 CIL ACC（同 grid；不用於選擇）
  重算   LIN8(γ)、MV_RDG(γ) 的現用值與選到的值、FINALB：t = 1…4、兩序、十折；另存 t = 4 逐張的 CIL 與告訴任務對錯
  查核   KD 現用值的三列對 lad1/cells（≤ 1e-9、逐張 CIL 對錯相同）；KE 逐張告訴任務對錯的任務等權平均 = wp_t[3]（不過即停）
  配對   格 1 − LIN8、格 1 − FINAL-B：CIL ACC 與 WP 的 ext1_d.sign_pair ＋ ext1_d.bootstrap

    NAVCIL_MACHINE=mac python scripts/ext3_gamma.py
輸出：outputs/navcil/<machine>/ext3/g/<row>/<order>_fold<f>.json（＋ .done）、ext3/gamma.json、EXT3_gamma_perfold.csv、ext3/SECTION_gamma.md
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import ext1_c as CC                                                       # noqa: E402
import ext1_d as DD                                                       # noqa: E402
import ext2_a as A2                                                       # noqa: E402
from ext1_c import C, D64, E, M, N5, T, log                               # noqa: E402

ORDERS, FOLDS = CC.ORDERS, DD.FOLDS
GRID = T.RDG_G
CUR = {"LIN8": 0.01, "MV_RDG": 1e-3, "FINALB": 1e-3}                      # 現用值
STORAGE = {"LIN8": (1_052_676, 4_104), "MV_RDG": (1_052_676, 6_156), "FINALB": (2_105_352, 10_252)}
NAME = {"LIN8": "LIN8", "MV_RDG": "格 1（MV_RDG）", "FINALB": "FINAL-B"}
TOL, NEAR = 1e-9, 0.01


def tag(base: str, g: float) -> str:
    return base if base == "FINALB" else f"{base}_g{g:g}"


class Ctx(CC.Ctx):
    """ext1_c.Ctx 的唯讀來源不變；輸出寫到 ext3/g/。"""

    def __init__(self):
        super().__init__()
        self.out = self.base / "ext3"
        self.root = self.out / "g"


# ── 選擇（只用 validation）──────────────────────────────────────────────────
def lin8_val(cx: Ctx, f: int) -> tuple[dict, dict]:
    """一折：LIN8 每個 γ 在兩序 × t = 1…4 的 validation 告訴任務 WP_t 平均（選擇用）與 8 類 CIL ACC_t 平均（描述用）。"""
    D = T.data(cx.r3, "val", f)
    task, label, mv = D["task"], D["label"], D["mv"]
    wp, cil = {T.ckey(c): 0.0 for c in T.cands("RDG")}, {T.ckey(c): 0.0 for c in T.cands("RDG")}
    for o in ORDERS:
        pos = cx.r3.pos(o)
        for t in range(1, 5):
            for c in T.cands("RDG"):
                lg = M.lin8_stage(cx.st, f, o, t, mv, c[1])
                tell = E.pred_of(M.pick(M.d_cols(lg), task), task) == label
                pred = lg.nan_to_num(nan=float("-inf")).argmax(-1) == label
                wp[T.ckey(c)] += C.mean(tell[task == q].float().mean().item() for q in pos[:t]) / 8
                cil[T.ckey(c)] += C.mean(pred[task == q].float().mean().item() for q in pos[:t]) / 8
    return wp, cil


def choose(cx: Ctx) -> dict:
    per = {k: {T.ckey(c): [] for c in T.cands("RDG")} for k in ("MV_RDG", "LIN8", "LIN8_cil8")}
    for f in FOLDS:
        for k, v in T.val_objective(cx.r3, f, "mv", "RDG").items():
            per["MV_RDG"][k].append(v)
        wp, cil = lin8_val(cx, f)
        for k in wp:
            per["LIN8"][k].append(wp[k]); per["LIN8_cil8"][k].append(cil[k])
        cx.r3.drop()
    obj = {n: {k: C.mean(v) for k, v in d.items()} for n, d in per.items()}
    ext2 = json.loads((cx.base / "ext2" / "a.json").read_text())["gamma"]
    return {"grid": list(GRID), "obj": obj, "obj_per_fold": per,
            "star": {"MV_RDG": T.select("RDG", obj["MV_RDG"]), "LIN8": T.select("RDG", obj["LIN8"]),
                     "FINALB": {**ext2["star"], "source": "ext2/a.json"}},
            "obj_FINALB_ext2": ext2["obj"], "current": CUR,
            "same_obj_max_abs": max(abs(obj["MV_RDG"][k] - obj["LIN8"][k]) for k in obj["MV_RDG"]),
            "desc_argmax_LIN8_cil8": max(obj["LIN8_cil8"], key=obj["LIN8_cil8"].get)}


# ── 重算（test）─────────────────────────────────────────────────────────────
def lin8_eval(cx: Ctx, f: int, o: str, g: float, D: dict) -> dict:
    task, label, mv = D["task"], D["label"], D["mv"]
    last = {}

    def fn(seen):
        t = len(seen)
        m = torch.isin(task, torch.tensor(seen))
        lg = M.lin8_stage(cx.st, f, o, t, mv, g)
        pred = lg.nan_to_num(nan=float("-inf")).argmax(-1)
        mk = E.pred_of(M.pick(M.d_cols(lg), task), task)
        if t == 4:
            last.update({"ok4": (pred == label).long().tolist(), "tell4": (mk == label).long().tolist()})
        return pred[m], mk[m], {"labels": label[m], "task": task[m]}

    r = N5.cil_full(fn, o, cx.tasks)
    return {"acc_t": r["acc_t"], "mk1_t": r["masked_t"], "wp_t": r["masked_t"], "forgetting": r["forgetting"], "bwt": r["bwt"],
            "R": r["R"], "Rm_t1": r["Rm"], "Rm_orc": r["Rm"], **last}


def first_eval(cx: Ctx, f: int, o: str, kind: str, g: float, D: dict, ar) -> dict:
    task, label = D["task"], D["label"]
    first = A2.first_of(cx, f, o, D, kind, g)
    res = CC.cl_eval(first, ar, task, label, o, cx.tasks)
    res["tell4"] = ((2 * task + (~first(4, task, task)).long()) == label).long().tolist()
    return res


def recompute(cx: Ctx, gs: dict) -> None:
    for f in FOLDS:
        D = T.data(cx.r3, "test", f)
        for o in ORDERS:
            ar = T.ar_stages(cx.r3, f, o, D)
            for g in gs["LIN8"]:
                cx.save(tag("LIN8", g), o, f, {**lin8_eval(cx, f, o, g, D), "gamma": g})
            for g in gs["MV_RDG"]:
                cx.save(tag("MV_RDG", g), o, f, {**first_eval(cx, f, o, "mv", g, D, ar), "gamma": g})
            cx.save("FINALB", o, f, {**first_eval(cx, f, o, A2.KIND_B, CUR["FINALB"], D, ar), "gamma": CUR["FINALB"], "task": D["task"].tolist()})
        cx.r3.drop(); cx.r3.v2._vec.clear()
        log(f"fold {f}: 重算完成")


def row(cx: Ctx, name: str, o: str, f: int) -> dict:
    return json.loads(cx.path(name, o, f).read_text())


def checks(cx: Ctx, names: list[str]) -> dict:
    kd, ke = {}, {}
    for base, g in CUR.items():
        m, same = 0.0, True
        for o in ORDERS:
            for f in FOLDS:
                a, b = row(cx, tag(base, g), o, f), json.loads((cx.base / "lad1" / "cells" / base / f"{o}_fold{f}.json").read_text())
                for k in A2.K9_KEYS:
                    m = max([m] + [abs(x - y) for x, y in zip(A2.flat(a[k]), A2.flat(b[k]))])
                same = same and a["ok4"] == b["ok4"]
        kd[tag(base, g)] = {"max_abs": m, "ok4_same": same}
    for n in names:
        m = 0.0
        for o in ORDERS:
            for f in FOLDS:
                r = row(cx, n, o, f)
                task = torch.tensor(row(cx, "FINALB", o, f)["task"])
                m = max(m, abs(C.mean(C.task_mean(torch.tensor(r["tell4"]).bool(), task)) - r["wp_t"][3]))
        ke[n] = m
    return {"KD": {"rows": kd, "tol": TOL, "pass": all(v["max_abs"] <= TOL and v["ok4_same"] for v in kd.values())},
            "KE": {"rows": ke, "tol": TOL, "pass": all(v <= TOL for v in ke.values())}}


# ── 配對與判讀 ──────────────────────────────────────────────────────────────
def series(cx, name, o, key):
    return [DD.METRICS[key](row(cx, name, o, f)) for f in FOLDS]


def pair(cx: Ctx, tasks_pf, a: str, b: str, o: str) -> dict:
    out = {}
    for mk, key, per in (("ACC", "ACC", "ok4"), ("WP", "MaskedACC_oracle", "tell4")):
        c = DD.sign_pair(series(cx, a, o, key), series(cx, b, o, key))
        c["boot_ci95"] = DD.bootstrap(None, tasks_pf, *[[torch.tensor(row(cx, n, o, f)[per], dtype=D64) for f in FOLDS] for n in (a, b)])
        out[mk] = c
    return out


def analyse(cx: Ctx, sel: dict, names: list[str]) -> dict:
    tasks_pf = [torch.tensor(row(cx, "FINALB", ORDERS[0], f)["task"]) for f in FOLDS]
    star = {b: sel["star"][b]["gamma"] for b in CUR}
    keys = ("ACC", "MaskedACC_oracle", "MaskedACC_table1", "Forgetting", "BWT", "Abar")
    summary = {o: {n: {k: DD.ms(series(cx, n, o, k)) for k in keys} for n in names} for o in ORDERS}
    pairs = {}
    for label, gs in (("selected", star), ("current", CUR)):
        mv, l8 = tag("MV_RDG", gs["MV_RDG"]), tag("LIN8", gs["LIN8"])
        pairs[label] = {f"{mv} − {l8}": {o: pair(cx, tasks_pf, mv, l8, o) for o in ORDERS},
                        f"{mv} − FINALB": {o: pair(cx, tasks_pf, mv, "FINALB", o) for o in ORDERS}}
    changed = {b: star[b] != CUR[b] for b in CUR}
    c2 = pairs["selected"][f"{tag('MV_RDG', star['MV_RDG'])} − FINALB"][ORDERS[0]]["ACC"]
    step, ci = -c2["mean_diff"], [-c2["boot_ci95"][1], -c2["boot_ci95"][0]]
    verdict = {"changed": changed, "case": "b" if any(changed.values()) else "a",
               "second_step": {"mean": step, "boot_ci95": ci, "wins_FINALB": c2["losses"], "losses_FINALB": c2["wins"], "ties": c2["ties"],
                               "p_sign": c2["p_sign"], "lt_1pp": step < NEAR, "ci_has_0": ci[0] <= 0 <= ci[1]}}
    verdict["second_step"]["flag_c"] = verdict["second_step"]["lt_1pp"] or verdict["second_step"]["ci_has_0"]
    return {"star": star, "summary": summary, "pairs": pairs, "verdict": verdict, "n_boot": DD.N_BOOT, "boot_seed": DD.BOOT_SEED}


# ── 輸出 ────────────────────────────────────────────────────────────────────
def write_csv(cx: Ctx, names: list[str]) -> None:
    def fm(x):
        return "" if x is None else f"{x:.10f}"

    with (cx.base / "EXT3_gamma_perfold.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["row", "order", "fold", "t", "CIL_ACC", "WP", "MaskedACC_table1", "MaskedACC_oracle", "Forgetting", "BWT", "storage_bytes"])
        for n in names:
            sp = STORAGE[n.split("_g")[0]]
            for o in ORDERS:
                for f in FOLDS:
                    r = row(cx, n, o, f)
                    for t in range(4):
                        w.writerow([n, o, f, t + 1, fm(r["acc_t"][t]), fm(r["wp_t"][t]), fm(r["mk1_t"][t]), fm(r["wp_t"][t]),
                                    fm(r["forgetting"]) if t == 3 else "", fm(r["bwt"]) if t == 3 else "", sp[0] + (t + 1) * sp[1]])


def pm(v) -> str:
    return f"{v[0]:.4f} ± {v[1]:.4f}"


def yn(x: bool) -> str:
    return "是" if x else "否"


def pair_tables(L: list, pairs: dict) -> None:
    for name, per in pairs.items():
        L += [f"**{name}**（CIL ACC 與 WP 兩序逐折相同時兩序的列相同）", "",
              "| 序 | 指標 | 前者 | 後者 | 平均差 | 贏／輸／平手 | p（符號檢定） | bootstrap 95% CI |", "|---|---|---|---|---|---|---|---|"]
        for o in ORDERS:
            for mk, h in (("ACC", "CIL ACC"), ("WP", "WP")):
                c = per[o][mk]
                p = "不適用" if c["p_sign"] is None else f"{c['p_sign']:.4f}"
                L.append(f"| {o} | {h} | {pm(c['a'])} | {pm(c['b'])} | {c['mean_diff']:+.4f} | {c['wins']}／{c['losses']}／{c['ties']} | {p} | "
                         f"[{c['boot_ci95'][0]:+.4f}, {c['boot_ci95'][1]:+.4f}] |")
        L.append("")


def write_section(cx: Ctx, res: dict) -> None:
    sel, an, ck, pv = res["selection"], res["analysis"], res["checks"], res["provenance"]
    star, V = an["star"], an["verdict"]
    ks = [T.ckey(c) for c in T.cands("RDG")]

    def line(name, obj, s=None):
        return f"| {name} | " + " | ".join(f"**{obj[k]:.4f}**" if s is not None and k == f"{s:g}" else f"{obj[k]:.4f}" for k in ks) + " |"

    def edge(g):
        return g in (min(GRID), max(GRID))

    L = ["## 6　第二段：γ 統一用同一條驗證規則選（PREREG-24）", "",
         "### 6-0　執行", "", "| 項目 | 值 |", "|---|---|",
         f"| 程式 commit | `{pv['commit']}`（scripts／selector／data／configs 無未 commit 變更：{yn(not pv['dirty'])}） |",
         "| 指令 | `NAVCIL_MACHINE=mac caffeinate -ims /Users/aaron/venvs/navcil/bin/python scripts/ext3_gamma.py` |",
         f"| 機器、環境 | 同第 0 節（{pv['hw_model']}，cpu、threads {pv['threads']}、torch {pv['torch']}）；完成 {res['finished']}，{res['wall_s']} 秒 |",
         f"| KD（現用值三列對 `lad1/cells/`） | 最大絕對差 {max(v['max_abs'] for v in ck['KD']['rows'].values()):.1e}；逐張 CIL 對錯相同："
         f"{yn(all(v['ok4_same'] for v in ck['KD']['rows'].values()))}；{'通過' if ck['KD']['pass'] else '不過'} |",
         f"| KE（逐張告訴任務對錯的任務等權平均對 WP） | 最大絕對差 {max(ck['KE']['rows'].values()):.1e}；{'通過' if ck['KE']['pass'] else '不過'} |", "",
         "### 6-1　LIN8 現用的 γ = 0.01 是不是同一條規則選的", "", "不是（D14）。", "",
         "| 項目 | LIN8 當初（NC-5） | 本規則（PREREG-19 細則 5；EXT-2 對 FINAL-B 用的） |", "|---|---|---|",
         "| 出處 | `scripts/nc5_report.py:301-309`；REPORT_stage7.md:16、:114 | `moe3_common.val_objective`／`select`；REPORT_ext2 0-4 |",
         "| grid | {0.01, 0.1, 1, 10} | {1e-5, 1e-4, 1e-3, 1e-2, 1e-1} |",
         "| 目標 | validation 8 類 CIL ACC，只看 t = 4（兩序 × 十折） | validation 告訴任務 WP，兩序 × t = 1…4 平均（十折） |",
         "| 同分 | 取較小的 γ | 取較大的 γ |",
         "| 選到的值 | 0.01（該 grid 的下界） | 見 6-2 |", "",
         "### 6-2　validation 目標值與選定的 γ", "",
         "目標 = 兩序 × t = 1…4 的 validation 告訴任務 WP 平均，十折平均。粗體 = 選定。", "",
         "| 列 | " + " | ".join(f"γ = {k}" for k in ks) + " |", "|---|" + "---|" * len(ks),
         line("格 1（MV_RDG）", sel["obj"]["MV_RDG"], star["MV_RDG"]), line("LIN8", sel["obj"]["LIN8"], star["LIN8"]),
         line("FINAL-B（引用 `ext2/a.json`）", sel["obj_FINALB_ext2"], star["FINALB"]),
         line("描述：LIN8 的 validation 8 類 CIL ACC（不用於選擇）", sel["obj"]["LIN8_cil8"]), "",
         "| 列 | 現用值 | 選定值 | 是否改變 | 選定值在 grid 邊界 |", "|---|---|---|---|---|"]
    L += [f"| {NAME[b]} | {CUR[b]:g} | {star[b]:g} | {yn(V['changed'][b])} | {yn(edge(star[b]))} |" for b in ("LIN8", "MV_RDG", "FINALB")]
    L += ["", f"格 1 與 LIN8 兩組目標值的最大絕對差 {sel['same_obj_max_abs']:.1e}（兩者是同一個讀出，D15）。"
          f"描述列的最大值在 γ = {sel['desc_argmax_LIN8_cil8']}。", "",
          "### 6-3　t = 4（十折 mean ± sd；test）", ""]
    names = res["rows"]
    mark = {tag(b, star[b]) for b in CUR}
    for o in ORDERS:
        L += [f"序 {o}", "", "| row | γ | 選定 | CIL ACC | WP | Masked ACC（Table 1） | Forgetting | BWT | Ā |", "|---|---|---|---|---|---|---|---|---|"]
        for n in names:
            s = an["summary"][o][n]
            g = row(cx, n, o, 1)["gamma"]
            L.append(f"| {n} | {g:g} | {yn(n in mark)} | " + " | ".join(pm(s[k]) for k in ("ACC", "MaskedACC_oracle", "MaskedACC_table1", "Forgetting", "BWT", "Abar")) + " |")
        L.append("")
    L += ["### 6-4　選定 γ 的逐折（t = 4）", ""]
    for b in ("LIN8", "MV_RDG"):
        n = tag(b, star[b])
        L += [f"**{n}**", "", "| fold | CIL ACC | WP | Masked ACC（Table 1） | Forgetting（reverse） | Forgetting（paper） |", "|---|---|---|---|---|---|"]
        for f in FOLDS:
            a, p = row(cx, n, "reverse", f), row(cx, n, "paper", f)
            L.append(f"| {f} | {a['acc_t'][3]:.4f} | {a['wp_t'][3]:.4f} | {a['mk1_t'][3]:.4f} | {a['forgetting']:.4f} | {p['forgetting']:.4f} |")
        sr, sp = an["summary"]["reverse"][n], an["summary"]["paper"][n]
        L += [f"| mean ± sd | {pm(sr['ACC'])} | {pm(sr['MaskedACC_oracle'])} | {pm(sr['MaskedACC_table1'])} | {pm(sr['Forgetting'])} | {pm(sp['Forgetting'])} |", ""]
    L += ["### 6-5　逐折配對（t = 4；差 = 前者 − 後者）", "",
          f"贏 = 前者較大；p = exact binomial 雙尾符號檢定；bootstrap = slide 層級、(折, 任務) 分層、配對、{an['n_boot']} 次、seed {an['boot_seed']}"
          "（WP 的逐張值 = 告訴任務時是否判對）。", "", "主（各列用選定的 γ）：", ""]
    pair_tables(L, an["pairs"]["selected"])
    if V["case"] == "b":
        L += ["補充（各列用現用值）：", ""]
        pair_tables(L, an["pairs"]["current"])
    else:
        L += ["選定值與現用值相同，沒有第二組配對。", ""]
    s2 = V["second_step"]
    L += ["### 6-6　PREREG-24 的判讀", "", "| 項目 | 結果 |", "|---|---|",
          f"| a／b：γ 是否改變 | LIN8：{yn(V['changed']['LIN8'])}（{CUR['LIN8']:g} → {star['LIN8']:g}）；格 1：{yn(V['changed']['MV_RDG'])}"
          f"（{CUR['MV_RDG']:g} → {star['MV_RDG']:g}）；FINAL-B：{yn(V['changed']['FINALB'])} → 情況 **{V['case']}** |",
          f"| c：第二階（FINAL-B − 格 1，選定 γ）的 CIL ACC | {s2['mean']:+.4f}（{s2['mean'] * 100:+.2f} pp）；bootstrap 95% CI [{s2['boot_ci95'][0]:+.4f}, {s2['boot_ci95'][1]:+.4f}] |",
          f"| c：小於 +1 pp | {yn(s2['lt_1pp'])} |", f"| c：CI 含 0 | {yn(s2['ci_has_0'])} |",
          f"| c：是否標明 | **{yn(s2['flag_c'])}** |",
          f"| 並列（不屬於 c）：第二階的符號檢定 | FINAL-B 贏／輸／平手 {s2['wins_FINALB']}／{s2['losses_FINALB']}／{s2['ties']}；p = {s2['p_sign']:.4f}；"
          f"依 PREREG-23 的定義顯著：{yn(s2['p_sign'] < 0.05)} |", ""]
    (cx.out / "SECTION_gamma.md").write_text("\n".join(L))


def provenance() -> dict:
    def sh(*cmd):
        return subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()
    return {"commit": sh("git", "rev-parse", "HEAD"), "dirty": sh("git", "status", "--porcelain", "--", "scripts", "selector", "data", "configs"),
            "torch": torch.__version__, "hw_model": sh("sysctl", "-n", "hw.model"), "cpu": sh("sysctl", "-n", "machdep.cpu.brand_string")}


def main() -> int:
    pv = provenance()
    sys.argv = sys.argv[:1]
    cx = Ctx()
    cx.root.mkdir(parents=True, exist_ok=True)
    pv["threads"] = torch.get_num_threads()
    t0 = time.perf_counter()
    try:
        sel = choose(cx)
        log(f"選擇：格 1 γ = {sel['star']['MV_RDG']['gamma']:g}、LIN8 γ = {sel['star']['LIN8']['gamma']:g}（FINAL-B 引用 {sel['star']['FINALB']['gamma']:g}）")
        gs = {b: sorted({CUR[b], sel["star"][b]["gamma"]}) for b in ("LIN8", "MV_RDG")}
        names = [tag("LIN8", g) for g in gs["LIN8"]] + [tag("MV_RDG", g) for g in gs["MV_RDG"]] + ["FINALB"]
        recompute(cx, gs)
        ck = checks(cx, names)
        log(f"KD {'通過' if ck['KD']['pass'] else '不過'}；KE {'通過' if ck['KE']['pass'] else '不過'}")
        if not (ck["KD"]["pass"] and ck["KE"]["pass"]):
            raise M.CheckFailed(f"查核不過：{ck}")
        res = {"stage": "gamma", "wall_s": round(time.perf_counter() - t0, 1), "finished": M.now(), "provenance": pv, "rows": names,
               "selection": sel, "checks": ck, "analysis": analyse(cx, sel, names)}
        (cx.out / "gamma.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=M._default))
        write_csv(cx, names)
        write_section(cx, res)
        log(f"gamma：完成（{res['wall_s']:.0f}s）")
        return 0
    except Exception:
        (cx.out / "FAILED_gamma.txt").write_text(f"[{M.now()}] 階段 gamma 失敗\n{traceback.format_exc()}")
        traceback.print_exc()
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
