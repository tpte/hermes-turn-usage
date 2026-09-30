"""turn-usage 的純函式層：增量計算與格式化。不碰 I/O，方便測試。"""
from __future__ import annotations

from collections import Counter
import re
from typing import Optional

FIELDS = ("calls", "in", "out", "cache", "cost", "reasoning")

#: 摘要區塊的辨識錨點。`format_block` 的標題行固定以它開頭。
BLOCK_MARKER = "<details><summary>📊用量"
#: 舊版標題（一行式之前），只用來剥掉歷史訊息裡殘留的舊區塊。
LEGACY_MARKER = "📊 **本輪用量**"

#: 用量摘要的辨識：`<details><summary>` 裡含 📊 與「用量」就算——**不靠字面比對**。
#: 為什麼：歷史上有過多種標題寫法（`📊用量`、`📊 用量 ｜ ⏱️ 活躍`、`📊 **本輪用量**`…），
#: 字面比對只要差一個空格就找不到 → 舊的那一份留著、新的又接上去，畫面就出現兩份。
#: 實例：同一則回覆裡「📊 用量 …」接著「📊用量 …」，一新一舊並存
#: （舊那份來自某個先前載入的版本）。
_BLOCK_RE = re.compile(r"<details\s*>\s*<summary\s*>[^<]*📊[^<]*用量")
#: 更早的兩行式（沒有 `<details>` 包住）：`📊 **本輪用量** 💰…`
_LEGACY_RE = re.compile(r"📊\s*\*\*本輪用量\*\*")


def _cut_leading_rule(head: str) -> str:
    """摘要前面是我們自己接的 `---`；連它一起收，免得留下孤零零的水平線。"""
    cut = head.rfind("\n---")
    if cut != -1 and not head[cut + 1:].strip("- \t\n"):
        return head[:cut]
    return head


def strip_previous_block(text: str) -> str:
    """剝掉先前已附加的摘要區塊（讓 hook 冪等）。

    為什麼需要：`transform_llm_output` 在同一個 turn 內可能被叫不只一次
    （Hermes 的 `turn_finalizer` 有兩個 persist 呼叫點），而每次進來的
    ``response_text`` 已經含著上一次附加的摘要 —— 直接再接一段就會**疊成好幾份**。
    實測症狀：同一則回覆出現兩份、三份摘要。

    這裡把**所有**已存在的摘要區塊整段收掉（連同前面的 ``---``），而且用正則認標題，
    所以不管被呼叫幾次、也不管那一份是哪個舊版本附加的，最後都只會剩一份。
    """
    while True:
        match = _BLOCK_RE.search(text)
        legacy = None if match else _LEGACY_RE.search(text)
        found = match or legacy
        if found is None:
            return text.rstrip()
        head = _cut_leading_rule(text[:found.start()])
        if legacy is not None:
            # 舊式那條沒有 <details> 包住，內容一路到結尾。
            text = head
            continue
        end = text.find("</details>", found.end())
        tail = text[end + len("</details>"):] if end != -1 else ""
        text = head + (("\n\n" + tail.strip()) if tail.strip() else "")


def cache_pct(input_tokens: int, cache_read_tokens: int) -> Optional[int]:
    """快取命中率（0–100）。沒有快取讀取時回 None，不假造 0。

    分母用 prompt 總量（= 全新輸入 + 快取讀取），與 Hermes 自己的
    `_get_usage` 算法一致（`tui_gateway/server.py`）。
    """
    total = int(input_tokens) + int(cache_read_tokens)
    if total <= 0 or cache_read_tokens <= 0:
        return None
    return max(0, min(100, round(cache_read_tokens / total * 100)))


def diff_totals(now: dict, prev: Optional[dict]) -> dict:
    """本輪增量 = 現在累計 − 上一輪快照。沒有快照就整份算本輪。"""
    if not prev:
        return {k: now.get(k, 0) for k in FIELDS}
    return {k: now.get(k, 0) - prev.get(k, 0) for k in FIELDS}


def _k(n: float) -> str:
    """3200 → 3.2K；920 → 920；1200000 → 1.2M"""
    n = float(n)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return f"{int(n)}"


def route_text(providers: list, upstreams: list) -> str:
    """組出「路由」欄：Hermes 的設定名 → router 的實際落點。

    providers 來自 DB 的 billing_provider；upstreams 是 agent.log 裡
    每筆呼叫的實際供應商。落點超過一家時只列前兩名並標佔比。
    """
    alias = "、".join(providers) if providers else "?"
    if not upstreams:
        return alias
    counts = Counter(upstreams)
    total = len(upstreams)
    top = counts.most_common(2)
    if len(top) == 1 or top[0][1] / total >= 0.9:
        return f"{alias} → {top[0][0]}"
    return f"{alias} → " + "／".join(f"{name} {count * 100 // total}%" for name, count in top)


def _bar(pct: Optional[int], width: int = 10) -> str:
    """把 0–100 畫成方塊條（純文字，任何渲染器都吃得下）。"""
    if pct is None:
        return "░" * width
    filled = max(0, min(width, round(pct / 100 * width)))
    return "▓" * filled + "░" * (width - filled)


def _pct(v: Optional[int]) -> str:
    return f"{v}%" if v is not None else "—"


#: 訊息內容的分類。key 是顯示名，值是 tool_name（`""` = 使用者／助理的對話內容）。
#: 「改檔」與「讀檔」刻意分開：diff 進 context 的量是「我改了幾個檔」的證據，
#: 和「讀了多少檔案」是兩件事，混在一起會誤導。
COMPOSITION_GROUPS = (
    ("💻 命令行", ("terminal", "execute_code", "process_manage")),
    ("🔧 改檔", ("patch", "write_file")),
    ("📄 讀檔", ("read_file", "search_files")),
    ("👁️ 圖片識別", ("vision_analyze",)),
    ("🌐 網路", ("web_search", "web_extract", "browser_exec")),
    ("📚 skill", ("skill_view", "skill_manage", "skills_list")),
    ("🧠 記憶", ("memory",)),
)


def merge_sources(sources, groups=None) -> list:
    """把 [(tool_name, 字元數)] 彙總成 [(顯示名, 佔比)]，大的在前；不足 1% 併成「其他」。

    目的：回答「這次的 prompt 大部分花在哪」——命令行／改檔／讀檔各佔多少。
    分母是訊息內容的總字元數（不含系統提示與工具定義），所以是**相對**佔比。
    ``groups`` 可傳自訂分類表（`((顯示名, (工具, …)), …)`）覆寫內建分類。
    """
    total = sum(chars for _name, chars in sources) or 0
    if not total:
        return []
    table = groups or COMPOSITION_GROUPS
    buckets: dict[str, int] = {}
    for name, chars in sources:
        label = None
        for group_label, members in table:
            if name in members:
                label = group_label
                break
        if label is None:
            label = "💬 對話" if not name else ("🧰 其他工具")
        buckets[label] = buckets.get(label, 0) + chars
    rows = sorted(buckets.items(), key=lambda kv: -kv[1])
    out, rest = [], 0
    for label, chars in rows:
        pct = round(chars / total * 100)
        if pct < 1:
            rest += chars
            continue
        out.append((label, pct))
    if rest:
        out.append(("▫️ 其他", round(rest / total * 100)))
    return out


def _tool_label(tool: str, table) -> str:
    """工具名 → 大類顯示名；不在分類表裡的一律歸「🧰 其他工具」。"""
    for group_label, members in table:
        if tool in members:
            return group_label
    return "🧰 其他工具"


def tool_time_rows(by_tool, cum_by_tool=None, groups=None, active_s=0.0,
                   min_pct: int = 10) -> list:
    """把 {工具名: [秒數, 次數]} 併成大類，回傳要渲染的時間明細列。

    回傳 [(層級, 顯示名, 本輪秒, 累積秒, 佔比, 次數)]，層級 0=大類、1=細項。
    佔比的分母是活躍時間（模型 + 工具），所以跟上一列可以直接相加。

    為什麼要分兩層：大類回答「哪一類工作吃掉時間」（圖片識別？寫 code？），
    細項回答「具體哪一支工具」。但只有佔比過 ``min_pct``、且類裡不只一支
    工具的大類才展開細項——否則一整排 0.1s 的工具會把明細淹掉。
    """
    table = groups or COMPOSITION_GROUPS
    buckets: dict[str, list] = {}   # 大類 → [秒, 次, {工具: [秒, 次]}]
    for name, pair in (by_tool or {}).items():
        try:
            secs, cnt = float(pair[0]), int(pair[1])
        except (TypeError, ValueError, IndexError):
            continue
        slot = buckets.setdefault(_tool_label(name, table), [0.0, 0, {}])
        slot[0] += secs
        slot[1] += cnt
        sub = slot[2].setdefault(name, [0.0, 0])
        sub[0] += secs
        sub[1] += cnt

    cum = cum_by_tool or {}

    def _cum(name: str) -> float:
        try:
            return float((cum.get(name) or [0, 0])[0])
        except (TypeError, ValueError, IndexError):
            return 0.0

    rows = []
    for label, (secs, cnt, members) in sorted(buckets.items(),
                                              key=lambda kv: -kv[1][0]):
        pct = round(secs / active_s * 100) if active_s else 0
        rows.append((0, label, secs, sum(_cum(n) for n in members), pct, cnt))
        if pct < min_pct or len(members) < 2:
            continue
        for name, (sub_s, sub_c) in sorted(members.items(),
                                           key=lambda kv: -kv[1][0]):
            sub_pct = round(sub_s / active_s * 100) if active_s else 0
            if sub_pct < 1:
                continue
            rows.append((1, name, sub_s, _cum(name), sub_pct, sub_c))
    return rows


def _dur(seconds: float) -> str:
    """132.4 → 2m12s；45.2 → 45s；7.3 → 7.3s；3725 → 1h02m。"""
    if seconds < 10:
        return f"{seconds:.1f}s" if seconds > 0 else "0s"
    s = int(round(seconds))
    if s >= 3600:
        return f"{s // 3600}h{(s % 3600) // 60:02d}m"
    if s >= 60:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s}s"


def _cost_table(delta: dict, p_in: float, p_cache: float, p_out: float,
                enabled: bool = True) -> list:
    """花費組成：把帳單拆成「輸入／快取命中／輸出」三塊並給出佔比。

    為什麼這樣算：帳單只給一個總額，但三塊的單價差一個量級，看不出錢花在哪。
    做法是用「單價 × 各自 token 量」求三塊的**相對比例**，再把比例乘回帳單的
    實際總額——總和必定等於帳單，而比例反映真實結構。
    單價是設定項（供應商折扣會變動，改設定即時生效，不必改程式）。
    """
    if not enabled or not (p_in or p_cache or p_out):
        return []
    inp, cache, out = (delta.get("in") or 0), (delta.get("cache") or 0), (delta.get("out") or 0)
    parts = [("📥 輸入（未快取）", inp, inp * p_in),
             ("💾 快取命中", cache, cache * p_cache),
             ("📤 輸出", out, out * p_out)]
    weight = sum(w for _n, _t, w in parts)
    if not weight:
        return []
    actual = delta.get("cost") or 0.0
    table = [
        "",
        f"**花費組成**（單價 in ${p_in}／cache ${p_cache}／out ${p_out} per M；"
        f"依比例對齊帳單 ${actual:.4f}）",
        "",
        "| 項目 | token | 花費 | 佔比 |",
        "|---|---|---|---|",
    ]
    for name, tokens, w in parts:
        table.append(f"| {name} | {_k(tokens)} | ${actual * w / weight:.4f} | "
                     f"{round(w / weight * 100)}% |")
    table.append("")
    return table


def format_block(
    delta: dict,
    totals: dict,
    *,
    model: str,
    route: str,
    prompt_now: Optional[int] = None,
    context_max: Optional[int] = None,
    context_hint_pct: int = 70,
    show_total: bool = True,
    show_time: bool = True,
    show_composition: bool = True,
    show_bars: bool = True,
    show_cost_breakdown: bool = True,
    price_in: float = 0.0,
    price_cache: float = 0.0,
    price_out: float = 0.0,
    bar_width: int = 10,
    cost_decimals: int = 3,
    groups=None,
    sources=None,
    sources_delta=None,
    timing=None,
    time_detail_min_pct: int = 10,
) -> str:
    """組出附在回覆尾端的 markdown 摘要：一行重點 ＋ 可折疊的明細與圖表。

    外行只留五個數字（花費／呼叫／進／出／context），其餘收進 <details>：
    本輪與累計對照表，加上快取命中、推理佔比、context 使用率三條方塊條。
    資訊一項都沒少，只是分層——要細節再點開。
    """
    prompt_total = delta["in"] + delta["cache"]
    hit = cache_pct(delta["in"], delta["cache"])
    reason_pct = round(delta["reasoning"] / delta["out"] * 100) if delta["out"] else None
    t_in = totals.get("in", 0) + totals.get("cache", 0)
    t_hit = cache_pct(totals.get("in", 0), totals.get("cache", 0))
    t_reason = round(totals["reasoning"] / totals["out"] * 100) if totals.get("out") else None
    ctx_pct = round(prompt_now / context_max * 100) if (prompt_now and context_max) else None
    # context 用滿時在外行那行就標出來：長 context 每輪都重送，又貴又慢，
    # 這是「該換新對話」最直接的訊號（門檻 context_hint_pct 可調，0 = 關）。
    ctx_hint = bool(ctx_pct and context_hint_pct and ctx_pct >= context_hint_pct)

    cost = f"${delta['cost']:.{max(0, min(6, int(cost_decimals)))}f}"
    think_s = float(timing.get("think_s") or 0) if timing else 0.0
    tool_s = float(timing.get("tool_s") or 0) if timing else 0.0
    active = think_s + tool_s
    cum_active = float(timing.get("cum_active_s") or 0) if timing else 0.0
    head = ("<details><summary>📊用量 " + cost
            + (f" ⏱️活躍 {_dur(active)}" if active else "")
            + (f" ⚠️context {_pct(ctx_pct)} 建議換新對話" if ctx_hint else "")
            + "</summary>")

    time_row, time_table = "", []
    if show_time and timing and (think_s or tool_s):
        n_calls = int(timing.get("calls") or 0)
        n_tools = int(timing.get("tools") or 0)
        # 活躍時間＝模型呼叫＋工具執行。刻意不含使用者閱讀與輸入的時間——
        # 那段的長度取決於人，放進來會讓「哪裡慢」失去意義。
        active = think_s + tool_s
        tp = round(think_s / active * 100) if active else 0
        op = round(tool_s / active * 100) if active else 0
        cum_think = float(timing.get("cum_think_s") or 0)
        cum_tool = float(timing.get("cum_tool_s") or 0)
        time_table = [
            "",
            "**時間分配**（不含你閱讀與輸入的時間）",
            "",
            "| 來源 | 本輪 | 累積 | 本輪佔比 | 次數 |",
            "|---|---|---|---|---|",
            f"| 🧠 模型思考／生成 | {_dur(think_s)} | "
            f"{_dur(cum_think) if cum_think else '—'} | {tp}% | {n_calls} |",
            f"| ⚙️ 工具執行 | {_dur(tool_s)} | "
            f"{_dur(cum_tool) if cum_tool else '—'} | {op}% | {n_tools} |",
        ]
        # 工具那一列再往下拆：大類（⤷）回答「哪一類工作」，細項（└）回答
        # 「具體哪一支工具」。原本只有一個總和，看不出圖片識別或寫 code 各花多少。
        for depth, label, secs, cum_s, pct, cnt in tool_time_rows(
                timing.get("by_tool"), timing.get("cum_by_tool"),
                groups, active, min_pct=time_detail_min_pct):
            mark = "⤷ " if depth == 0 else "　└ "
            # 有花時間但不到 1% 就寫 <1%，寫「0%」看起來像沒跑到。
            pct_txt = f"{pct}%" if pct else ("<1%" if secs > 0 else "0%")
            time_table.append(
                f"| {mark}{label} | {_dur(secs)} | "
                f"{_dur(cum_s) if cum_s else '—'} | {pct_txt} | {cnt} |")
        # 等待型工具（clarify 卡在 UI 等你回答）不算工時：單獨一列並標「不計入」，
        # 否則「等你」會被讀成「我很慢」，而它根本不在活躍時間的分母裡。
        wait_s = float(timing.get("wait_s") or 0)
        n_waits = int(timing.get("waits") or 0)
        if wait_s or n_waits:
            cum_wait = float(timing.get("cum_wait_s") or 0)
            time_table.append(
                f"| ⏸️ 等你回應 | {_dur(wait_s)} | "
                f"{_dur(cum_wait) if cum_wait else '—'} | 不計入 | {n_waits} |")
        time_table.append("")

    if show_total:
        table = [
            "|  | 本輪 | 累計 |",
            "|---|---|---|",
            f"| ⚡ 呼叫 | {delta['calls']} | {totals.get('calls', 0)} |",
            f"| 📥 進 | {_k(prompt_total)} | {_k(t_in)} |",
            f"| 💾 快取命中 | {_pct(hit)} | {_pct(t_hit)} |",
            f"| 📤 出 | {_k(delta['out'])} | {_k(totals.get('out', 0))} |",
            f"| 🤔 推理 | {_pct(reason_pct)} | {_pct(t_reason)} |",
            f"| ⏱️ 活躍 | {_dur(active)} | {_dur(cum_active) if cum_active else '—'} |",
            f"| 💰 花費 | ${delta['cost']:.4f} | ${totals.get('cost', 0) or 0:.4f} |",
        ]
    else:
        table = [
            "|  | 本輪 |",
            "|---|---|",
            f"| ⚡ 呼叫 | {delta['calls']} |",
            f"| 📥 進 | {_k(prompt_total)} |",
            f"| 💾 快取命中 | {_pct(hit)} |",
            f"| 📤 出 | {_k(delta['out'])} |",
            f"| 🤔 推理 | {_pct(reason_pct)} |",
            f"| 💰 花費 | ${delta['cost']:.4f} |",
        ]

    ctx_note = f"（{_k(prompt_now)}／{_k(context_max)}）" if (prompt_now and context_max) else ""
    now_rows = merge_sources(sources, groups) if sources else []
    delta_rows = merge_sources(sources_delta, groups) if sources_delta else []
    comp_table = []
    if show_composition and (now_rows or delta_rows):
        names = []
        for lbl, _p in delta_rows + now_rows:
            if lbl not in names:
                names.append(lbl)
        d_map, n_map = dict(delta_rows), dict(now_rows)
        comp_table = [
            "**prompt 組成**（不含系統提示與工具定義）",
            "",
            "| 來源 | 本輪 | 累積 |",
            "|---|---|---|",
        ] + [f"| {lbl} | {d_map.get(lbl, 0)}% | {n_map.get(lbl, 0)}% |" for lbl in names] + [""]
    detail = [
        "",
        f"`{model}`　路由 {route}",
        "",
        f"🧠 context `{_k(prompt_now) if prompt_now else '—'}`"
        + (f"／上限 {_k(context_max)}" if context_max else ""),
        "",
    ] + table + ([] if not show_bars else [
        "",
        f"快取命中　{_bar(hit, bar_width)} {_pct(hit)}",
        f"推理佔比　{_bar(reason_pct, bar_width)} {_pct(reason_pct)}",
        f"context　　{_bar(ctx_pct, bar_width)} {_pct(ctx_pct)}{ctx_note}",
        "",
    ]) + _cost_table(delta, price_in, price_cache, price_out,
                     show_cost_breakdown) + time_table + comp_table + [
        "</details>",
    ]
    return head + "\n" + "\n".join(detail)
