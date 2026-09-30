# hermes-turn-usage

A [Hermes Agent](https://github.com/NousResearch/hermes-agent) plugin that appends a compact,
collapsible **usage summary** to the end of every reply — model, route, tokens, cache hit
rate, cost and context occupancy — plus a **per-tool time breakdown**.

It is a pure-computation plugin: it never calls a model, and any failure leaves the reply
untouched.

## What it looks like

Collapsed (one line, always visible):

```
📊用量 $0.050 ⏱️活躍 5m00s
```

Expanded:

| | 本輪 | 累計 |
|---|---|---|
| ⚡ 呼叫 | 21 | 141 |
| 📥 進 | 4.0M | 16.4M |
| 💾 快取命中 | 100% | 99% |
| 📤 出 | 13.8K | 123.7K |
| 🤔 推理 | 57% | 67% |
| ⏱️ 活躍 | 3m11s | 28m59s |
| 💰 花費 | $0.0239 | $0.1487 |

**時間分配**（不含你閱讀與輸入的時間）

| 來源 | 本輪 | 累積 | 本輪佔比 | 次數 |
|---|---|---|---|---|
| 🧠 模型思考／生成 | 3m03s | 25m35s | 96% | 21 |
| ⚙️ 工具執行 | 7.7s | 3m24s | 4% | 21 |
| ⤷ 💻 命令行 | 7.3s | 7.3s | 4% | 14 |
| ⤷ 📄 讀檔 | 0.1s | 0.1s | 0% | 2 |
| ⤷ 👁️ 圖片識別 | 0.0s | 0.0s | 0% | 1 |

The tool breakdown is two-level: `⤷` categories (command line / file edit / file read /
image recognition / network / skill), then an indented row per tool. A category is expanded
only when it exceeds `time_detail_min_pct` and holds more than one tool.

## Install

```bash
# clone into your plugin directory
git clone https://github.com/<you>/hermes-turn-usage ~/.hermes/plugins/turn-usage

# enable + configure
hermes plugins enable turn-usage
# edit config: auxiliary providers use per-provider prices
```

Then **restart Hermes** — the plugin is loaded at session start.

## Configure

Everything is in the plugin's `config_schema` (`plugin.yaml`):

| Key | Default | Meaning |
|---|---|---|
| `show_session_total` | `true` | add the cumulative column |
| `show_time_table` | `true` | the time-breakdown table |
| `time_detail_min_pct` | `10` | a tool category must reach this share of active time (and hold >1 tool) before its per-tool rows are expanded. `0` = always expand |
| `show_composition` | `true` | prompt-composition table |
| `show_bars` | `true` | the meter bars |
| `bar_width` | `10` | meter width in cells |
| `context_hint_pct` | `70` | warn in the collapsed line once context occupancy reaches this % |
| `price_in` / `price_cache` / `price_out` | `0.14` / `0.0042` / `0.42` | **example** unit prices ($ per M tokens). Set these to your own provider's live prices |
| `composition_groups` | `""` | override the tool→category mapping, one `Label = tool1,tool2` per line |

## Design notes

- **Cost is `tokens × live unit price`**, keyed on `billing_provider` — *not* Hermes'
  `estimated_cost_usd`, which under-reports (measured 19–29% on main turns, 51–61% on
  auxiliary forks) and drifts in direction, so a constant correction factor does not work.
  Keying on the model name instead of the billing provider is a trap: the same model served
  by two providers can carry different cache rates.
- **Per-tool timing is parsed from `agent.log`** (`tool <name> completed (Xs, …)` and the
  model-call `latency=` lines), so no extra instrumentation is needed.
- **Waiting time is excluded.** `clarify` blocks on the human, so its seconds are reported
  on their own `⏸️` row marked *not counted* and kept out of the denominator — otherwise
  "waiting for you" reads as "the agent is slow".
- **Cache cost = context size × number of calls.** That is why the summary surfaces both:
  the cache column is usually the largest single line, and a long conversation pays it on
  every call.
- The block is re-rendered idempotently: any previously attached summary is stripped first,
  because the hook can fire more than once per turn.

## Requirements

- Hermes Agent ≥ 0.21 (uses the `transform_llm_output` hook)
- Python 3.9+ (stdlib only)

## Tests

```bash
python3 -m unittest discover -s tests -t .
```

The pure-function layer (`summary.py`) is covered by 38 unit tests — no network, no fixtures
on disk.

## Notes

- Source comments are written in Traditional Chinese; the code and configuration are plain
  English identifiers.

## License

MIT
