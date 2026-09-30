"""turn-usage 純函式測試：增量計算與格式化。

用標準庫 unittest（這台機器兩個 python 都沒裝 pytest）。
跑法：cd ~/.hermes/plugins/turn-usage && python3 -m unittest discover -s tests -t . -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from summary import (cache_pct, diff_totals, format_block, merge_sources,
                      route_text, strip_previous_block)  # noqa: E402


class TestCachePct(unittest.TestCase):
    def test_uses_cache_over_prompt_total(self):
        # CanonicalUsage.prompt_tokens = input + cache_read
        self.assertEqual(cache_pct(input_tokens=8000, cache_read_tokens=92000), 92)

    def test_none_without_cache_reads(self):
        self.assertIsNone(cache_pct(input_tokens=8000, cache_read_tokens=0))


class TestDiffTotals(unittest.TestCase):
    def test_subtracts_previous_snapshot(self):
        now = {"calls": 12, "in": 34200, "out": 3100, "cache": 410000, "cost": 0.0410, "reasoning": 1800}
        prev = {"calls": 5, "in": 14000, "out": 1200, "cache": 160000, "cost": 0.0170, "reasoning": 700}
        d = diff_totals(now, prev)
        self.assertEqual(d["calls"], 7)
        self.assertEqual(d["in"], 20200)
        self.assertEqual(d["out"], 1900)
        self.assertEqual(d["cache"], 250000)
        self.assertEqual(d["reasoning"], 1100)
        # 金額用 approx：0.041 - 0.017 在浮點下不是精確的 0.024
        self.assertAlmostEqual(d["cost"], 0.024, places=9)

    def test_without_previous_is_identity(self):
        now = {"calls": 3, "in": 100, "out": 20, "cache": 0, "cost": 0.5, "reasoning": 5}
        self.assertEqual(diff_totals(now, None), now)


class TestRouteText(unittest.TestCase):
    def test_single_upstream(self):
        self.assertEqual(route_text(["my-alias"], ["ProviderA"] * 10), "my-alias → ProviderA")

    def test_dominant_upstream_is_summarised(self):
        # 90% 以上只報一家，避免雜訊
        self.assertEqual(route_text(["my-alias"], ["ProviderA"] * 9 + ["ProviderB"]), "my-alias → ProviderA")

    def test_mixed_upstreams_show_top_two_with_pct(self):
        text = route_text(["my-alias"], ["ProviderA"] * 7 + ["ProviderB"] * 3)
        self.assertIn("ProviderA 70%", text)
        self.assertIn("ProviderB 30%", text)

    def test_no_upstreams_falls_back_to_alias(self):
        self.assertEqual(route_text(["my-alias"], []), "my-alias")

    def test_no_alias_no_upstreams(self):
        self.assertEqual(route_text([], []), "?")


class TestFormatBlock(unittest.TestCase):
    def test_contains_all_six_requested_fields(self):
        delta = {"calls": 7, "in": 20200, "out": 1900, "cache": 250000, "cost": 0.0240, "reasoning": 1100}
        totals = {"calls": 12, "in": 34200, "out": 3100, "cache": 410000, "cost": 0.0410, "reasoning": 1800}
        text = format_block(delta, totals, model="deepseek/deepseek-v4.1-flash",
                            route="my-alias → ProviderA", prompt_now=44704, context_max=131072, show_total=True)
        self.assertIn("deepseek/deepseek-v4.1-flash", text)  # 模型
        self.assertIn("ProviderA", text)                        # 經過路由
        self.assertIn("270.2K", text)                        # token 進（20.2K + 250K）
        self.assertIn("1.9K", text)                          # token 出
        self.assertIn("93%", text)                           # cache 比例（250k / 270.2k = 92.5 → 93）
        self.assertIn("$0.0240", text)                       # 本輪花費
        self.assertIn("$0.0410", text)                       # 累計花費
        self.assertIn("44.7K", text)                         # 目前 prompt 大小
        self.assertIn("34%", text)                           # context 使用比例（44704 / 131072）

    def test_handles_missing_context_max(self):
        delta = {"calls": 1, "in": 10, "out": 5, "cache": 0, "cost": 0.001, "reasoning": 0}
        text = format_block(delta, dict(delta), model="m", route="?",
                            prompt_now=44704, context_max=None, show_total=False)
        self.assertIn("44.7K", text)
        ctx_line = [ln for ln in text.splitlines() if "context" in ln][0]
        self.assertNotIn("(", ctx_line.split("context")[1])  # 沒有上限就不給百分比

    def test_omits_total_when_disabled(self):
        delta = {"calls": 1, "in": 10, "out": 5, "cache": 0, "cost": 0.001, "reasoning": 0}
        text = format_block(delta, dict(delta), model="m", route="?",
                            prompt_now=None, context_max=None, show_total=False)
        self.assertNotIn("累計", text)

    def test_handles_no_data_at_all(self):
        delta = {"calls": 0, "in": 0, "out": 0, "cache": 0, "cost": 0.0, "reasoning": 0}
        text = format_block(delta, dict(delta), model="m", route="?",
                            prompt_now=None, context_max=None, show_total=True)
        self.assertIn("—", text)          # 沒有資料時留白（不假造 0）
        self.assertIn("context", text)
        self.assertNotIn("%", text.split("context")[-1])   # context 那行不假造百分比


class TestMergeSources(unittest.TestCase):
    def test_groups_tools_into_named_categories(self):
        rows = dict(merge_sources([
            ("terminal", 520), ("execute_code", 50), ("read_file", 200),
            ("skill_view", 90), ("", 140),
        ]))
        self.assertEqual(rows["💻 命令行"], 57)   # (520+50)/1000
        self.assertEqual(rows["📄 讀檔"], 20)
        self.assertEqual(rows["📚 skill"], 9)
        self.assertEqual(rows["💬 對話"], 14)

    def test_patch_is_its_own_category_not_filed_under_reads(self):
        """改檔的 diff 和讀檔是兩件事，混在一起會讓人以為「讀了很多檔案」。"""
        rows = dict(merge_sources([("patch", 300), ("read_file", 100)]))
        self.assertEqual(rows["🔧 改檔"], 75)
        self.assertEqual(rows["📄 讀檔"], 25)
        self.assertNotIn("📄 檔案", rows)

    def test_small_buckets_collapse_into_other(self):
        rows = merge_sources([("terminal", 995), ("web_search", 5)])
        labels = [label for label, _pct in rows]
        self.assertNotIn("🌐 網路", labels)
        self.assertIn("▫️ 其他", labels)

    def test_empty_is_empty(self):
        self.assertEqual(merge_sources([]), [])
        self.assertEqual(merge_sources([("terminal", 0)]), [])


class TestOneLineForm(unittest.TestCase):
    """整包收進折疊，那行本身就是展開鈕（外行只剩一行）。"""

    def _text(self, timing=None):
        delta = {"calls": 5, "in": 33000, "out": 4200, "cache": 2_267_000,
                 "cost": 0.0083, "reasoning": 3100}
        totals = {"calls": 306, "in": 1_400_000, "out": 330_000,
                  "cache": 70_900_000, "cost": 0.494, "reasoning": 208_000}
        return format_block(delta, totals, model="m", route="r",
                            prompt_now=485_100, context_max=1_048_576, timing=timing)

    def test_first_line_is_the_details_opener(self):
        text = self._text({"think_s": 61, "tool_s": 0.9, "calls": 5, "tools": 5,
                           "cum_active_s": 754})
        first = text.splitlines()[0]
        self.assertTrue(first.startswith("<details><summary>📊用量 $0.008"))
        self.assertIn("⏱️活躍 1m02s", first)
        self.assertTrue(first.endswith("</summary>"))
        self.assertEqual(text.count("<details>"), 1)
        self.assertEqual(text.count("</details>"), 1)

    def test_no_stray_line_outside_the_fold(self):
        text = self._text({"think_s": 61, "tool_s": 0.9, "calls": 5, "tools": 5})
        self.assertTrue(text.splitlines()[0].startswith("<details>"))

    def test_new_block_is_stripped_too(self):
        block = self._text({"think_s": 10, "tool_s": 1, "calls": 2, "tools": 2})
        body = "回覆正文"
        self.assertEqual(strip_previous_block(body + "\n\n" + block), body)
        twice = strip_previous_block(body + "\n\n" + block) + "\n\n" + block
        self.assertEqual(twice.count("<details><summary>📊用量"), 1)


class TestStripVariants(unittest.TestCase):
    """同一則回覆出現兩份摘要，一新一舊。

    舊那份的標題是 `<details><summary>📊 用量 $X ｜ ⏱️ 活躍 Ys`（📊 後面有空格、
    中間有 ｜），而舊的字面比對只認 `<details><summary>📊用量`（無空格）→
    找不到就把舊的留著、新的接上去，畫面變成兩份。所以剝除改成用正則認「標題含 📊 與 用量」。
    """

    OLD = ("<details><summary>📊 用量 $0.0100 ｜ ⏱️ 活躍 9m09s</summary>\n\n"
           "`m`　路由 r\n\n|  | 本輪 | 累計 |\n|---|---|---|\n| ⚡ 呼叫 | 7 | 295 |\n\n</details>")
    NEW = ("<details><summary>📊用量 $0.0200 ⏱️活躍 8m08s</summary>\n\n"
           "`m`　路由 r\n\n|  | 本輪 | 累計 |\n|---|---|---|\n| ⚡ 呼叫 | 6 | 301 |\n\n</details>")
    LEGACY = "📊 **本輪用量** 💰$0.0300 ｜ ⚡5 call ｜ 📥2.3M"

    def test_strips_the_spaced_variant(self):
        self.assertEqual(strip_previous_block("正文\n\n" + self.OLD), "正文")

    def test_strips_both_when_old_and_new_are_stacked(self):
        """真實症狀：一則訊息裡舊格式接著新格式。"""
        text = "正文\n\n" + self.OLD + "\n\n" + self.NEW
        out = strip_previous_block(text)
        self.assertEqual(out, "正文")
        self.assertNotIn("📊", out)

    def test_strips_three_of_mixed_formats(self):
        text = "正文\n\n" + self.LEGACY + "\n\n" + self.OLD + "\n\n" + self.NEW
        self.assertEqual(strip_previous_block(text), "正文")

    def test_keeps_text_after_the_block(self):
        out = strip_previous_block("正文\n\n" + self.NEW + "\n\n---\n\n附註")
        self.assertIn("附註", out)
        self.assertNotIn("📊用量", out)

    def test_strips_the_separator_too(self):
        out = strip_previous_block("正文\n\n---\n\n" + self.NEW)
        self.assertEqual(out, "正文")


class TestStripPreviousBlock(unittest.TestCase):
    """hook 在同一個 turn 內會被叫多次，必須冪等。

    實測症狀：同一則回覆尾端疊了三份摘要，而且三份的累計值各不相同（因為每次
    讀到的當下累計不同）—— 那不是「三次獨立統計」，是同一件事長了三層。
    """

    BLOCK = ("---\n📊 **本輪用量** · `m` · 路由 x · 1 次呼叫\n"
             "token 進 1（其中快取 0%）／出 1（推理佔 0%）· 本輪 $0.0001 · context 1\n"
             "累計 $0.1 · 1 次呼叫 · 進 1／出 1")

    def test_no_block_is_untouched(self):
        body = "回答內容\n\n第二段"
        self.assertEqual(strip_previous_block(body), body)

    def test_single_block_is_removed(self):
        self.assertEqual(strip_previous_block(f"回答內容\n\n{self.BLOCK}"), "回答內容")

    def test_three_stacked_blocks_all_removed(self):
        text = "回答內容\n\n" + "\n\n".join([self.BLOCK] * 3)
        out = strip_previous_block(text)
        self.assertEqual(out, "回答內容")
        self.assertNotIn("本輪用量", out)

    def test_separator_line_is_removed_too(self):
        # 不留孤零零的 ---
        self.assertNotIn("---", strip_previous_block(f"回答內容\n\n{self.BLOCK}"))

    def test_idempotent(self):
        text = "回答內容\n\n" + "\n\n".join([self.BLOCK] * 2)
        once = strip_previous_block(text)
        self.assertEqual(strip_previous_block(once), once)

    def test_keeps_legitimate_horizontal_rule_in_body(self):
        # 正文自己用 --- 當分隔線時不能被誤刪
        self.assertEqual(strip_previous_block("上半\n\n---\n\n下半"), "上半\n\n---\n\n下半")

    def test_keeps_body_before_a_real_block(self):
        self.assertEqual(
            strip_previous_block("第一段\n\n---\n\n第二段\n\n" + self.BLOCK),
            "第一段\n\n---\n\n第二段",
        )

    def test_reappending_stays_single(self):
        """模擬真正的使用情境：連續三次呼叫，每次都剝掉再接。"""
        text = "回答內容"
        for _ in range(3):
            text = strip_previous_block(text) + "\n\n" + self.BLOCK
        self.assertEqual(text.count("本輪用量"), 1)


class TestContextHint(unittest.TestCase):
    """context 用滿時，外行那行直接標 ⚠️（門檻 context_hint_pct，0 = 關）。"""

    DELTA = {"calls": 3, "in": 1000, "cache": 9000, "out": 500, "cost": 0.01, "reasoning": 200}

    def _render(self, **kw):
        return format_block(self.DELTA, dict(self.DELTA), model="m", route="?", **kw)

    def test_hint_appears_at_or_above_threshold(self):
        text = self._render(prompt_now=800_000, context_max=1_000_000)
        self.assertIn("⚠️context 80%", text)
        self.assertIn("建議換新對話", text)

    def test_silent_below_threshold(self):
        self.assertNotIn("建議換新對話", self._render(prompt_now=100_000, context_max=1_000_000))

    def test_can_be_disabled_with_zero(self):
        self.assertNotIn(
            "建議換新對話",
            self._render(prompt_now=900_000, context_max=1_000_000, context_hint_pct=0),
        )

    def test_custom_threshold(self):
        self.assertIn(
            "建議換新對話",
            self._render(prompt_now=100_000, context_max=1_000_000, context_hint_pct=5),
        )

    def test_silent_without_context_max(self):
        self.assertNotIn("建議換新對話", self._render(prompt_now=900_000, context_max=None))


if __name__ == "__main__":
    unittest.main()
