"""Offline contract/regression tests. No credentials or real model calls."""
import asyncio
import base64
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from pipeline import analyze, providers, store
from pipeline.prompts import load_knowledge


class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.provider = providers.build_providers(providers.load_config())["omni"]
        # Synthetic test value, never a real key. Network is forbidden even on test mistakes.
        self.env = patch.dict(os.environ, {"MIMO_API_KEY": "offline-test-only"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.guard = patch("socket.socket.connect", side_effect=AssertionError("network forbidden"))
        self.guard.start()
        self.addCleanup(self.guard.stop)
        self.result = {"candidates": [{"label": "放松", "confidence": .8}],
                       "insufficient_evidence": False, "visual_signals": [], "audio_events": []}

    def media(self, ext):
        p = self.root / ("fixture" + ext)
        # These bytes only test wire format, NOT media decoding or model understanding.
        p.write_bytes(b"synthetic-media")
        return str(p)

    def response(self, content=None, reason="stop", **message):
        return json.dumps({"choices": [{"finish_reason": reason, "message": {
            "content": json.dumps(self.result) if content is None else content, **message}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 40,
                      "completion_tokens_details": {"reasoning_tokens": 20}}})

    def test_active_config_and_token_budget(self):
        p = self.provider._build_payload("system", "text")
        self.assertEqual(p["model"], "mimo-v2.6-flash")
        self.assertEqual(p["max_completion_tokens"], 32768)
        self.assertEqual(p["thinking"], {"type": "enabled"})
        self.assertEqual(p["response_format"], {"type": "json_object"})
        self.assertNotIn("fps", p)
        self.assertNotIn("media_resolution", p)
        self.assertNotIn("temperature", p)

    def test_invalid_token_limits(self):
        for value in (0, -1, 131073, True, "1024", 1.5):
            with self.subTest(value=value), self.assertRaises(providers.ProviderConfigError):
                providers.ProviderConfig(type="openai", max_completion_tokens=value)

    def test_mimo_option_validation_and_thinking_disabled(self):
        cfg = dict(type="openai", media_format="mimo", media_param_placement="content_part",
                   max_completion_tokens=32768)
        for overrides in ({"fps": 0}, {"fps": 11}, {"media_resolution": "invalid"},
                          {"media_param_placement": "content"}, {"max_completion_tokens": None}):
            with self.subTest(overrides=overrides), self.assertRaises(providers.ProviderConfigError):
                providers.ProviderConfig(**{**cfg, **overrides})
        p = providers.OpenAICompatibleProvider(providers.ProviderConfig(**cfg, thinking=False))
        self.assertEqual(p._build_payload("", "test")["thinking"], {"type": "disabled"})

    def test_builtin_mock_mode_needs_no_key(self):
        cfg = providers.load_config()
        cfg["mock"] = True
        p = providers.build_providers(cfg)["omni"]
        self.assertIsInstance(p, providers.MockProvider)
        with patch.object(providers, "_post_json", side_effect=AssertionError("mock made HTTP")):
            out = analyze.analyze("", "dog", "test", "text", p,
                                  load_knowledge("dog"), text_description="狗趴着")
        self.assertNotIn("error", out["result"])

    def test_video_url_and_inline_parameters(self):
        for src in ("https://example.invalid/video.mp4", self.media(".mp4")):
            part = self.provider._media_part("video", src)
            self.assertEqual(part["fps"], 2)
            self.assertEqual(part["media_resolution"], "default")
            self.assertEqual(set(part["video_url"]), {"url"})
        self.assertTrue(part["video_url"]["url"].startswith("data:video/mp4;base64,"))

    def test_image_url_and_inline_have_no_video_options(self):
        for src in ("https://example.invalid/image.png", self.media(".png")):
            part = self.provider._media_part("image", src)
            self.assertEqual(set(part), {"type", "image_url"})
            self.assertEqual(set(part["image_url"]), {"url"})
        self.assertTrue(part["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_mimo_audio_data_uri_and_url(self):
        for ext, mime in ((".wav", "audio/wav"), (".mp3", "audio/mpeg"),
                          (".flac", "audio/flac"), (".m4a", "audio/mp4"), (".ogg", "audio/ogg")):
            part = self.provider._media_part("audio", self.media(ext))
            self.assertEqual(set(part["input_audio"]), {"data"})
            prefix, data = part["input_audio"]["data"].split(",", 1)
            self.assertEqual(prefix, f"data:{mime};base64")
            self.assertEqual(base64.b64decode(data), b"synthetic-media")
        url = "https://example.invalid/audio.wav"
        self.assertEqual(self.provider._media_part("audio", url)["input_audio"], {"data": url})

    def test_unsupported_audio_is_explicit_error(self):
        result = self.provider.call_omni(self.media(".aac"), "audio", "test")
        self.assertIn("ProviderConfigError", result["error"])

    def test_generic_openai_audio_still_uses_bare_base64(self):
        p = providers.OpenAICompatibleProvider(providers.ProviderConfig(type="openai"))
        audio = p._media_part("audio", self.media(".wav"))["input_audio"]
        self.assertEqual(audio["format"], "wav")
        self.assertFalse(audio["data"].startswith("data:"))

    def test_json_filtering_and_usage(self):
        obj = {"candidates": [{"label": "保留", "confidence": .5},
                              {"label": "拒绝", "confidence": .49},
                              {"label": "缺置信度"}]}
        with patch.object(providers, "_post_json", return_value=self.response(json.dumps(obj))):
            r = self.provider.call_text("test")
        self.assertNotIn("error", r)
        self.assertEqual([x["label"] for x in r["candidates"]], ["保留"])
        self.assertEqual(r["_meta"]["model"], "mimo-v2.6-flash")
        self.assertEqual(self.provider.last_usage["completion_tokens"], 40)

    def test_low_confidence_refusal(self):
        obj = {"candidates": [{"label": "拒绝", "confidence": .1}]}
        with patch.object(providers, "_post_json", return_value=self.response(json.dumps(obj))):
            r = self.provider.call_text("test")
        self.assertTrue(r["insufficient_evidence"])
        self.assertEqual(store.summarize_result(r), (None, None))

    def test_bad_or_fenced_json(self):
        for text, error in (("not json", True), ("```json\n" + json.dumps(self.result) + "\n```", False)):
            with patch.object(providers, "_post_json", return_value=self.response(text)):
                r = self.provider.call_text("test")
            self.assertEqual("error" in r, error)

    def test_incomplete_output_is_rejected_without_retry_or_stale_text(self):
        for reason in (None, "length", "content_filter", "tool_calls", "repetition_truncation"):
            self.provider.last_raw_text = "stale"
            with patch.object(providers, "_post_json", return_value=self.response(reason=reason)) as post:
                r = self.provider.call_text("test")
            self.assertIn("error", r)
            self.assertEqual(post.call_count, 1)
            self.assertIsNone(self.provider.last_raw_text)

    def test_reasoning_is_never_a_final_answer(self):
        with patch.object(providers, "_post_json", return_value=self.response(
                "", reasoning_content=json.dumps(self.result))):
            r = self.provider.call_text("test")
        self.assertIn("error", r)

    def test_all_analysis_modes_offline(self):
        knowledge = load_knowledge("dog")
        fake_video = types.ModuleType("pipeline.video")
        fake_video.motion_features = lambda _: {"duration": 1, "motion_peaks": []}
        fake_video.select_frames = lambda *a, **k: [{"t": 0, "image_b64": "dGVzdA=="}]
        fake_video.analysis_window = lambda _: (0, 1)
        fake_video.extract_audio = lambda *a, **k: self.media(".wav")
        for mode, ext, calls in (("text", ".txt", 1), ("quick", ".png", 1),
                                 ("audio", ".wav", 1), ("native", ".mp4", 1), ("frames", ".mp4", 3)):
            with self.subTest(mode=mode), patch.dict(sys.modules, {"pipeline.video": fake_video}), \
                 patch.object(providers, "_post_json", return_value=self.response()) as post:
                out = analyze.analyze(self.media(ext), "dog", "test", mode,
                                      self.provider, knowledge, text_description="狗趴着")
                self.assertNotIn("error", out["result"])
                self.assertEqual(out["model"], "mimo-v2.6-flash")
                self.assertEqual(out["tokens_out"], 40 * calls)
                self.assertEqual(post.call_count, calls)
                for call in post.call_args_list:
                    self.assertEqual(call.args[1]["max_completion_tokens"], 32768)
                if mode == "audio":
                    self.assertEqual(out["result"]["visual_signals"], [])

    def test_records_feedback_and_export_roundtrip(self):
        db = str(self.root / "petread.db")
        store.init_db(db)
        label, confidence = store.summarize_result(self.result)
        rid = store.insert_record(species="dog", mode="text", context="test", media_path="",
            result=self.result, predicted_label=label, confidence=confidence, tokens_in=100,
            tokens_out=40, latency_s=.1, provider="openai", model="mimo-v2.6-flash",
            client_id="device-a", db_path=db)
        rec = store.get_record(rid, db)
        self.assertEqual(rec["model"], "mimo-v2.6-flash")
        self.assertEqual(rec["result"], self.result)
        with patch.object(store, "DEFAULT_DB_PATH", db):
            import app
            response = asyncio.run(app.feedback(rid, {"actual_label": "放松", "user_note": "确认"}))
            self.assertEqual(response.status_code, 200)
            with self.assertRaises(app.HTTPException) as cm:
                asyncio.run(app.feedback(rid, {}))
            self.assertEqual(cm.exception.status_code, 422)
        self.assertEqual(store.list_feedback(rid, db)[0]["user_note"], "确认")
        self.assertEqual(store.list_records("device-b", db_path=db), [])
        self.assertEqual(store.list_records("device-a", db_path=db)[0]["feedback_count"], 1)
        rows = list(store.iter_export_rows(db))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["model"], "mimo-v2.6-flash")


if __name__ == "__main__":
    unittest.main()
