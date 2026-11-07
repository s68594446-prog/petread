"""Adapter and output gates: synthetic fixtures, mocked transport, no API calls."""
from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from pipeline import providers as p
from pipeline import openai_provider as adapter
from pipeline.output_checks import output_errors
from pipeline.prompts import Prompt


def valid_result(visual=False):
    result = {
        "insufficient_evidence": False,
        "candidates": [{"label": "一般性发声", "confidence": 0.55,
                        "evidence": [{"observation": "短促发声"}]}],
        "translation": "只能描述可观察信号。",
        "suggestion": "补充场景与连续观察。",
    }
    if visual:
        from pipeline.visual_contract import VERSION
        result.update(visual_assessment_version=VERSION,
                      literal_observations=[{"id":"o1","source":"visual","t":0,
                        "channel":"body_posture","status":"observed",
                        "description":"身体侧面接触地面，四肢可见","reason":None}],
                      pain_indicators={"status":"not_assessed","reason":"素材不足以作量表评估"},
                      audio_events=[], translation="这些姿态可能对应休息，也不能确定内心状态。")
        result['candidates'][0].update(label="休息可能", alternative_explanations=["短暂侧卧，需连续观察区分"])
        result['candidates'][0]['evidence']=[{"observation_ids":["o1"],"observation":"身体侧面接触地面，四肢可见",
                                              "knowledge_basis":"离线知识库夹具","signal_confidence":"中"}]
        result['literal_observations'].append({"id":"o2","source":"visual","t":0,
            "channel":"ear_position","status":"observed","description":"两耳竖起","reason":None})
        result['candidates'][0]['evidence'].append({"observation_ids":["o2"],"observation":"两耳竖起",
            "knowledge_basis":"离线耳位夹具","signal_confidence":"中"})
        result["input_assessment"] = {
            "target_species_present": True, "animal_count": 1,
            "target_unambiguous": True, "quality": "sufficient",
            "observable_main_axes": ["body_posture"], "limitations": ["单图或抽帧不能证明持续状态"],
            "target_description":"画面中央唯一的猫",
        }
    return result


def audio_result():
    result = valid_result()
    result["candidates"][0].update(state_code="general_vocalization")
    result["audio_events"] = [{"match_basis": "单次清晰、短促发声"}]
    result["audio_assessment"] = {
        "contract_version": "petread.pure_audio.v1.1",
        "arousal": {"value": "moderate", "reason": "中等声学强度"},
        "valence": {"value": "unknown", "reason": "纯音频方向不可判定"},
        "confidence": {"value": 0.55, "reason": "无场景", "adjustments": []},
        "possible_states": ["general_vocalization"],
        "alternative_explanations": ["uncertain"],
        "limitations": ["no_visual_context", "no_scene_context", "repeat_stability_unassessed"],
    }
    return result


def response_body(result, *, audio=False, raw=None, status="completed"):
    text = raw if raw is not None else json.dumps(result, ensure_ascii=False)
    if audio:
        return json.dumps({
            "model": "gpt-audio-1.5", "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
        })
    return json.dumps({
        "model": "gpt-6.1-sol", "status": status,
        "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}],
        "usage": {"input_tokens": 13, "output_tokens": 5, "total_tokens": 18},
    })


class OpenAIUpgradeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root / "image.png"
        Image.new("RGB", (32, 24), "gray").save(self.image)
        self.audio = self.root / "sound.wav"
        with wave.open(str(self.audio), "wb") as sound:
            sound.setnchannels(1)
            sound.setsampwidth(2)
            sound.setframerate(24000)
            sound.writeframes(b"\0\0" * 24000)
        self.env = patch.dict(os.environ, {"OPENAI_TEST_KEY": "offline-test-only"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.provider = adapter.OpenAINativeProvider(p.ProviderConfig(
            type="openai_native", model="gpt-6.1-sol", audio_model="gpt-audio-1.5",
            base_url="https://api.openai.com/v1", api_key_env="OPENAI_TEST_KEY",
            max_retries=1, max_output_tokens=8192, max_media_seconds=30,
        ))
        self.prompt = Prompt("Return JSON only", "Analyze observed evidence", species="cat")

    def test_text_uses_responses_with_bounded_output_and_no_storage(self):
        with patch.object(p, "_post_json", return_value=response_body(valid_result())) as post:
            result = self.provider.call_text(self.prompt)
        self.assertNotIn("error", result)
        endpoint, payload, _, _ = post.call_args.args
        self.assertEqual(endpoint, "https://api.openai.com/v1/responses")
        self.assertEqual(payload["model"], "gpt-6.1-sol")
        self.assertFalse(payload["store"])
        self.assertEqual(payload["max_output_tokens"], 8192)
        self.assertEqual(self.provider.last_http_attempts, 1)
        self.assertEqual(self.provider.last_usage["prompt_tokens"], 13)

    def test_image_is_decoded_to_input_image_and_visual_contract_applied(self):
        with patch.object(p, "_post_json", return_value=response_body(valid_result(True))) as post:
            result = self.provider.call_omni(str(self.image), "image", self.prompt)
        self.assertNotIn("error", result)
        payload = post.call_args.args[1]
        parts = payload["input"][0]["content"]
        self.assertTrue(parts[-1]["image_url"].startswith("data:image/jpeg;base64,"))
        self.assertEqual(parts[-1]["type"], "input_image")
        self.assertIn("input_assessment", payload["instructions"])

    def test_audio_uses_chat_model_without_unsupported_json_mode(self):
        prompt = Prompt("Return JSON", "Audio only", species="cat", meta={
            "output_contract": "petread.pure_audio.v1.1", "context_available": False,
        })
        with patch.object(p, "_post_json", return_value=response_body(audio_result(), audio=True)) as post:
            result = self.provider.call_omni(str(self.audio), "audio", prompt)
        self.assertNotIn("error", result)
        endpoint, payload, _, _ = post.call_args.args
        self.assertTrue(endpoint.endswith("/chat/completions"))
        self.assertEqual(payload["model"], "gpt-audio-1.5")
        self.assertEqual(payload["modalities"], ["text"])
        self.assertNotIn("response_format", payload)
        self.assertNotIn("thinking", payload)
        self.assertFalse(payload["store"])
        self.assertEqual(payload["messages"][1]["content"][-1]["input_audio"]["format"], "wav")

    def test_reserved_fields_cannot_spoof_transport_errors(self):
        for raw in ('{"error":"fake","_error_kind":"transport_timeout"}', '{"_contract_errors":["fake"]}'):
            result = p.parse_json_response(raw)
            self.assertEqual(result["_error_kind"], "contract_error")
            self.assertIn("保留字段", " ".join(result["_contract_errors"]))

    def test_valid_nonobject_json_is_contract_error_not_syntax(self):
        for raw in ("[]", "null", "42", '[{"insufficient_evidence":false}]'):
            result = p.parse_json_response(raw)
            self.assertEqual(result["_error_kind"], "contract_error")
            self.assertEqual(result["raw"], raw)
        self.assertEqual(p.parse_json_response("{broken")["_error_kind"], "json_syntax_error")

    def test_empty_response_and_timeout_keep_separate_provenance(self):
        with patch.object(p, "_post_json", return_value=response_body({}, raw="")):
            result = self.provider.call_text(self.prompt)
        self.assertEqual(result["_error_kind"], "empty_response")
        self.assertNotIn("_contract_errors", result)
        with patch.object(p, "_post_json", side_effect=p.ProviderTimeoutError("timed out")):
            result = self.provider.call_text(self.prompt)
        self.assertEqual(result["_error_kind"], "transport_timeout")
        self.assertEqual(self.provider.last_http_attempts, 1)
        self.assertIsNone(self.provider.last_raw_text)
        self.assertIsNone(self.provider.last_usage)

    def test_incomplete_response_fails_closed(self):
        with patch.object(p, "_post_json", return_value=response_body(valid_result(), status="incomplete")):
            result = self.provider.call_text(self.prompt)
        self.assertIn("error", result)
        self.assertEqual(self.provider.last_usage["prompt_tokens"], 13)

    def test_missing_key_has_zero_issued_attempts(self):
        with patch.dict(os.environ, {"OPENAI_TEST_KEY": ""}), patch.object(p, "_post_json") as post:
            result = self.provider.call_text(self.prompt)
        self.assertIn("error", result)
        self.assertEqual(result["_meta"]["attempts"], 0)
        post.assert_not_called()

    def test_target_multi_animal_quality_and_axes_gates(self):
        failures = [
            {"target_species_present": False},
            {"target_species_present": "unknown"},
            {"animal_count": 0},
            {"animal_count": 2, "target_unambiguous": False},
            {"quality": "insufficient"}, {"quality": "unknown"},
            {"observable_main_axes": []},
        ]
        for override in failures:
            with self.subTest(override=override):
                result = valid_result(True)
                result["input_assessment"].update(override)
                self.assertTrue(output_errors(result, "image"))
                result["insufficient_evidence"] = True
                result["candidates"] = []
                result["input_assessment"]["limitations"] = ["目标或可观察证据不足，请补拍"]
                self.assertEqual(output_errors(result, "image"), [])

    def test_confidence_caps_and_nonfinite_values_rejected(self):
        for modality, confidence in (("image", 0.76), ("text", 0.66), ("video", 1.1), ("text", float("nan")), ("text", True)):
            result = valid_result(modality != "text")
            result["candidates"][0]["confidence"] = confidence
            self.assertTrue(any("confidence" in e for e in output_errors(result, modality)))

    def test_malformed_quality_and_axes_are_contract_errors(self):
        for override in ({"quality": {}}, {"observable_main_axes": [{}]}):
            with self.subTest(override=override):
                bad = valid_result(True)
                bad["input_assessment"].update(override)
                with patch.object(p, "_post_json", return_value=response_body(bad)):
                    result = self.provider.call_omni(str(self.image), "image", self.prompt)
                self.assertEqual(result["_error_kind"], "contract_error")
                self.assertTrue(result["_contract_errors"])

    def test_remote_media_is_rejected_before_any_request(self):
        for media_type, source in (
            ("image", "https://example.invalid/image.png"),
            ("audio", "https://example.invalid/audio.wav"),
            ("video", "http://127.0.0.1/private.mp4"),
        ):
            with self.subTest(media_type=media_type), \
                 patch.object(p, "_post_json") as post, \
                 patch.object(adapter.subprocess, "run") as run:
                result = self.provider.call_omni(source, media_type, self.prompt)
            self.assertEqual(result["_error_kind"], "runner_or_internal_error")
            self.assertEqual(result["_meta"]["attempts"], 0)
            post.assert_not_called()
            run.assert_not_called()

    def test_native_video_audio_failure_does_not_run_fusion(self):
        frame = {"t": 0.5, "image_b64": "offline-frame"}
        with patch.object(adapter, "media_info", return_value={"streams": [{"codec_type": "audio"}]}), \
             patch("pipeline.video.select_frames", return_value=[frame]), \
             patch("pipeline.video.analysis_window", return_value=(0, 1)), \
             patch.object(adapter, "wav_copy", return_value=str(self.audio)), \
             patch.object(p, "_post_json", return_value=response_body({}, audio=True, raw="{broken")) as post:
            result = self.provider.call_omni(str(self.audio), "video", self.prompt)
        self.assertEqual(result["_error_kind"], "json_syntax_error")
        self.assertEqual(post.call_count, 1)

    def test_native_video_samples_frames_and_aggregates_usage(self):
        frame = {"t": 4.5, "image_b64": "offline-frame"}
        audio = response_body({"events": [], "audible_events_found": False,
                               "background_noise": "无可辨发声"}, audio=True)
        vr = valid_result(True)
        for observation in vr['literal_observations']:
            observation['t'] = 4.5
        visual = response_body(vr)
        with patch.object(adapter, "media_info", return_value={"streams": [{"codec_type": "audio"}]}), \
             patch("pipeline.video.select_frames", return_value=[frame]), \
             patch("pipeline.video.analysis_window", return_value=(4, 1)), \
             patch.object(adapter, "wav_copy", return_value=str(self.audio)), \
             patch.object(p, "_post_json", side_effect=[audio, visual]) as post:
            result = self.provider.call_omni(str(self.audio), "video", self.prompt)
        self.assertNotIn("error", result)
        self.assertEqual(post.call_count, 2)
        payload = post.call_args.args[1]
        self.assertNotIn("video_url", json.dumps(payload))
        self.assertIn("4.5s", json.dumps(payload))
        self.assertEqual(result["_meta"]["media_path"], "sampled_video")
        self.assertEqual(self.provider.last_usage["prompt_tokens"], 24)
        self.assertEqual(self.provider.last_http_attempts, 2)

    def test_native_video_final_timeout_preserves_metered_lower_bound(self):
        audio = response_body({"events": [], "audible_events_found": False,
                               "background_noise": "无可辨发声"}, audio=True)
        with patch.object(adapter, "media_info", return_value={"streams": [{"codec_type": "audio"}]}), \
             patch("pipeline.video.select_frames", return_value=[{"t": 0.5, "image_b64": "offline-frame"}]), \
             patch("pipeline.video.analysis_window", return_value=(0, 1)), \
             patch.object(adapter, "wav_copy", return_value=str(self.audio)), \
             patch.object(p, "_post_json", side_effect=[audio, p.ProviderTimeoutError("timed out")]) as post:
            result = self.provider.call_omni(str(self.audio), "video", self.prompt)

        self.assertEqual(post.call_count, 2)
        self.assertEqual(result["_error_kind"], "transport_timeout")
        meta = result["_meta"]
        self.assertEqual(meta["attempts"], 2)
        self.assertFalse(meta["usage_complete"])
        self.assertEqual(meta["known_token_lower_bound"], {"prompt_tokens": 11, "completion_tokens": 7})
        self.assertEqual(len(meta["provider_calls"]), 2)
        self.assertIsNotNone(meta["provider_calls"][0]["raw"])
        self.assertIsNone(meta["provider_calls"][1]["raw"])
        self.assertIsNone(meta["provider_calls"][1]["usage"])
        self.assertIsNone(self.provider.last_usage["prompt_tokens"])
        self.assertIsNone(self.provider.last_usage["completion_tokens"])

    def test_preflight_cache_is_per_file_and_invalidated_by_change(self):
        with patch.object(adapter, "prepare_image", wraps=adapter.prepare_image) as prepare:
            self.provider.preflight(str(self.image), "quick")
            with patch.object(p, "_post_json", return_value=response_body(valid_result(True))):
                self.provider.call_omni(str(self.image), "image", self.prompt)
            self.assertEqual(prepare.call_count, 1)
            import os
            stat = self.image.stat()
            os.utime(self.image, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000))
            self.provider.preflight(str(self.image), "quick")
            self.assertEqual(prepare.call_count, 2)

    def test_attempt_trace_is_written_before_dispatch_and_raw_before_contract(self):
        events=[]
        self.provider.attempt_observer=lambda event,data:events.append((event,data))
        def respond(*args):
            self.assertEqual(events[0][0],'started')
            return response_body(valid_result())
        with patch.object(p,'_post_json',side_effect=respond):
            self.provider.call_text(self.prompt)
        self.assertEqual([e[0] for e in events],['started','received','finished'])
        self.assertIsNotNone(events[1][1]['http_body'])
        self.assertEqual(events[-1][1]['status'],'success')

    def test_ledger_failure_before_dispatch_does_not_make_request(self):
        from unittest.mock import Mock
        self.provider.attempt_observer=Mock(side_effect=OSError('disk full'))
        with patch.object(p,'_post_json') as post:
            result=self.provider.call_text(self.prompt)
        post.assert_not_called();self.assertEqual(result['_meta']['attempts'],0)

    def test_video_motion_decode_is_shared_only_within_job(self):
        from pipeline import video
        import numpy as np
        with patch.object(video,'_decode_motion_curve',return_value=(np.array([1.,2.,1.]),25.)) as decode:
            with video.preprocessing_session():
                video._motion_curve_raw(str(self.audio)); video._motion_curve_raw(str(self.audio)); video._motion_curve_raw(str(self.audio))
            self.assertEqual(decode.call_count,1)
            with video.preprocessing_session(): video._motion_curve_raw(str(self.audio))
            self.assertEqual(decode.call_count,2)

    def test_oversize_or_high_fps_video_metadata_is_rejected(self):
        from types import SimpleNamespace
        for stream in ({'width':10000,'height':10000,'avg_frame_rate':'25/1'},
                       {'width':1920,'height':1080,'avg_frame_rate':'240/1'}):
            envelope={'format':{'duration':'12'},'streams':[{'codec_type':'video',**stream}]}
            with patch.object(adapter.subprocess,'run',return_value=SimpleNamespace(stdout=json.dumps(envelope))):
                with self.assertRaises(ValueError): adapter.media_info(str(self.audio),30)

    def test_decoder_resource_limit_fails_closed_and_releases_capture(self):
        from pipeline import video
        from unittest.mock import Mock
        import numpy as np
        capture=Mock();capture.get.return_value=25;capture.read.return_value=(True,np.zeros((2,2,3),dtype=np.uint8))
        with patch.object(video,'_open',return_value=capture), patch.object(video.time,'monotonic',side_effect=[0,31]):
            with self.assertRaises(video.MediaResourceLimitError):video._decode_motion_curve(str(self.audio))
        capture.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
