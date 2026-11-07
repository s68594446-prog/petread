"""Offline API/privacy regressions; all state lives in a temporary directory."""
from __future__ import annotations

import asyncio
import io
import json
import sqlite3
import subprocess
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
from fastapi import HTTPException, UploadFile

import app as server
from pipeline import store, runtime


OWNER = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
OTHER = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"


class ServerUpgradeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.patches = [
            patch.object(store, "DEFAULT_DB_PATH", str(self.root / "test.db")),
            patch.object(server, "UPLOAD_DIR", str(self.root)),
            patch.object(server, "_JOBS", {}),
            patch.object(server, "_TASKS", set()),
            patch.object(server, "_STATE", {"config": {"mock": True}}),
        ]
        for p in self.patches:
            p.start()
        store.init_db()
        runtime.init()
        self.tokens = {}
        for cid in (OWNER, OTHER):
            token, identity = runtime.create_session()
            with store.connect() as conn:
                conn.execute("UPDATE device_sessions SET client_id=?,consent_version=? WHERE client_id=?",
                             (cid, runtime.CONSENT_VERSION, identity["client_id"]))
            self.tokens[cid] = token
        self.record_id = store.insert_record(
            species="cat", mode="text", context="=1+1", media_path="private-path",
            result={"translation": "=HYPERLINK(\"x\")"}, predicted_label="test",
            confidence=0.5, tokens_in=None, tokens_out=None, latency_s=None,
            provider="fake", model="fake", client_id=OWNER,
        )
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app), base_url="http://offline.test",
            headers={"Authorization": "Bearer " + self.tokens[OWNER], "Idempotency-Key": "offline-request-0001"},
        )

    async def asyncTearDown(self):
        if server._TASKS:
            await asyncio.gather(*server._TASKS)
        await self.client.aclose()
        for p in reversed(self.patches):
            p.stop()
        self.temp.cleanup()

    async def test_record_and_feedback_require_owner(self):
        for cid in (OTHER, "invalid"):
            resp = await self.client.get(f"/record/{self.record_id}", params={"client_id": cid})
            self.assertIn(resp.status_code, (403, 404, 422))
            resp = await self.client.post(f"/feedback/{self.record_id}", json={
                "client_id": cid, "actual_label": "wrong-device",
            })
            self.assertIn(resp.status_code, (403, 404, 422))
        own = await self.client.get(f"/record/{self.record_id}", params={"client_id": OWNER})
        self.assertEqual(own.status_code, 200)
        self.assertNotIn("media_path", own.json())
        own_feedback = await self.client.post(f"/feedback/{self.record_id}", json={
            "client_id": OWNER, "actual_label": "observed",
        })
        self.assertEqual(own_feedback.status_code, 200)
        self.assertEqual(len(store.list_feedback(self.record_id)), 1)

    async def test_profiles_are_isolated_by_device(self):
        for cid, note in ((OWNER, "one"), (OTHER, "two")):
            resp = await self.client.post("/profiles", headers={"Authorization": "Bearer " + self.tokens[cid]}, json={
                "client_id": cid, "pet_name": "Mimi", "species": "cat", "calibration_notes": note,
            })
            self.assertEqual(resp.status_code, 200)
        response = await self.client.get("/profiles", params={"client_id": OWNER})
        self.assertEqual(len(response.json()["items"]), 1)
        self.assertEqual(response.json()["items"][0]["calibration_notes"], "one")
        self.assertEqual((await self.client.get("/profiles")).status_code, 200)

    async def test_legacy_profile_migration_preserves_unassigned_rows(self):
        legacy_path = str(self.root / "legacy.db")
        with closing(sqlite3.connect(legacy_path)) as conn:
            conn.executescript(store.SCHEMA.replace("    client_id         TEXT NOT NULL DEFAULT '',\n", ""))
            conn.execute(
                "INSERT INTO profiles (pet_name, species, calibration_notes, created_at) VALUES (?,?,?,?)",
                ("legacy pet", "cat", "keep this", "2020-01-01"),
            )
            conn.commit()
        store.init_db(legacy_path)
        with store.connect(legacy_path) as conn:
            row = conn.execute("SELECT * FROM profiles").fetchone()
            self.assertEqual(row["calibration_notes"], "keep this")
            self.assertEqual(row["client_id"], "")
        self.assertEqual(store.list_profiles(legacy_path, client_id=OWNER), [])

    async def test_sse_hides_foreign_job_events(self):
        jid, job = server._new_job(OWNER)
        server._job_finish(job, "done", {"private": "never-leak"})
        foreign = await self.client.get(f"/analyze/stream/{jid}", params={"client_id": OTHER}, headers={"Authorization": "Bearer " + self.tokens[OTHER]})
        self.assertNotIn("never-leak", foreign.text)
        self.assertIn("event: error", foreign.text)
        own = await self.client.get(f"/analyze/stream/{jid}", params={"client_id": OWNER})
        self.assertIn("never-leak", own.text)

    async def test_job_admission_bounds_and_completed_slot_reuse(self):
        with patch.object(server, "MAX_ACTIVE_JOBS", 1):
            _, job = server._new_job(OWNER)
            with self.assertRaises(HTTPException) as blocked:
                server._new_job(OTHER)
            self.assertEqual(blocked.exception.status_code, 429)
            server._job_finish(job, "done", {})
            server._new_job(OTHER)

    async def test_export_requires_header_and_neutralizes_formulas(self):
        with patch.dict("os.environ", {"EXPORT_KEY": "local-test-only"}):
            old = await self.client.get("/export", params={"key": "local-test-only"})
            self.assertEqual(old.status_code, 403)
            response = await self.client.get("/export", headers={"X-Export-Key": "local-test-only"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("'=1+1", response.text)
        self.assertEqual(response.headers["cache-control"], "no-store")

    async def test_provider_is_created_per_job(self):
        with patch.object(server.providers_mod, "build_providers", side_effect=lambda cfg: {"omni": object()}) as build:
            one, two = server._provider(), server._provider()
        self.assertIsNot(one, two)
        self.assertEqual(build.call_count, 2)

    async def test_upload_size_failure_removes_partial_file(self):
        upload = UploadFile(file=io.BytesIO(b"too much"), filename="file.wav")
        with patch.object(server, "MAX_AUDIO_BYTES", 1):
            with self.assertRaises(HTTPException) as error:
                await server._save_upload(upload, "audio")
        self.assertEqual(error.exception.status_code, 413)
        self.assertTrue(upload.file.closed)
        self.assertEqual(list(self.root.glob("*.wav")), [])

    async def test_video_ten_mb_limit_and_health_policy(self):
        self.assertEqual(server.MAX_VIDEO_BYTES, 10*1024*1024)
        upload=UploadFile(file=io.BytesIO(b'x'*(server.MAX_VIDEO_BYTES+1)),filename='video.mp4')
        with self.assertRaises(HTTPException) as error:
            await server._save_upload(upload,'native')
        self.assertEqual(error.exception.status_code,413)
        self.assertEqual(list(self.root.glob('*.mp4')),[])
        limits=(await self.client.get('/healthz')).json()['media_limits']
        self.assertEqual(limits['video_seconds'],15)
        self.assertEqual(limits['video_fps'],30)

    async def test_busy_job_cleans_new_upload(self):
        server._new_job(OWNER)
        with patch.object(server, "_provider", return_value=object()), \
             patch.object(server, "load_knowledge", return_value={}):
            response = await self.client.post("/analyze", data={
                "client_id": OWNER, "species": "cat", "mode": "audio", "context": "unavailable",
            }, files={"file": ("clip.wav", b"offline", "audio/wav")})
        self.assertEqual(response.status_code, 429)
        self.assertEqual(list(self.root.glob("*.wav")), [])

    async def test_model_trace_and_cleanup_on_success(self):
        jid, job = server._new_job(OWNER)
        path = self.root / "uploaded.wav"
        path.write_bytes(b"offline fixture")
        provider = SimpleNamespace(last_model="actual-audio-model")
        fake_out = {
            "result": {"translation": "observed", "candidates": []},
            "tokens_in": 7, "tokens_out": 8, "model": "default-vision-model",
        }
        with patch.object(server.analyze_mod, "analyze", return_value=fake_out):
            await server._run_analysis_job(
                job, media_path=str(path), size=15, species="cat", context="", mode="audio",
                provider=provider, knowledge={}, max_frames=8, text="", client_id=OWNER,
            )
        self.assertFalse(path.exists())
        event, payload = job["events"][-1]
        self.assertEqual(event, "done")
        self.assertEqual(payload["meta"]["model"], "actual-audio-model")
        stored = store.get_record(payload["record_id"], client_id=OWNER)
        self.assertEqual(stored["model"], "actual-audio-model")

    async def test_internal_error_details_are_not_sent_to_browser(self):
        _, job = server._new_job(OWNER)
        with patch.object(server.analyze_mod, "analyze", side_effect=RuntimeError("secret/server/path")):
            await server._run_analysis_job(
                job, media_path="", size=0, species="cat", context="", mode="text",
                provider=object(), knowledge={}, max_frames=8, text="meow", client_id=OWNER,
            )
        self.assertNotIn("secret/server/path", json.dumps(job["events"]))
        self.assertTrue(job["done"])

    async def test_openai_preflight_failure_stops_before_analyzer(self):
        _, job = server._new_job(OWNER)
        path = self.root / "invalid.mp4"
        path.write_bytes(b"not a video")
        provider = SimpleNamespace(
            cfg=SimpleNamespace(type="openai_native"),
            preflight=Mock(side_effect=ValueError("invalid local video")),
        )
        with patch.object(server.analyze_mod, "analyze") as analyze:
            await server._run_analysis_job(
                job, media_path=str(path), size=11, species="cat", context="", mode="native",
                provider=provider, knowledge={}, max_frames=8, text="", client_id=OWNER,
            )
        analyze.assert_not_called()
        provider.preflight.assert_called_once_with(str(path), "native")
        self.assertEqual(job["events"][-1][0], "error")
        self.assertEqual(job["events"][-1][1]["error_type"], "media_validation_error")
        self.assertIn("15 秒", job["events"][-1][1]["message"])
        self.assertNotIn("invalid local video", job["events"][-1][1]["message"])
        self.assertFalse(path.exists())

    async def test_http_upload_real_media_validation_no_model(self):
        """Multipart -> real decoder -> fake analysis -> SSE; no browser claim."""
        from PIL import Image
        from pipeline.openai_provider import OpenAINativeProvider
        cfg = server.providers_mod.ProviderConfig(type="openai_native", model="gpt-6.1-sol",
            base_url="https://api.openai.com/v1", audio_model="gpt-audio-1.5")
        jpeg = io.BytesIO()
        Image.new("RGB", (64, 64), "gray").save(jpeg, "JPEG")
        cases = [("valid.jpg", jpeg.getvalue(), "quick", True),
                 ("corrupt.mp4", b"not a video", "native", False)]
        for name, duration, fps, valid in (("valid", 2, 30, True),
                                         ("long", 16, 30, False), ("fast", 2, 60, False)):
            path = self.root / (name + "-source.mp4")
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                            f"color=c=gray:s=64x64:r={fps}", "-t", str(duration),
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
                           check=True, capture_output=True, timeout=15)
            cases.append((name + ".mp4", path.read_bytes(), "native", valid))
        for number, (filename, content, mode, valid) in enumerate(cases):
            with self.subTest(filename=filename), \
                 patch.object(server, "_provider", side_effect=lambda: OpenAINativeProvider(cfg)), \
                 patch.object(server.providers_mod, "_post_json", side_effect=AssertionError("NO REAL API")) as network, \
                 patch.object(server.analyze_mod, "analyze", return_value={
                     "result": {"insufficient_evidence": True, "candidates": [], "translation": "Offline fixture"},
                     "model": "offline-fixture", "tokens_in": None, "tokens_out": None,
                 }) as analyzer:
                response = await self.client.post("/analyze", headers={"Idempotency-Key": f"offline-http-upload-{number}"},
                    data={"client_id": OWNER, "species": "cat", "mode": mode, "context": "unavailable"},
                    files={"file": (filename, content, "image/jpeg" if mode == "quick" else "video/mp4")})
                self.assertEqual(response.status_code, 200, response.text)
                if server._TASKS:
                    await asyncio.gather(*list(server._TASKS))
                jid = response.json()["job_id"]
                event, payload = server._JOBS[jid]["events"][-1]
                self.assertEqual(event, "done" if valid else "error")
                self.assertEqual(analyzer.call_count, int(valid))
                network.assert_not_called()
                stream = await self.client.get(f"/analyze/stream/{jid}")
                self.assertIn(f"event: {event}", stream.text)
                self.assertFalse(Path(server._JOBS[jid]["upload_path"]).exists())
                if not valid:
                    self.assertEqual(payload["error_type"], "media_validation_error")

    async def test_openai_frames_uses_adapter_video_route_and_records_requested_mode(self):
        _, job = server._new_job(OWNER)
        path = self.root / "offline.mp4"
        path.write_bytes(b"offline fixture")
        calls = []
        provider = SimpleNamespace(
            cfg=SimpleNamespace(type="openai_native"),
            preflight=Mock(side_effect=lambda *args: calls.append("preflight")),
            last_model="actual-model",
        )
        def fake_analyze(*args, **kwargs):
            calls.append("analyze")
            self.assertEqual(args[3], "native")
            return {"result": {"candidates": [], "translation": "offline"}, "model": "fake"}
        with patch.object(server.analyze_mod, "analyze", side_effect=fake_analyze):
            await server._run_analysis_job(
                job, media_path=str(path), size=15, species="cat", context="", mode="frames",
                provider=provider, knowledge={}, max_frames=8, text="", client_id=OWNER,
            )
        self.assertEqual(calls, ["preflight", "analyze"])
        event, payload = job["events"][-1]
        self.assertEqual(event, "done")
        self.assertEqual(payload["meta"]["mode"], "frames")
        self.assertEqual(payload["meta"]["execution_mode"], "native")
        self.assertEqual(payload["meta"]["execution_path"], "openai_sampled_video")
        record = store.get_record(payload["record_id"], client_id=OWNER)
        self.assertEqual(record["mode"], "frames")
        self.assertEqual(record["result"]["_meta"]["execution_path"], "openai_sampled_video")

    async def test_bounds_reject_before_analysis(self):
        for updates in ({"max_frames": "10000"}, {"context": "x" * 2001}):
            response = await self.client.post("/analyze", data={
                "client_id": OWNER, "species": "cat", "mode": "text", "text": "cat meows", **updates,
            })
            self.assertIn(response.status_code, (413, 422))

    async def test_yellow_asset_is_versionable_and_dynamic_data_is_not_cached(self):
        asset = await self.client.get('/assets/mark-yellow.png?v=20261106-yellow-v1')
        self.assertEqual(asset.status_code, 200)
        self.assertIn('image/png', asset.headers['content-type'])
        self.assertEqual(asset.headers['cache-control'], 'public, max-age=3600')
        history = await self.client.get('/history')
        self.assertIn('no-store', history.headers['cache-control'])

    async def test_health_does_not_confuse_liveness_with_analysis_readiness(self):
        with patch.object(server, '_STATE', {}):
            result = (await self.client.get('/healthz')).json()
        self.assertTrue(result['ok'])
        self.assertFalse(result['analysis_ready'])
        self.assertEqual(result['release_id'], server.RELEASE_ID)


if __name__ == "__main__":
    unittest.main()
