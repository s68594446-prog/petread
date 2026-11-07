"""No inference/network. Real ASGI + temporary SQLite + fake analyzer."""
import asyncio
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
import app as server
from pipeline import runtime, store


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.patches = [patch.object(store, "DEFAULT_DB_PATH", str(self.root / "db")),
                        patch.object(server, "UPLOAD_DIR", str(self.root)),
                        patch.object(server, "_JOBS", {}), patch.object(server, "_TASKS", set()),
                        patch.object(server, "_provider", return_value=object()),
                        patch.object(server, "load_knowledge", return_value={})]
        for p in self.patches: p.start()
        store.init_db(); runtime.init()
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="https://test.invalid",
                                       headers={"X-PetRead-Client": "web"})
        self.identity = (await self.client.post("/session")).json()

    async def asyncTearDown(self):
        if server._TASKS: await asyncio.gather(*server._TASKS)
        await self.client.aclose()
        for p in reversed(self.patches): p.stop()
        self.temp.cleanup()

    async def accept(self):
        return await self.client.post("/consent", json={"accepted": True, "version": runtime.CONSENT_VERSION})

    async def submit(self, key="test-idempotency-0001"):
        return await self.client.post("/analyze", headers={"Idempotency-Key": key},
                                      data={"species":"cat", "mode":"text", "text":"one meow"})

    async def test_device_id_alone_is_not_authentication(self):
        response = await self.client.get("/history", params={"client_id":"forged-identity-123"})
        self.assertEqual(response.status_code, 403)
        self.client.cookies.clear()
        self.assertEqual((await self.client.get("/history")).status_code, 401)

    async def test_only_token_hash_stored_and_cookie_is_protected(self):
        with store.connect() as conn:
            row = conn.execute("SELECT * FROM device_sessions").fetchone()
            self.assertEqual(row["token_hash"], hashlib.sha256(self.identity["token"].encode()).hexdigest())
            self.assertNotIn(self.identity["token"], str(tuple(row)))
        cookie = list(self.client.cookies.jar)[0]
        self.assertTrue(cookie.secure)
        self.assertIn('HttpOnly', cookie._rest)
        self.assertEqual(cookie._rest['SameSite'], 'strict')

    async def test_consent_required_and_revocation_blocks_next_call(self):
        with patch.object(server.analyze_mod, "analyze") as analyze:
            self.assertEqual((await self.submit()).status_code, 403)
            await self.accept()
            await self.client.post("/consent", json={"accepted":False,"version":runtime.CONSENT_VERSION})
            self.assertEqual((await self.submit()).status_code, 403)
        analyze.assert_not_called()

    async def test_tls_termination_uses_configured_origin_not_client_headers(self):
        for origin, secure in (("", False), ("http://offline.test", False),
                               ("https://public.test", True)):
            with self.subTest(origin=origin), patch.dict("os.environ", {"PETREAD_PUBLIC_ORIGIN": origin}):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=server.app), base_url="http://offline.test",
                    headers={"X-PetRead-Client": "web", "X-Forwarded-Proto": "https",
                             "X-Forwarded-For": "203.0.113.1", "Origin": origin or "http://offline.test"},
                ) as client:
                    response = await client.post("/session")
                    self.assertEqual(response.status_code, 200)
                    cookie = list(client.cookies.jar)[0]
                    self.assertEqual(cookie.secure, secure)
                    self.assertIn("HttpOnly", cookie._rest)
                    self.assertEqual(cookie._rest["SameSite"], "strict")
                    self.assertEqual("strict-transport-security" in response.headers, secure)
                    rejected = await client.post("/session", headers={"Origin": "https://evil.test"})
                    self.assertEqual(rejected.status_code, 403)

    async def test_csrf_and_oversize_headers_rejected(self):
        self.assertEqual((await self.client.post("/consent", headers={"Origin":"https://evil.invalid"}, json={})).status_code,403)
        self.assertEqual((await self.client.post("/consent", headers={"X-PetRead-Client":""}, json={})).status_code,403)
        self.assertEqual((await self.client.post("/analyze", headers={"Content-Length":"999999999"})).status_code,413)

    async def test_success_idempotency_and_durable_result_after_cache_loss(self):
        await self.accept()
        fake = {"result":{"translation":"offline fixture","candidates":[]},"latency_s":0.01}
        with patch.object(server.analyze_mod,"analyze",return_value=fake) as analyze:
            first = await self.submit(); self.assertEqual(first.status_code,200)
            if server._TASKS: await asyncio.gather(*server._TASKS)
            second = await self.submit()
            self.assertEqual(first.json()["job_id"],second.json()["job_id"])
            self.assertEqual(analyze.call_count,1)
        server._JOBS.clear()
        response = await self.client.get('/jobs/'+first.json()['job_id'])
        self.assertTrue(response.json()['done'])
        self.assertEqual(response.json()['events'][-1][0],'done')

    async def test_restart_marks_unknown_without_replay(self):
        cid=self.identity['client_id']; jid,job=server._new_job(cid)
        runtime.create_job(job,'crash-request-0001'); job['durable']=True
        server._job_emit(job,'progress',{'message':'started'})
        with patch.object(server.analyze_mod,'analyze') as analyze: runtime.init(recover=True)
        persisted=runtime.get_job(jid,cid)
        self.assertTrue(persisted['done'])
        self.assertEqual(persisted['events'][-1][1]['error_type'],'interrupted_or_outcome_unknown')
        analyze.assert_not_called()

    async def test_daily_budget_survives_history_deletion(self):
        cid=self.identity['client_id']
        with patch.dict('os.environ',{'PETREAD_DAILY_JOBS':'1'}):
            jid,job=server._new_job(cid);runtime.create_job(job,'limit-request-0001');job['durable']=True
            server._job_finish(job,'done',{})
            runtime.delete_device(cid)
            _, new=runtime.create_session(); _, job2=server._new_job(new['client_id'])
            with self.assertRaises(runtime.AdmissionError): runtime.create_job(job2,'limit-request-0002')

    async def test_delete_rejects_active_then_revokes_all_credentials(self):
        cid=self.identity['client_id']; jid,job=server._new_job(cid)
        runtime.create_job(job,'delete-request-0001');job['durable']=True
        self.assertEqual((await self.client.delete('/me')).status_code,409)
        server._job_finish(job,'done',{})
        self.assertEqual((await self.client.delete('/me')).status_code,200)
        self.assertIsNone(runtime.authenticate(self.identity['token']))
        self.assertIsNone(runtime.get_job(jid,cid))
        self.assertEqual((await self.client.get('/history')).status_code,401)

    async def test_delete_does_not_orphan_uncleaned_media(self):
        cid=self.identity['client_id'];jid,job=server._new_job(cid)
        path=self.root/('b'*32+'.wav');path.write_bytes(b'offline')
        job['upload_path']=str(path);runtime.create_job(job,'uncleaned-upload-001');job['durable']=True
        server._job_finish(job,'done',{})
        with patch.object(server,'_cleanup_media'):
            self.assertEqual((await self.client.delete('/me')).status_code,409)
        self.assertIsNotNone(runtime.get_job(jid,cid));self.assertTrue(path.exists())
        self.assertEqual((await self.client.delete('/me')).status_code,200)
        self.assertFalse(path.exists())

    async def test_export_only_own_data_and_no_server_path(self):
        for cid in (self.identity['client_id'],'unrelated-device-123'):
            store.insert_record(species='cat',mode='text',context=cid,media_path='secret-path',result={},
                                predicted_label=None,confidence=None,tokens_in=None,tokens_out=None,latency_s=None,
                                provider='fake',model='fake',client_id=cid)
        response=await self.client.get('/me/export')
        self.assertNotIn('unrelated-device',response.text);self.assertNotIn('secret-path',response.text)
        self.assertEqual(len(response.json()['records']),1)

    async def test_assets_policy_and_security_headers(self):
        response=await self.client.get('/')
        self.assertEqual(response.headers['x-frame-options'],'DENY')
        self.assertNotIn('unsafe-inline',response.headers['content-security-policy'])
        self.assertEqual((await self.client.get('/assets/app.js')).status_code,200)
        self.assertEqual((await self.client.get('/assets/.env')).status_code,404)
        self.assertEqual((await self.client.get('/privacy')).status_code,200)

    async def test_chunked_request_without_content_length_is_bounded(self):
        async def oversized():
            for _ in range(22): yield b'x' * (1024 * 1024)
        response = await self.client.post('/consent',content=oversized(),headers={'Content-Type':'application/json'})
        self.assertEqual(response.status_code,413)

    async def test_second_worker_is_blocked_before_recovery(self):
        lock=runtime.acquire_worker_lock()
        try:
            with self.assertRaises(RuntimeError): runtime.acquire_worker_lock()
        finally: lock.close()
        runtime.acquire_worker_lock().close()

    async def test_startup_recovers_and_only_cleans_registered_upload(self):
        cid=self.identity['client_id']; jid,job=server._new_job(cid)
        upload=self.root/('a'*32+'.wav'); upload.write_bytes(b'offline')
        unrelated=self.root/'keep-user-asset.wav';unrelated.write_bytes(b'keep')
        job['upload_path']=str(upload);runtime.create_job(job,'startup-request-001')
        from fastapi.testclient import TestClient
        with patch.object(server.providers_mod,'load_config',return_value={'mock':True}), \
             patch.object(server.providers_mod,'build_providers',return_value={}), \
             patch.object(server,'_STATE',{}):
            with TestClient(server.app) as client:
                self.assertEqual(client.get('/healthz').status_code,200)
                self.assertFalse(upload.exists());self.assertTrue(unrelated.exists())
                self.assertTrue(runtime.get_job(jid,cid)['done'])
        self.assertIsNone(runtime.interrupted_uploads()[0] if runtime.interrupted_uploads() else None)

    async def test_attempt_write_ahead_raw_and_unknown_recovery(self):
        cid=self.identity['client_id'];jid,job=server._new_job(cid)
        runtime.create_job(job,'ledger-request-0001');job['durable']=True
        trace=runtime.attempt_observer(jid)
        trace('started',{'model':'fake'})
        rows=runtime.attempts_for_device(cid)
        self.assertEqual(len(rows),1);self.assertIsNone(rows[0]['issued']);self.assertIsNone(rows[0]['raw'])
        trace('received',{'http_body':'offline response'})
        self.assertEqual(runtime.attempts_for_device(cid)[0]['http_body'],'offline response')
        trace('finished',{'status':'success','raw':'{}','usage':None,'latency_s':1.2})
        trace('started',{'model':'fake'})
        runtime.init(recover=True)
        rows=runtime.attempts_for_device(cid)
        self.assertEqual(rows[0]['issued'],1);self.assertIsNone(rows[0]['usage'])
        self.assertEqual(rows[1]['status'],'interrupted_or_outcome_unknown')
        self.assertIsNone(rows[1]['issued']);self.assertIsNone(rows[1]['latency_s'])
        runtime.delete_device(cid)
        with store.connect() as conn: self.assertEqual(conn.execute('SELECT COUNT(*) FROM production_attempts').fetchone()[0],0)


if __name__ == '__main__': unittest.main()
