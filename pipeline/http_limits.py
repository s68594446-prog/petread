"""Streaming body cap, including uploads without Content-Length."""
from starlette.exceptions import HTTPException


class BodyLimitMiddleware:
    def __init__(self, app, limit: int):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        received = 0
        async def bounded_receive():
            nonlocal received
            message = await receive()
            if message['type'] == 'http.request':
                received += len(message.get('body', b''))
                if received > self.limit:
                    raise HTTPException(413, '请求超过上传上限')
            return message
        await self.app(scope, bounded_receive, send)
