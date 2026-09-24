"""Private Socket.IO transport for the unchanged historical Open WebUI bridge."""
from __future__ import annotations

import asyncio
import contextvars
import copy
import fcntl
import hmac
import importlib.util
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import requests
from fastapi import FastAPI, HTTPException, Request

from .core import StopRun, read, require, safe_id, write
from .integration import audit, bundle, upstream, validate_native_bridge_urls
from .inference_limits import native_episode_seconds


def _private_module(relative: str) -> ModuleType:
    """Load a fresh upstream module without changing its files or shared module globals."""
    name = '_sa_native_' + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(name, bundle() / relative)
    require(spec is not None and spec.loader is not None, 'Native adapter module unavailable')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _final_text(message: dict[str, Any], model: str) -> str:
    """Accept only the final completed assistant item, never reasoning or tool output."""
    require(message.get('model') == model and not message.get('error'), 'Stored assistant identity or error mismatch')
    output = message.get('output')
    require(isinstance(output, list) and output, 'Missing stored native output')
    last = output[-1]
    require(isinstance(last, dict) and last.get('type') == 'message' and
            last.get('role') == 'assistant' and last.get('status') == 'completed',
            'No completed final assistant message')
    parts = last.get('content')
    require(isinstance(parts, list) and parts and all(
        isinstance(part, dict) and part.get('type') == 'output_text' and isinstance(part.get('text'), str)
        for part in parts), 'Invalid final assistant content')
    answer = '\n'.join(part['text'] for part in parts if part['text']).strip()
    require(answer, 'Empty final assistant message')
    return answer


class NativeState:
    """Keep one experiment transport active and retain unresolved task identities on disk."""

    def __init__(self, root: Path, base_url: str, socket_factory: Callable[..., Any]) -> None:
        """Lock the private state directory so a second worker cannot allocate a session."""
        self.root, self.base_url, self.socket_factory = root, base_url, socket_factory
        root.mkdir(parents=True, exist_ok=True)
        root.chmod(0o700)
        self.process_lock = (root / '.worker.lock').open('a')
        fcntl.flock(self.process_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.lock = threading.Lock()
        self.path = root / 'latest.json'
        self.state = read(self.path) if self.path.exists() else {'status': 'idle', 'safe_to_end': True}

    def update(self, **fields: Any) -> None:
        """Persist bounded metadata without prompts, answers, credentials or reasoning."""
        self.state.update(fields)
        write(self.path, self.state)

    def summary(self) -> dict[str, Any]:
        """Expose only the durable transport state needed before recorder closure."""
        state = copy.deepcopy(self.state)
        safe = not self.lock.locked() and state.get('safe_to_end') is True
        try:
            timeout = native_episode_seconds()
        except StopRun:
            timeout = None
        return {**state, 'transport': 'socketio-native-v1', 'safe_to_end': safe,
                'episode_timeout_seconds': timeout, 'inference_limits_valid': timeout is not None,
                'can_begin': safe and timeout is not None and state.get('status') != 'reconciliation_required'}

    async def run(self, client: Any, delegate: Callable[..., dict[str, Any]],
                  kwargs: dict[str, Any], context: contextvars.ContextVar[Any]) -> dict[str, Any]:
        """Authenticate a real socket, run the original client, and preserve ambiguity."""
        require(self.lock.acquire(blocking=False), 'Another native episode is active')
        session = None
        token = None
        started = False
        try:
            require(self.state.get('safe_to_end') is True and self.state.get('status') != 'reconciliation_required',
                    'Reconcile the prior native operation before another episode')
            require(client.base_url == self.base_url, 'Native test instance changed')
            require(kwargs['model'].startswith('sa-') and kwargs['tool_ids'] and
                    all(tid.startswith(('sa-', 'sa_')) for tid in kwargs['tool_ids']),
                    'Native adapter requires isolated sa-* model and tools')
            require(not (kwargs.get('extra_payload') or {}).get('session_id'), 'Session IDs are assigned by Socket.IO')
            self.state = {'status': 'authenticating', 'safe_to_end': False,
                          'model': kwargs['model'], 'base_url': self.base_url}
            started = True
            self.update()
            session = NativeSession(self, client)
            await session.connect()
            token = context.set(session)
            result = await asyncio.to_thread(delegate, **kwargs)
            raw = result.get('raw') or {}
            require(raw.get('chat_id') == self.state.get('chat_id'), 'Returned native chat identity changed')
            require(result.get('output') == _final_text(raw.get('assistant_message') or {}, kwargs['model']),
                    'Returned output is not the final native assistant answer')
            self.update(status='completed', safe_to_end=True)
            return result
        except BaseException as exc:
            if session is not None:
                safe = not session.post_started or session.inactive
                ambiguous_creation = session.creation_started and not session.creation_confirmed
                self.update(status='failed' if safe and not ambiguous_creation else 'reconciliation_required',
                            safe_to_end=safe, creation_unconfirmed=ambiguous_creation,
                            error=str(exc) if isinstance(exc, StopRun) else type(exc).__name__)
            elif started:
                self.update(status='failed', safe_to_end=True, error=type(exc).__name__)
            raise StopRun('Native episode failed; inspect the private native-state receipt') from None
        finally:
            if token is not None:
                context.reset(token)
            try:
                if session is not None:
                    await session.close()
            finally:
                self.lock.release()


class NativeSession:
    """Join one authenticated socket and wait for both exact completion and task cleanup."""

    def __init__(self, state: NativeState, client: Any) -> None:
        """Set a finite episode deadline before any socket or HTTP request."""
        self.owner, self.client = state, client
        limit = native_episode_seconds()
        require(client.timeout > 0, 'Invalid native episode timeout')
        self.deadline = time.monotonic() + min(limit, client.timeout)
        self.socket = state.socket_factory(reconnection=False, logger=False, engineio_logger=False)
        self.heartbeat: asyncio.Task[Any] | None = None
        self.done = False
        self.event_error = False
        self.socket_error = False
        self.post_started = False
        self.creation_started = False
        self.creation_confirmed = False
        self.inactive = False
        self.sid = ''
        self.user_id = ''

    def timeout(self) -> float:
        """Bound every HTTP operation by the remaining episode time."""
        remaining = self.deadline - time.monotonic()
        require(remaining > 0, 'Native episode timeout; task reconciliation required')
        return min(15, remaining)

    def get(self, path: str) -> dict[str, Any]:
        """Read a native endpoint once without exposing response bodies on HTTP errors."""
        with requests.get(self.client.base_url + path, headers=self.client._headers(False),
                          timeout=self.timeout(), allow_redirects=False) as response:
            require(response.status_code == 200, 'Native read failed; task reconciliation required')
            value = response.json()
        require(isinstance(value, dict), 'Invalid native response schema')
        return value

    async def connect(self) -> None:
        """Check REST identity and the authenticated Socket.IO user-join acknowledgment."""
        user = await asyncio.to_thread(self.get, '/api/v1/auths/')
        require(isinstance(user.get('id'), str) and user['id'], 'Native authentication identity missing')
        self.user_id = user['id']
        self.socket.on('events', self.event)
        await self.socket.connect(self.client.base_url, socketio_path='ws/socket.io',
                                  transports=['websocket'], auth={'token': self.client.api_key},
                                  wait_timeout=self.timeout())
        ack = await self.socket.call('user-join', {'auth': {'token': self.client.api_key}},
                                     timeout=self.timeout())
        require(isinstance(ack, dict) and ack.get('id') == self.user_id, 'Socket.IO user identity mismatch')
        self.sid = self.socket.get_sid('/')
        require(self.socket.connected and isinstance(self.sid, str) and self.sid, 'Socket.IO session missing')
        self.owner.update(status='connected', user_id=self.user_id, session_id=self.sid)
        self.heartbeat = asyncio.create_task(self.keep_alive())

    async def keep_alive(self) -> None:
        """Maintain the authenticated session without retrying a disconnected transport."""
        try:
            while self.socket.connected:
                await self.socket.emit('heartbeat', {})
                await asyncio.sleep(20)
        except Exception:
            self.socket_error = True

    async def event(self, envelope: Any) -> None:
        """Observe only completion or failure events for the exact active chat/message."""
        state = self.owner.state
        if not isinstance(envelope, dict) or envelope.get('chat_id') != state.get('chat_id') or \
                envelope.get('message_id') != state.get('message_id'):
            return
        event = envelope.get('data') or {}
        if not isinstance(event, dict):
            return
        data = event.get('data') or {}
        if event.get('type') in {'chat:tasks:cancel', 'chat:message:error'}:
            self.event_error = True
        if event.get('type') == 'chat:completion' and isinstance(data, dict):
            self.event_error = self.event_error or bool(data.get('error'))
            self.done = self.done or data.get('done') is True

    def wait(self) -> None:
        """Wait through early final events until the owned chat task is actually inactive."""
        state = self.owner.state
        while True:
            self.timeout()
            self.inactive = False
            self.owned_chat()
            tasks = self.get('/api/tasks/chat/' + state['chat_id']).get('task_ids')
            require(isinstance(tasks, list) and all(isinstance(item, str) for item in tasks) and
                    set(tasks) <= {state['task_id']}, 'Unexpected native chat task identity')
            self.inactive = not tasks
            if self.inactive:
                self.owner.update(task_inactive=True)
                require(not self.event_error, 'Native task reported an error or cancellation')
                if self.done:
                    chat = self.owned_chat()
                    messages = ((chat.get('chat') or {}).get('history') or {}).get('messages') or {}
                    _final_text(messages.get(state['message_id']) or {}, state['model'])
                    return
            require(self.socket.connected and not self.socket_error, 'Native socket disconnected')
            time.sleep(min(.25, self.timeout()))

    def owned_chat(self) -> dict[str, Any]:
        """Verify ownership because an empty task list also represents missing foreign chats."""
        chat_id = self.owner.state['chat_id']
        chat = self.get('/api/v1/chats/' + chat_id)
        require(chat.get('id') == chat_id and chat.get('user_id') == self.user_id,
                'Native chat identity or ownership changed')
        return chat

    async def close(self) -> None:
        """Stop only this socket heartbeat; never stop a remote task automatically."""
        if self.heartbeat is not None:
            self.heartbeat.cancel()
            await asyncio.gather(self.heartbeat, return_exceptions=True)
        try:
            await asyncio.wait_for(self.socket.disconnect(), timeout=10)
        except Exception:
            self.owner.update(socket_disconnect_unconfirmed=True)


class NativeRequests:
    """Intercept only the historical client module's transport, scoped to one call."""

    def __init__(self, context: contextvars.ContextVar[Any], base_url: str) -> None:
        """Keep the real requests module untouched for other clients and production code."""
        self.context = context
        self.base_url = base_url

    def __getattr__(self, name: str) -> Any:
        """Delegate unchanged read operations and request helpers."""
        return getattr(requests, name)

    def post(self, url: str, **kwargs: Any) -> requests.Response:
        """Send each POST once, waiting for native async completion before returning it."""
        session = self.context.get()
        if session is None and url == self.base_url + '/api/v1/files/?process=true&process_in_background=false':
            # The upstream bridge uploads attachments before entering its chat
            # session. Forward the original stream once; deepcopy cannot copy it.
            files = kwargs.get('files')
            require(isinstance(files, dict) and set(files) == {'file'}
                    and callable(getattr(files['file'], 'read', None))
                    and 'json' not in kwargs and 'data' not in kwargs,
                    'Invalid native attachment upload')
            return requests.post(url, **{**kwargs, 'allow_redirects': False})
        require(session is not None, 'Native POST outside the isolated server-tools session')
        state = session.owner
        kwargs = copy.deepcopy(kwargs)
        kwargs['timeout'] = session.timeout()
        kwargs['allow_redirects'] = False
        if url == session.client.base_url + '/api/v1/chats/new':
            require(not state.state.get('chat_id'), 'Native chat already created; no POST retry')
            title = 'sa-native-' + uuid.uuid4().hex
            kwargs['json']['chat']['title'] = title
            kwargs['json']['chat']['tags'] = ['sa-native-evaluation']
            session.creation_started = True
            state.update(status='creating_chat', creation_name=title)
            response = requests.post(url, **kwargs)
            try:
                require(response.status_code == 200, 'Native chat creation failed; no POST retry')
                chat = response.json()
                chat_id = safe_id(chat.get('id', ''))
                state.update(chat_id=chat_id)
                require(chat.get('user_id') == session.user_id, 'Created native chat ownership mismatch')
                session.creation_confirmed = True
                state.update(status='chat_created')
            finally:
                response.close()
            return response
        require(url == session.client.base_url + '/api/chat/completions', 'Unapproved native POST endpoint')
        require(not session.post_started, 'Native completion already submitted; no POST retry')
        payload = kwargs['json']
        require(payload.get('chat_id') == state.state.get('chat_id') and
                payload.get('model') == state.state['model'], 'Native completion identity mismatch')
        message_id = safe_id(payload.get('id', ''))
        payload['session_id'] = session.sid
        payload['background_tasks'] = {}
        state.update(message_id=message_id, status='submitting', safe_to_end=False)
        session.post_started = True
        response = requests.post(url, **kwargs)
        try:
            require(response.status_code == 200, 'Native completion HTTP error; reconcile without retry')
            ack = response.json()
            require(isinstance(ack, dict) and ack.get('status') is True, 'Invalid native async acknowledgment')
            task_id = safe_id(ack.get('task_id', ''))
            state.update(task_id=task_id, status='running')
            session.wait()
            state.update(status='task_completed', safe_to_end=True)
        except BaseException:
            response.close()
            raise
        return response


def app_factory() -> FastAPI:
    """Expose the upstream bridge with explicit private transport and durable task state."""
    require(audit()['ok'], 'Upstream hashes changed before native bridge loading')
    base = os.environ.get('SA_NATIVE_BASE_URL', '').rstrip('/')
    require(base and base == os.environ.get('OPENWEBUI_BASE_URL', '').rstrip('/') and
            os.environ.get('SA_NATIVE_INSTANCE_REVIEWED') == 'true', 'Reviewed native test instance required')
    validate_native_bridge_urls(base+'/generate',base+'/strategy-a/native-state')
    require(os.environ.get('GENERATION_BACKEND', 'openwebui') == 'openwebui', 'Native bridge requires Open WebUI')
    for key in ('SUMMARIZER_MODEL_ID', 'ALGORITHM', 'TARGET_LENGTH', 'STRUCTURE'):
        require(not os.environ.get('OPENWEBUI_DEFAULT_' + key), 'Unrelated summarizer defaults enabled')
    secret = os.environ.get('SA_ADMIN_TOKEN', '')
    require(len(secret) >= 24, 'Native state admin token required')
    import socketio
    state = NativeState(Path(os.environ['SA_NATIVE_STATE_DIR']), base, socketio.AsyncClient)
    upstream('lib/bundle_common.py')
    client_module = _private_module('lib/openwebui_client.py')
    bridge = _private_module('api/openwebui_bridge.py')
    context: contextvars.ContextVar[Any] = contextvars.ContextVar('sa_native_session', default=None)
    client_module.requests = NativeRequests(context, base)

    class NativeClient(client_module.OpenWebUIClient):
        """Change only the server-tools transport of this isolated upstream module."""

        def _chat_with_server_tools(self, *, model: str, messages: list[dict[str, Any]],
                                    user_prompt: str, files_payload: list[dict[str, str]] | None,
                                    tool_ids: list[str], extra_payload: dict[str, Any] | None,
                                    include_trace: bool) -> dict[str, Any]:
            """Reuse every original request format and trace field under native waiting."""
            kwargs = {'model': model, 'messages': messages, 'user_prompt': user_prompt,
                      'files_payload': files_payload, 'tool_ids': tool_ids,
                      'extra_payload': extra_payload, 'include_trace': include_trace}
            return asyncio.run(state.run(self, super()._chat_with_server_tools, kwargs, context))

    bridge.OpenWebUIClient = NativeClient
    app = bridge.app
    app.state.native_state = state

    @app.get('/strategy-a/native-state')
    def native_state(request: Request) -> dict[str, Any]:
        """Return targeted reconciliation IDs without ending a recorder or remote task."""
        if not hmac.compare_digest(request.headers.get('authorization', ''), 'Bearer ' + secret):
            raise HTTPException(status_code=401, detail='Unauthorized')
        return state.summary()

    return app
