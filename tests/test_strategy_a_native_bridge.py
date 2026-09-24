"""Exercise the real historical client through a private simulated Socket.IO transport."""
from __future__ import annotations

import asyncio
import copy
import importlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Callable

import pytest
import requests
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.promptfoo_openwebui_eval.posttrain import native_bridge as native


def response(payload: Any, status: int = 200) -> requests.Response:
    """Build a requests response with a cached JSON body and no live connection."""
    result = requests.Response()
    result.status_code = status
    result._content = json.dumps(payload).encode()
    result.encoding = 'utf-8'
    result.headers['content-type'] = 'application/json'
    return result


@pytest.fixture
def bridge(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    """Run the actual upstream bridge/client with deterministic socket and HTTP doubles."""
    state: dict[str, Any] = {'posts': [], 'gets': [], 'clock': 0., 'active_polls': 2,
        'persisted': False, 'event_kind': 'done', 'socket_user_id': 'user-1', 'sid': 'socket-1',
        'owner': 'user-1', 'foreign_task': False, 'creation_error': False, 'post_error': False,
        'empty_final': False, 'socket_lost': False, 'final_gets': 0}

    class Socket:
        """Emulate only the authenticated websocket/event operations used by the adapter."""

        def __init__(self, **kwargs: Any) -> None:
            """Assert that the adapter disables reconnects and credential-bearing logs."""
            assert kwargs == {'reconnection': False, 'logger': False, 'engineio_logger': False}
            self.connected = False
            self.handler: Callable[..., Any] | None = None
            self.loop: asyncio.AbstractEventLoop | None = None
            state['socket'] = self

        def on(self, event: str, handler: Callable[..., Any]) -> None:
            """Register the exact Open WebUI event envelope handler."""
            assert event == 'events'
            self.handler = handler

        async def connect(self, url: str, **kwargs: Any) -> None:
            """Verify websocket transport and the authenticated handshake contract."""
            assert url == 'http://127.0.0.1:18080'
            assert kwargs['socketio_path'] == 'ws/socket.io' and kwargs['transports'] == ['websocket']
            assert kwargs['auth'] == {'token': 'UNIT_TEST_JWT'}
            self.connected = True
            self.loop = asyncio.get_running_loop()

        async def call(self, event: str, data: dict[str, Any], timeout: float) -> dict[str, Any]:
            """Supply an acknowledgment independent of a successful socket connection."""
            assert event == 'user-join' and data == {'auth': {'token': 'UNIT_TEST_JWT'}}
            assert timeout > 0
            return {'id': state['socket_user_id'], 'name': 'Test operator'}

        def get_sid(self, namespace: str) -> str:
            """Return the namespace SID, not the Engine.IO transport ID."""
            assert namespace == '/'
            return state['sid']

        async def emit(self, event: str, data: dict[str, Any]) -> None:
            """Record the heartbeat without extra messages or model calls."""
            assert event == 'heartbeat' and data == {}
            state['heartbeat'] = True

        async def disconnect(self) -> None:
            """Close only this simulated socket."""
            self.connected = False
            state['disconnected'] = True

    def get(url: str, **kwargs: Any) -> requests.Response:
        """Expose owned chat/task snapshots with persistence racing the first read."""
        state['gets'].append(url)
        if url.endswith('/api/v1/files/file-1/process/status'):
            return response({'status': 'completed'})
        if url.endswith('/api/v1/auths/'):
            return response({'id': 'user-1'})
        if url.endswith('/api/tasks/chat/chat-1'):
            if state['foreign_task']:
                return response({'task_ids': ['foreign-task']})
            if state['active_polls'] > 0:
                state['active_polls'] -= 1
                return response({'task_ids': ['task-1']})
            state['persisted'] = True
            return response({'task_ids': []})
        assert url.endswith('/api/v1/chats/chat-1')
        output = []
        if state['persisted']:
            state['final_gets'] += 1
            output = [
                {'type': 'function_call', 'call_id': 'call-1', 'name': 'search', 'arguments': '{}',
                 'status': 'completed'},
                {'type': 'function_call_output', 'call_id': 'call-1', 'status': 'completed',
                 'output': [{'type': 'input_text', 'text': 'Retrieved evidence, not the final answer.'}]},
                {'type': 'message', 'role': 'assistant', 'status': 'completed',
                 'content': [{'type': 'output_text', 'text': '' if state['empty_final'] else 'Final answer.'}]},
            ]
        message = {'model': 'sa-test-train', 'output': output,
                   'usage': {'prompt_tokens': 50, 'completion_tokens': 20, 'total_tokens': 70}}
        return response({'id': 'chat-1', 'user_id': state['owner'], 'chat': {
            'history': {'messages': {state.get('message_id', 'unknown'): message}}}})

    def post(url: str, **kwargs: Any) -> requests.Response:
        """Return async acknowledgment once and emit completion before task cleanup."""
        if url.endswith('/api/v1/files/?process=true&process_in_background=false'):
            assert kwargs['allow_redirects'] is False
            state['attachment_bytes'] = kwargs['files']['file'].read()
            state['posts'].append((url, {'uploaded': True}))
            return response({'id': 'file-1'})
        state['posts'].append((url, copy.deepcopy(kwargs['json'])))
        if url.endswith('/api/v1/chats/new'):
            assert kwargs['json']['chat']['title'].startswith('sa-native-')
            if state['creation_error']:
                raise requests.Timeout('Simulated ambiguous creation')
            return response({'id': 'chat-1', 'user_id': 'user-1'})
        assert url.endswith('/api/chat/completions')
        payload = kwargs['json']
        assert payload['session_id'] == 'socket-1' and payload['background_tasks'] == {}
        state['message_id'] = payload['id']
        if state['post_error']:
            raise requests.Timeout('Simulated ambiguous completion')
        envelope = {'chat_id': 'chat-1', 'message_id': payload['id'],
                    'data': {'type': 'chat:completion', 'data': {'done': True}}}
        if state['event_kind'] == 'other_chat':
            envelope['chat_id'] = 'other-chat'
        elif state['event_kind'] == 'other_message':
            envelope['message_id'] = 'other-message'
        elif state['event_kind'] == 'error':
            envelope['data']['data'] = {'error': {'message': 'Simulated native error'}}
        elif state['event_kind'] == 'cancel':
            envelope['data'] = {'type': 'chat:tasks:cancel'}
        socket = state['socket']
        asyncio.run_coroutine_threadsafe(socket.handler(envelope), socket.loop).result(timeout=2)
        if state['socket_lost']:
            socket.connected = False
        return response({'status': True, 'task_id': 'task-1'})

    def sleep(seconds: float) -> None:
        """Advance only the adapter's fake clock, leaving asyncio's clock unchanged."""
        state['clock'] += seconds

    socket_module = ModuleType('socketio')
    socket_module.AsyncClient = Socket
    monkeypatch.setitem(sys.modules, 'socketio', socket_module)
    monkeypatch.setattr(native.requests, 'get', get)
    monkeypatch.setattr(native.requests, 'post', post)
    monkeypatch.setattr(native, 'time', SimpleNamespace(monotonic=lambda: state['clock'], sleep=sleep))
    for key, value in {'SA_NATIVE_BASE_URL': 'http://127.0.0.1:18080',
        'OPENWEBUI_BASE_URL': 'http://127.0.0.1:18080', 'SA_NATIVE_INSTANCE_REVIEWED': 'true',
        'SA_NATIVE_STATE_DIR': str(tmp_path / 'native'), 'SA_ADMIN_TOKEN': 'A' * 32,
        'OPENWEBUI_API_KEY': 'UNIT_TEST_JWT', 'OPENWEBUI_FILE_CACHE_PATH': str(tmp_path / 'files.json'),
        'SA_NATIVE_MAX_SECONDS': '2', 'GENERATION_BACKEND': 'openwebui'}.items():
        monkeypatch.setenv(key, value)
    for suffix in ('SUMMARIZER_MODEL_ID', 'ALGORITHM', 'TARGET_LENGTH', 'STRUCTURE'):
        monkeypatch.delenv('OPENWEBUI_DEFAULT_' + suffix, raising=False)
    app = native.app_factory()
    state['app'], state['client'] = app, TestClient(app)
    state['payload'] = {'request_prompt': 'What does the evidence say?',
        'openwebui_pipe_model': 'sa-test-train', 'openwebui_tool_ids_json': '["sa_test_search"]',
        'openwebui_include_trace': True}
    return state


def test_native_async_race_waits_for_done_task_cleanup_and_fresh_final(bridge: dict[str, Any]) -> None:
    """An early done event cannot expose a stale stored response or tool-output fallback."""
    result = bridge['client'].post('/generate', json=bridge['payload'])
    assert result.status_code == 200 and result.json()['output'] == 'Final answer.'
    assert len(bridge['posts']) == 2 and bridge['active_polls'] == 0
    assert bridge['final_gets'] >= 2 and bridge['heartbeat'] and bridge['disconnected']
    assert result.json()['trace']['tool_calls'][0]['name'] == 'search'
    summary = bridge['app'].state.native_state.summary()
    assert summary['status'] == 'completed' and summary['safe_to_end'] and summary['can_begin']
    assert summary['task_id'] == 'task-1' and summary['chat_id'] == 'chat-1'
    original = importlib.import_module('lib.openwebui_client')
    assert original.requests is requests and not isinstance(original.requests, native.NativeRequests)


@pytest.mark.parametrize('field,value', [('socket_user_id', 'other-user'), ('sid', '')])
def test_socket_authentication_failure_prevents_posts(bridge: dict[str, Any], field: str, value: str) -> None:
    """A connected socket without the exact authenticated namespace identity is rejected."""
    bridge[field] = value
    result = bridge['client'].post('/generate', json=bridge['payload'])
    assert result.status_code == 500 and not bridge['posts']
    assert bridge['app'].state.native_state.summary()['safe_to_end'] is True


@pytest.mark.parametrize('event_kind', ['other_chat', 'other_message', 'error', 'cancel'])
def test_foreign_or_failed_events_never_become_success(bridge: dict[str, Any], event_kind: str) -> None:
    """Only the exact successful event can finish the owned inactive task."""
    bridge['event_kind'] = event_kind
    result = bridge['client'].post('/generate', json=bridge['payload'])
    assert result.status_code == 500 and len(bridge['posts']) == 2
    assert bridge['app'].state.native_state.summary()['status'] == 'failed'


@pytest.mark.parametrize('fault', ['active_timeout', 'post_error', 'foreign_owner', 'foreign_task', 'socket_lost'])
def test_unconfirmed_task_retains_state_and_blocks_another_post(bridge: dict[str, Any], fault: str) -> None:
    """An unresolved native operation remains identifiable and cannot be paid twice."""
    if fault == 'active_timeout':
        bridge['active_polls'] = 1000
    elif fault == 'foreign_owner':
        bridge['owner'] = 'other-user'
    else:
        bridge[fault] = True
    result = bridge['client'].post('/generate', json=bridge['payload'])
    summary = bridge['app'].state.native_state.summary()
    assert result.status_code == 500 and len(bridge['posts']) == 2
    assert summary['status'] == 'reconciliation_required' and not summary['safe_to_end']
    assert summary['chat_id'] == 'chat-1' and not summary['can_begin']
    bridge['client'].post('/generate', json=bridge['payload'])
    assert len(bridge['posts']) == 2


def test_ambiguous_creation_blocks_new_chat_without_claiming_a_paid_task(bridge: dict[str, Any]) -> None:
    """Keep the unique sa-* creation name even when no chat ID was returned."""
    bridge['creation_error'] = True
    assert bridge['client'].post('/generate', json=bridge['payload']).status_code == 500
    summary = bridge['app'].state.native_state.summary()
    assert summary['creation_unconfirmed'] and summary['creation_name'].startswith('sa-native-')
    assert summary['safe_to_end'] is True and summary['can_begin'] is False
    bridge['client'].post('/generate', json=bridge['payload'])
    assert len(bridge['posts']) == 1


def test_empty_assistant_never_uses_the_tool_fallback(bridge: dict[str, Any]) -> None:
    """Even done plus inactive cannot turn a tool result into an assistant final answer."""
    bridge['empty_final'] = True
    result = bridge['client'].post('/generate', json=bridge['payload'])
    assert result.status_code == 500 and len(bridge['posts']) == 2
    assert bridge['app'].state.native_state.summary()['status'] == 'failed'


def test_constructor_failure_and_admin_state_authentication(
    bridge: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail before any POST on an invalid limit and expose state only to the private admin."""
    monkeypatch.setenv('SA_NATIVE_MAX_SECONDS', '0')
    assert bridge['client'].post('/generate', json=bridge['payload']).status_code == 500
    assert not bridge['posts'] and bridge['app'].state.native_state.summary()['safe_to_end']
    assert bridge['client'].get('/strategy-a/native-state').status_code == 401
    result = bridge['client'].get('/strategy-a/native-state', headers={'Authorization': 'Bearer ' + 'A' * 32})
    assert result.status_code == 200 and result.json()['transport'] == 'socketio-native-v1'


def test_native_bridge_uploads_and_attaches_files_before_chat(
        bridge: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The original multipart stream and resulting file ID reach the native completion and persisted user message."""
    document = tmp_path / 'evidence.md'
    document.write_text('Source-ID: vsrc_0123456789abcdef0123\nOriginal evidence.\n')
    monkeypatch.setenv('OPENWEBUI_INCLUDE_FILES_PAYLOAD', 'true')
    payload = dict(bridge['payload'], source_paths_json=json.dumps([str(document)]))
    result = bridge['client'].post('/generate', json=payload)
    assert result.status_code == 200, result.text
    assert result.json()['file_ids'] == ['file-1']
    assert bridge['attachment_bytes'] == document.read_bytes()
    assert len(bridge['posts']) == 3
    expected = [{'type': 'file', 'id': 'file-1'}]
    assert bridge['posts'][2][1]['user_message']['files'] == expected
    assert bridge['posts'][2][1]['files'] == expected
    assert bridge['app'].state.native_state.summary()['safe_to_end'] is True


@pytest.mark.parametrize('endpoint', ['/api/chat/completions', '/api/v1/tools/create',
    '/api/v1/files/', '/api/v1/files/?process=true&process_in_background=false&extra=1'])
def test_attachment_allowlist_does_not_open_other_posts(endpoint: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the exact private upstream upload endpoint works outside a chat session."""
    import contextvars
    import io
    transport = native.NativeRequests(contextvars.ContextVar('test_native', default=None), 'http://127.0.0.1:18080')
    calls: list[str] = []
    monkeypatch.setattr(native.requests, 'post', lambda url, **kwargs: calls.append(url))
    with pytest.raises(native.StopRun, match='outside the isolated'):
        transport.post('http://127.0.0.1:18080'+endpoint, files={'file': io.BytesIO(b'text')})
    assert not calls


def test_attachment_upload_timeout_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """An ambiguous upload remains one submission and propagates the failure."""
    import contextvars
    import io
    transport = native.NativeRequests(contextvars.ContextVar('test_native', default=None), 'http://127.0.0.1:18080')
    calls: list[str] = []
    def fail(url: str, **kwargs: Any) -> requests.Response:
        """Record a single multipart attempt with an unknown response."""
        calls.append(url)
        raise requests.Timeout('Simulated upload ambiguity')
    monkeypatch.setattr(native.requests, 'post', fail)
    with pytest.raises(requests.Timeout):
        transport.post('http://127.0.0.1:18080/api/v1/files/?process=true&process_in_background=false',
                       files={'file': io.BytesIO(b'text')}, timeout=30)
    assert len(calls) == 1
