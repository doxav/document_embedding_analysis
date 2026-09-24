"""CPU tests of fixed-split retrieval; the native retrieval backend is mocked."""
from __future__ import annotations

import asyncio
import json
import sys
import types
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.promptfoo_openwebui_eval.posttrain.core import StopRun
from scripts.promptfoo_openwebui_eval.posttrain.split_retrieval import render_tool

MARKER = 'vsrc_' + 'a' * 20
SOURCES = {'file-train': MARKER}


def tool_instance(monkeypatch: pytest.MonkeyPatch, raw: str, **options: int) -> tuple[Any, AsyncMock]:
    """Load the generated source with only its native retrieval dependency mocked."""
    backend = types.ModuleType('open_webui.tools.builtin')
    backend.query_knowledge_files = AsyncMock(return_value=raw)
    monkeypatch.setitem(sys.modules, 'open_webui.tools.builtin', backend)
    namespace: dict[str, Any] = {}
    exec(compile(render_tool('kb-train', SOURCES, **options), '<split-retrieval>', 'exec'), namespace)
    return namespace['Tools'](), backend.query_knowledge_files


def chunk(**overrides: Any) -> dict[str, Any]:
    """Build one native result tied to a real uploaded-page mapping fixture."""
    return {'content': 'Les recettes sont de 42 EUR.', 'source': MARKER + '.md',
            'file_id': 'file-train', 'distance': 0.1, **overrides}


def search(tool: Any, query: Any = 'recettes', count: Any = 3, **context: Any) -> str:
    """Run one tool call with an authenticated context without a live Open WebUI app."""
    return asyncio.run(tool.search(query, count, __request__=context.get('request', object()),
                                   __user__=context.get('user', {'id': 'test-user', 'role': 'admin'})))


def test_content_and_fixed_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    """An admin receives only the fixed split and intact native evidence."""
    evidence = chunk(content='Source-ID: ' + MARKER + '\nContenu original FR/EN.')
    tool, backend = tool_instance(monkeypatch, json.dumps([evidence]))
    result = json.loads(search(tool))
    assert result == [{**evidence, 'source_marker': MARKER}]
    kwargs = backend.await_args.kwargs
    assert kwargs['knowledge_ids'] == ['kb-train']
    assert kwargs['__model_knowledge__'] == [{'type': 'collection', 'id': 'kb-train'}]
    assert kwargs['query'] == 'recettes'
    assert kwargs['count'] == 3


@pytest.mark.parametrize('query,count', [('', 3), ('   ', 3), (None, 3), ('\ud800', 3), ('é' * 1025, 3),
                                        ('query', True), ('query', '3'), ('query', 0), ('query', 4)])
def test_invalid_inputs_do_not_retrieve(monkeypatch: pytest.MonkeyPatch, query: Any, count: Any) -> None:
    """Invalid requests fail before any embedding or database call."""
    tool, backend = tool_instance(monkeypatch, json.dumps([chunk()]))
    with pytest.raises(ValueError):
        search(tool, query, count)
    backend.assert_not_awaited()


@pytest.mark.parametrize('context', [{'request': None}, {'user': {}}, {'user': None}])
def test_missing_context(monkeypatch: pytest.MonkeyPatch, context: dict[str, Any]) -> None:
    """A missing authenticated request cannot invoke native retrieval."""
    tool, backend = tool_instance(monkeypatch, json.dumps([chunk()]))
    with pytest.raises(ValueError, match='context'):
        search(tool, **context)
    backend.assert_not_awaited()


@pytest.mark.parametrize('raw', [
    'not json', '[]', '{"error":"private provider error"}', '[null]',
    json.dumps([chunk(file_id='file-dev')]), json.dumps([chunk(file_id=None)]),
    json.dumps([chunk(content='')]), json.dumps([chunk(source='')]),
    json.dumps([chunk(content='Source-ID: vsrc_' + 'b' * 20)]),
    json.dumps([chunk(distance=float('nan'))]), json.dumps([chunk(distance=True)]),
    '[{"file_id":"file-dev","file_id":"file-train"}]', json.dumps([chunk()] * 4),
])
def test_invalid_native_output(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    """Malformed, empty and cross-split observations are technical failures."""
    tool, backend = tool_instance(monkeypatch, raw)
    with pytest.raises(RuntimeError):
        search(tool)
    backend.assert_awaited_once()


def test_no_content_invented_for_marker(monkeypatch: pytest.MonkeyPatch) -> None:
    """A chunk without a header gets only provenance metadata, with content unchanged."""
    evidence = chunk(content='Original later page chunk.')
    tool, _ = tool_instance(monkeypatch, json.dumps([evidence]))
    result = json.loads(search(tool))[0]
    assert result['source_marker'] == MARKER
    assert result['content'] == evidence['content']


@pytest.mark.parametrize('limit_kind', ['raw', 'enriched'])
def test_output_budget_without_truncation(monkeypatch: pytest.MonkeyPatch, limit_kind: str) -> None:
    """Both raw and provenance-enriched response overflow stop the episode."""
    raw = json.dumps([chunk()], ensure_ascii=False, separators=(',', ':'))
    max_chars = 20 if limit_kind == 'raw' else len(raw)
    tool, backend = tool_instance(monkeypatch, raw, max_output_chars=max_chars)
    with pytest.raises(RuntimeError, match='output budget'):
        search(tool)
    backend.assert_awaited_once()


def test_scope_cannot_be_supplied_by_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """The public callable has no knowledge or file override parameter."""
    tool, backend = tool_instance(monkeypatch, json.dumps([chunk()]))
    with pytest.raises(TypeError):
        asyncio.run(tool.search('recettes', knowledge_ids=['kb-dev']))
    backend.assert_not_awaited()


def test_native_failure_is_sanitized_and_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """A technical backend error never leaks provider details or triggers a retry."""
    tool, backend = tool_instance(monkeypatch, '')
    backend.side_effect = RuntimeError('private-token-or-system-detail')
    with pytest.raises(RuntimeError, match='Native retrieval failed') as error:
        search(tool)
    assert 'private-token' not in str(error.value)
    backend.assert_awaited_once()


@pytest.mark.parametrize('knowledge,sources,options', [
    ('bad/id', SOURCES, {}), ('kb', {}, {}), ('kb', {'file': 'wrong'}, {}),
    ('kb', {'a': MARKER, 'b': MARKER}, {}), ('kb', SOURCES, {'count': True}),
    ('kb', SOURCES, {'count': 0}), ('kb', SOURCES, {'max_output_chars': 0}),
])
def test_invalid_render_configuration(knowledge: str, sources: Mapping[str, str], options: dict[str, int]) -> None:
    """Provisioning cannot render an ambiguous scope or an invalid budget."""
    with pytest.raises(StopRun):
        render_tool(knowledge, sources, **options)


def reader_instance(monkeypatch: pytest.MonkeyPatch, raw: Any,
                    max_source_chars: int = 12000) -> tuple[Any, AsyncMock]:
    """Load the optional reader while mocking only the native file transport."""
    tool, _ = tool_instance(monkeypatch, json.dumps([chunk()]), max_source_chars=max_source_chars)
    backend = sys.modules['open_webui.tools.builtin']
    backend.view_knowledge_file = AsyncMock(return_value=raw)
    return tool, backend.view_knowledge_file


def read_source(tool: Any, marker: Any = MARKER, **context: Any) -> str:
    """Invoke the reader with authenticated test context."""
    return asyncio.run(tool.read_source(marker, __request__=context.get('request', object()),
                                        __user__=context.get('user', {'id': 'test-user'})))


def source_page(**overrides: Any) -> dict[str, Any]:
    """Build the native full-file response independently of search chunk format."""
    return {'id': 'file-train', 'filename': MARKER + '.md',
            'content': 'Source-ID: ' + MARKER + '\nFull original source; no clipping.', **overrides}


def test_reader_intact_native_page(monkeypatch: pytest.MonkeyPatch) -> None:
    """The reader resolves an allowed marker and preserves the entire native page."""
    page = source_page()
    tool, backend = reader_instance(monkeypatch, json.dumps(page))
    evidence = json.loads(read_source(tool))
    assert evidence == [{'content': page['content'], 'source': page['filename'],
                         'file_id': page['id'], 'source_marker': MARKER}]
    backend.assert_awaited_once()
    assert backend.await_args.kwargs['file_id'] == 'file-train'
    assert backend.await_args.kwargs['__user__'] == {'id': 'test-user'}


def test_reader_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Existing profiles expose exactly the original search capability."""
    tool, _ = tool_instance(monkeypatch, json.dumps([chunk()]))
    assert not hasattr(tool, 'read_source')


@pytest.mark.parametrize('marker,context', [(None, {}), ('vsrc_' + 'b' * 20, {}),
    ('file-train', {}), (MARKER, {'request': None}), (MARKER, {'user': {}})])
def test_reader_invalid_input_never_reads(monkeypatch: pytest.MonkeyPatch, marker: Any,
                                         context: dict[str, Any]) -> None:
    """Foreign markers and unauthenticated inputs cannot read any native file."""
    tool, backend = reader_instance(monkeypatch, json.dumps(source_page()))
    with pytest.raises(ValueError):
        read_source(tool, marker, **context)
    backend.assert_not_awaited()


@pytest.mark.parametrize('raw', [None, '[]', '{', '{"error":"private detail"}',
    json.dumps(source_page(id='file-dev')), json.dumps(source_page(content='')),
    json.dumps(source_page(filename=None)), json.dumps(source_page(content='Source-ID: vsrc_' + 'b' * 20)),
    '{"id":"file-dev","id":"file-train"}', '{"id":"file-train","distance":NaN}',
    json.dumps(source_page(error='private detail'))])
def test_reader_invalid_native_response(monkeypatch: pytest.MonkeyPatch, raw: Any) -> None:
    """Native errors, ambiguous JSON and inconsistent provenance never become evidence."""
    tool, backend = reader_instance(monkeypatch, raw)
    with pytest.raises(RuntimeError) as error:
        read_source(tool)
    assert 'private detail' not in str(error.value)
    backend.assert_awaited_once()


def test_reader_failure_no_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Transport failures are sanitized and are not retried."""
    tool, backend = reader_instance(monkeypatch, '')
    backend.side_effect = RuntimeError('private detail')
    with pytest.raises(RuntimeError, match='Native source read failed'):
        read_source(tool)
    backend.assert_awaited_once()


@pytest.mark.parametrize('limit_kind', ['raw', 'enriched'])
def test_reader_budget_no_truncation(monkeypatch: pytest.MonkeyPatch, limit_kind: str) -> None:
    """Both native and enriched page sizes are bounded without dropping source text."""
    raw = json.dumps(source_page(), separators=(',', ':'))
    tool, backend = reader_instance(monkeypatch, raw, max_source_chars=10 if limit_kind == 'raw' else len(raw))
    with pytest.raises(RuntimeError, match='output budget'):
        read_source(tool)
    backend.assert_awaited_once()


@pytest.mark.parametrize('limit', [True, 0, -1, 100001, '12000', 1.5])
def test_reader_configuration_limit(limit: Any) -> None:
    """Reader limits must be explicit positive bounded integers."""
    with pytest.raises(StopRun, match='source reader'):
        render_tool('kb-train', SOURCES, max_source_chars=limit)
