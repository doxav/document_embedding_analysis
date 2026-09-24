"""Exercise split-bound provisioning against an in-memory Open WebUI API."""
from __future__ import annotations

import ast
import copy
import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.promptfoo_openwebui_eval.posttrain import integration
from scripts.promptfoo_openwebui_eval.posttrain.core import StopRun, safe_id


class Response:
    """Provide isolated JSON responses for the real integration adapter."""

    def __init__(self, payload: Any, status_code: int = 200) -> None:
        """Keep the fake server state separate from its returned JSON."""
        self.payload = copy.deepcopy(payload)
        self.status_code = status_code

    def json(self) -> Any:
        """Return a fresh deserialized response for each caller."""
        return copy.deepcopy(self.payload)


class Admin:
    """Model only the reviewed read/create routes, rejecting unexpected calls."""

    def __init__(self) -> None:
        """Start with one dedicated blueprint and one visible provider model."""
        self.models: dict[str, Any] = {
            'sa-blueprint': {'id': 'sa-blueprint', 'info': {
                'id': 'sa-blueprint',
                'params': {'system': 'Answer from evidence.', 'temperature': 0.9},
                'meta': {'toolIds': [], 'builtinTools': {'knowledge': True},
                         'capabilities': {'builtin_tools': True, 'file_context': True}, 'knowledge': []},
            }},
            'recorded/teacher': {'id': 'recorded/teacher'},
        }
        self.tools: dict[str, dict[str, Any]] = {}
        self.knowledge: dict[str, dict[str, Any]] = {}
        self.calls: list[tuple[str, str, dict[str, Any] | None]] = []
        self.workspace_reads: list[str] = []
        self.workspace_overrides: dict[str, Response] = {}

    def _request(self, method: str, path: str, **kwargs: Any) -> Response:
        """Simulate creation and readback without any network or external resources."""
        body = copy.deepcopy(kwargs.get('json'))
        self.calls.append((method, path, body))
        if method == 'GET':
            if path == '/api/models':
                models = copy.deepcopy(list(self.models.values()))
                for model in models:
                    if isinstance(model.get('info'), dict):
                        model['info']['params'] = None
                return Response({'data': models})
            if path == '/api/v1/models/model':
                assert set(kwargs) == {'params', 'expected'}
                assert kwargs['expected'] == (200, 404) and set(kwargs['params']) == {'id'}
                model_id = kwargs['params']['id']
                self.workspace_reads.append(model_id)
                if model_id in self.workspace_overrides:
                    return self.workspace_overrides[model_id]
                info = (self.models.get(model_id) or {}).get('info')
                return Response(info, 200 if info is not None else 404)
            if path == '/api/v1/tools/':
                return Response(list(self.tools.values()))
            if path.startswith('/api/v1/tools/id/'):
                tool_id = path.removeprefix('/api/v1/tools/id/').split('/')[0]
                assert tool_id in self.tools
                return Response({} if path.endswith('/valves') else self.tools[tool_id])
        if method == 'POST':
            assert isinstance(body, dict)
            if path == '/api/v1/knowledge/create':
                knowledge_id = f'kb-{len(self.knowledge) + 1}'
                self.knowledge[knowledge_id] = {'id': knowledge_id, **body, 'files': []}
                return Response(self.knowledge[knowledge_id])
            if path.startswith('/api/v1/knowledge/') and path.endswith('/file/add'):
                knowledge_id = path.split('/')[4]
                self.knowledge[knowledge_id]['files'].append(body['file_id'])
                return Response({'status': True})
            if path == '/api/v1/tools/create':
                assert body['id'].isidentifier() and body['id'] == body['id'].lower()
                assert body['id'] not in self.tools
                self.tools[body['id']] = {**body, 'specs': [{'name': 'search'}]}
                return Response(self.tools[body['id']])
            if path == '/api/v1/models/create':
                assert body['id'] not in self.models
                self.models[body['id']] = {'id': body['id'], 'info': body}
                return Response(body)
        raise AssertionError(f'Unexpected fake API operation: {method} {path}')


class Client:
    """Assign file IDs only when the provisioning code actually uploads."""

    def __init__(self) -> None:
        """Track source paths to validate provenance and split boundaries."""
        self.uploads: dict[str, Path] = {}

    def upload_file(self, path: Path) -> str:
        """Record the uploaded page fixture and return a fresh file ID."""
        assert path.is_file()
        file_id = f'file-{len(self.uploads) + 1}'
        self.uploads[file_id] = path
        return file_id


@pytest.fixture
def setup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Admin, Client, list[dict[str, Any]], dict[str, Any]]:
    """Provide real source files and fake transports for all three corpus splits."""
    admin, client = Admin(), Client()

    def clients(cache: Path | None = None) -> tuple[Client, Admin]:
        """Replace only HTTP/upload clients while retaining real adapter logic."""
        return client, admin

    monkeypatch.setattr(integration, 'clients', clients)
    cases: list[dict[str, Any]] = []
    for index, split in enumerate(('train', 'dev', 'test'), 1):
        marker = 'vsrc_' + f'{index:020x}'
        path = tmp_path / f'{marker}.md'
        path.write_text(f'Source-ID: {marker}\nOriginal {split} page.', encoding='utf-8')
        cases.append({'split': split, 'source_paths': [str(path)]})
    spec = {'namespace': 'sa-pilot', 'blueprint': 'sa-blueprint', 'base_model_id': 'recorded/teacher',
            'allowed_tool_ids': [], 'retrieval_mode': 'split_bound', 'isolated_instance_reviewed': True,
            'source_rights_reviewed': True, 'read_only_tools_reviewed': True,
            'retrieval_count': 2, 'max_retrieval_chars': 4000}
    return admin, client, cases, spec


def constants(source: str) -> dict[str, Any]:
    """Read generated scope constants without importing or executing the tool."""
    return {node.targets[0].id: ast.literal_eval(node.value)
            for node in ast.parse(source).body
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}


def test_plan_has_no_mutations(setup: tuple[Admin, Client, list[dict[str, Any]], dict[str, Any]], tmp_path: Path) -> None:
    """A dry run exposes all proposed split IDs without creating files or API objects."""
    admin, client, cases, spec = setup
    before = copy.deepcopy(admin.models)
    output = tmp_path / 'resources.json'
    plan = integration.provision(cases, output, spec)
    assert plan['status'] == 'planned'
    assert plan['created_tool_ids'] == []
    assert plan['tool_ids'] == []
    assert set(plan['splits']) == {'train', 'dev', 'test'}
    for split, entry in plan['splits'].items():
        assert entry['tool_ids'] == [f'sa_pilot_{split}_retrieval']
        assert safe_id(entry['tool_ids'][0]).isidentifier()
        assert entry['file_ids'] == []
    assert all(method == 'GET' for method, _, _ in admin.calls)
    assert not output.exists() and not client.uploads and not admin.tools and not admin.knowledge
    assert admin.models == before


def test_provision_three_fixed_scopes(setup: tuple[Admin, Client, list[dict[str, Any]], dict[str, Any]], tmp_path: Path) -> None:
    """Provisioned aliases expose only their own bounded tool and source mapping."""
    admin, client, cases, spec = setup
    blueprint_before = copy.deepcopy(admin.models['sa-blueprint'])
    output = tmp_path / 'resources.json'
    result = integration.provision(cases, output, spec, apply=True)
    assert result['status'] == 'complete'
    assert json.loads(output.read_text()) == result
    assert len(admin.knowledge) == len(admin.tools) == len(client.uploads) == 3
    assert len(result['created_tool_ids']) == len(result['tool_ids']) == 3
    assert admin.models['sa-blueprint'] == blueprint_before
    assert result['rag_params'] == {'system': 'Answer from evidence.', 'function_calling': 'native'}
    assert result['rag_fingerprint'] == integration.rag_fingerprint(admin, result['tool_ids'], result['rag_params'])
    for split, entry in result['splits'].items():
        assert entry['created'] is True and entry['operation_pending'] is None
        file_id, = entry['file_ids']
        tool_id, = entry['tool_ids']
        assert tool_id == f'sa_pilot_{split}_retrieval'
        assert admin.tools[tool_id]['name'] == f'sa-pilot-{split}-retrieval'
        assert entry['file_sources'] == {file_id: client.uploads[file_id].stem}
        assert client.uploads[file_id] == Path(next(case for case in cases if case['split'] == split)['source_paths'][0])
        scope = constants(admin.tools[tool_id]['content'])
        assert scope['_KNOWLEDGE_ID'] == entry['knowledge_id']
        assert scope['_FILE_SOURCES'] == entry['file_sources']
        assert scope['_MAX_COUNT'] == 2 and scope['_MAX_OUTPUT_CHARS'] == 4000
        assert admin.knowledge[entry['knowledge_id']]['files'] == [file_id]
        alias = admin.models[entry['model_id']]['info']
        assert alias['base_model_id'] == result['base_model_id']
        assert alias['meta']['toolIds'] == [tool_id]
        assert alias['meta']['capabilities']['builtin_tools'] is False
        assert alias['meta']['capabilities']['file_context'] is False
        assert alias['meta']['knowledge'] == [{'id': entry['knowledge_id'], 'name': entry['model_id'], 'type': 'collection'}]
        assert alias['access_grants'] == admin.tools[tool_id]['access_grants'] == []
        assert integration.verify_split_model(admin, result, split) == [tool_id]


def test_masked_inventory_keeps_the_full_blueprint_system(
    setup: tuple[Admin, Client, list[dict[str, Any]], dict[str, Any]], tmp_path: Path,
) -> None:
    """Copy the authoritative system even though the live-style inventory hides params."""
    admin, _, cases, spec = setup
    assert integration.inventory(admin)['sa-blueprint']['info']['params'] is None
    expected = copy.deepcopy(admin.models['sa-blueprint']['info'])
    plan = integration.provision(cases, tmp_path / 'resources.json', spec)
    assert admin.workspace_reads == ['sa-blueprint']
    assert plan['rag_params'] == {'system': expected['params']['system'], 'function_calling': 'native'}
    assert 'temperature' not in plan['rag_params']
    assert plan['blueprint_hash'] == integration.digest(expected)
    assert all(method == 'GET' for method, _, _ in admin.calls)


@pytest.mark.parametrize('response', [
    Response(None, 404), Response(None), Response([]), Response({}),
    Response({'id': 'other-model', 'params': {}, 'meta': {}}),
    Response({'id': 'sa-blueprint', 'meta': {}}),
    Response({'id': 'sa-blueprint', 'params': None, 'meta': {}}),
    Response({'id': 'sa-blueprint', 'params': {}, 'meta': None}),
])
def test_invalid_full_blueprint_prevents_every_write(
    setup: tuple[Admin, Client, list[dict[str, Any]], dict[str, Any]],
    tmp_path: Path, response: Response,
) -> None:
    """A visible inventory entry never substitutes for a missing or malformed raw record."""
    admin, client, cases, spec = setup
    admin.workspace_overrides['sa-blueprint'] = response
    output = tmp_path / 'resources.json'
    with pytest.raises(StopRun, match='Workspace model'):
        integration.provision(cases, output, spec, apply=True)
    assert admin.workspace_reads == ['sa-blueprint']
    assert all(method == 'GET' for method, _, _ in admin.calls)
    assert not output.exists() and not client.uploads and not admin.knowledge and not admin.tools


@pytest.mark.parametrize('fault', ['missing', 'wrong_id', 'masked_params'])
def test_invalid_full_split_record_is_rejected(
    setup: tuple[Admin, Client, list[dict[str, Any]], dict[str, Any]],
    tmp_path: Path, fault: str,
) -> None:
    """Split verification requires the exact full record even while the alias stays visible."""
    admin, _, cases, spec = setup
    result = integration.provision(cases, tmp_path / 'resources.json', spec, apply=True)
    model_id = result['splits']['train']['model_id']
    raw = copy.deepcopy(admin.models[model_id]['info'])
    if fault == 'missing':
        response = Response(None, 404)
    else:
        raw['id' if fault == 'wrong_id' else 'params'] = 'other-model' if fault == 'wrong_id' else None
        response = Response(raw)
    admin.workspace_overrides[model_id] = response
    admin.calls.clear()
    with pytest.raises(StopRun, match='Workspace model'):
        integration.verify_split_model(admin, result, 'train')
    assert all(method == 'GET' for method, _, _ in admin.calls)


def test_bind_reads_complete_source_workspace_models(
    setup: tuple[Admin, Client, list[dict[str, Any]], dict[str, Any]], tmp_path: Path,
) -> None:
    """Student binding reads full source metadata and keeps the manifest's frozen policy."""
    admin, _, cases, spec = setup
    result = integration.provision(cases, tmp_path / 'resources.json', spec, apply=True)
    admin.models['student/provider'] = {'id': 'student/provider'}
    admin.calls.clear()
    admin.workspace_reads.clear()
    output = tmp_path / 'student.json'
    plan = integration.bind(result, output, 'sa-student', 'student/provider')
    assert plan['status'] == 'planned' and plan['rag_params'] == result['rag_params']
    assert admin.workspace_reads == [entry['model_id'] for entry in result['splits'].values()]
    assert all(method == 'GET' for method, _, _ in admin.calls) and not output.exists()


@pytest.mark.parametrize('collision', ['model', 'tool', 'manifest'])
def test_collision_prevents_writes(setup: tuple[Admin, Client, list[dict[str, Any]], dict[str, Any]], tmp_path: Path, collision: str) -> None:
    """Existing resources are not replaced and prevent all proposed writes."""
    admin, client, cases, spec = setup
    output = tmp_path / 'resources.json'
    if collision == 'model':
        admin.models['sa-pilot-dev'] = {'id': 'sa-pilot-dev'}
    elif collision == 'tool':
        admin.tools['sa_pilot_test_retrieval'] = {'id': 'sa_pilot_test_retrieval'}
    else:
        output.write_text('existing manifest', encoding='utf-8')
    with pytest.raises(StopRun):
        integration.provision(cases, output, spec, apply=True)
    assert all(method == 'GET' for method, _, _ in admin.calls)
    assert not client.uploads and not admin.knowledge
    if collision == 'manifest':
        assert output.read_text() == 'existing manifest'


@pytest.mark.parametrize('review', ['isolated_instance_reviewed', 'source_rights_reviewed', 'read_only_tools_reviewed'])
def test_unreviewed_write_prevented(setup: tuple[Admin, Client, list[dict[str, Any]], dict[str, Any]], tmp_path: Path, review: str) -> None:
    """Required instance, source and tool reviews precede every external mutation."""
    admin, client, cases, spec = setup
    spec[review] = False
    with pytest.raises(StopRun):
        integration.provision(cases, tmp_path / 'resources.json', spec, apply=True)
    assert all(method == 'GET' for method, _, _ in admin.calls)
    assert not client.uploads and not admin.knowledge


@pytest.mark.parametrize('field', ['base_model', 'tool', 'builtin_tools', 'file_context', 'knowledge', 'knowledge_type', 'policy', 'missing'])
def test_model_binding_drift_rejected(setup: tuple[Admin, Client, list[dict[str, Any]], dict[str, Any]], tmp_path: Path, field: str) -> None:
    """Collection rejects changed tool, knowledge, capability and policy bindings."""
    admin, _, cases, spec = setup
    result = integration.provision(cases, tmp_path / 'resources.json', spec, apply=True)
    entry = result['splits']['train']
    alias = admin.models[entry['model_id']]['info']
    if field == 'base_model':
        alias['base_model_id'] = 'unrecorded/provider'
    elif field == 'tool':
        alias['meta']['toolIds'] = result['splits']['dev']['tool_ids']
    elif field in {'builtin_tools', 'file_context'}:
        alias['meta']['capabilities'][field] = True
    elif field == 'knowledge':
        alias['meta']['knowledge'][0]['id'] = result['splits']['dev']['knowledge_id']
    elif field == 'knowledge_type':
        alias['meta']['knowledge'][0]['type'] = 'file'
    elif field == 'policy':
        alias['params'] = {**alias['params'], 'system': 'Changed policy'}
    else:
        del admin.models[entry['model_id']]
    admin.calls.clear()
    with pytest.raises(StopRun):
        integration.verify_split_model(admin, result, 'train')
    assert all(method == 'GET' for method, _, _ in admin.calls)


@pytest.mark.parametrize('field,value', [('retrieval_count', 0), ('retrieval_count', True),
                                       ('max_retrieval_chars', 0), ('max_retrieval_chars', '4000'),
                                       ('source_marker', 'invalid-page-name')])
@pytest.mark.parametrize('apply', [False, True])
def test_invalid_scope_or_budget_prevents_writes(
    setup: tuple[Admin, Client, list[dict[str, Any]], dict[str, Any]],
    tmp_path: Path, field: str, value: Any, apply: bool,
) -> None:
    """Validate source markers and budgets during planning, before KBs or uploads."""
    admin, client, cases, spec = setup
    if field == 'source_marker':
        bad_path = tmp_path / f'{value}.md'
        bad_path.write_text('Original content.', encoding='utf-8')
        cases[0]['source_paths'] = [str(bad_path)]
    else:
        spec[field] = value
    output = tmp_path / 'resources.json'
    with pytest.raises(StopRun):
        integration.provision(cases, output, spec, apply=apply)
    assert all(method == 'GET' for method, _, _ in admin.calls)
    assert not output.exists() and not client.uploads and not admin.knowledge


@pytest.mark.parametrize('namespace', ['sa-' + 'x' * 62, 'sa-Pilot'])
def test_invalid_derived_tool_id_prevents_writes(
    setup: tuple[Admin, Client, list[dict[str, Any]], dict[str, Any]],
    tmp_path: Path, namespace: str,
) -> None:
    """Reject IDs that would overflow local limits or change when Open WebUI lowercases them."""
    admin, client, cases, spec = setup
    spec['namespace'] = namespace
    output = tmp_path / 'resources.json'
    with pytest.raises(StopRun):
        integration.provision(cases, output, spec, apply=True)
    assert all(method == 'GET' for method, _, _ in admin.calls)
    assert not output.exists() and not client.uploads and not admin.tools and not admin.knowledge


def test_longest_tool_id_is_valid(
    setup: tuple[Admin, Client, list[dict[str, Any]], dict[str, Any]], tmp_path: Path,
) -> None:
    """Keep the 80-character supported identifier boundary available without mutations."""
    _, _, cases, spec = setup
    spec['namespace'] = 'sa-' + 'x' * 61
    plan = integration.provision(cases, tmp_path / 'resources.json', spec)
    tool_id, = plan['splits']['train']['tool_ids']
    assert len(tool_id) == 80 and safe_id(tool_id).isidentifier()


@pytest.mark.parametrize('execute', [False, True])
def test_bind_source_reader_reuses_only_kb(
    setup: tuple[Admin, Client, list[dict[str, Any]], dict[str, Any]], tmp_path: Path, execute: bool,
) -> None:
    """A reader profile creates new tools/models while preserving shared indexed files and source aliases."""
    admin, client, cases, spec = setup
    original = integration.provision(cases, tmp_path / 'resources.json', spec, apply=True)
    frozen = copy.deepcopy(original)
    models, tools = copy.deepcopy(admin.models), copy.deepcopy(admin.tools)
    profile = {'isolated_instance_reviewed': True, 'retrieval_count': 2,
               'max_retrieval_chars': 4000, 'max_source_chars': 12000}
    admin.calls.clear()
    output = tmp_path / 'reader.json'
    result = integration.bind(original, output, 'sa-reader', 'recorded/teacher',
                              apply=execute, retrieval_spec=profile)
    assert original == frozen and result['retrieval_spec'] == profile
    assert len(client.uploads) == 3 and len(admin.knowledge) == 3
    assert all(admin.models[key] == value for key, value in models.items())
    assert all(admin.tools[key] == value for key, value in tools.items())
    assert result['rag_params']['system'].startswith(original['rag_params']['system'])
    assert 'read_source' in result['rag_params']['system']
    assert all(result['splits'][split]['knowledge_id'] == entry['knowledge_id']
               for split, entry in original['splits'].items())
    if execute:
        assert result['status'] == 'complete' and result['rag_fingerprint'] != original['rag_fingerprint']
        assert result['created_tool_ids'] == result['tool_ids'] and len(result['tool_ids']) == 3
        assert {path for method, path, _ in admin.calls if method == 'POST'} == {
            '/api/v1/tools/create', '/api/v1/models/create'}
        for split in original['splits']:
            integration.verify_split_model(admin, result, split)
        for tid in result['tool_ids']:
            compile(admin.tools[tid]['content'], '<reader>', 'exec')
            assert 'async def read_source(' in admin.tools[tid]['content']
    else:
        assert result['status'] == 'planned' and not output.exists()
        assert all(method == 'GET' for method, _, _ in admin.calls)


@pytest.mark.parametrize('fault', ['review', 'limit', 'missing', 'unknown', 'scope', 'collision', 'changed_model'])
def test_bind_reader_preflight_before_any_write(
    setup: tuple[Admin, Client, list[dict[str, Any]], dict[str, Any]], tmp_path: Path, fault: str,
) -> None:
    """Invalid configuration, drift or collisions fail before creating any new resource."""
    admin, _, cases, spec = setup
    original = integration.provision(cases, tmp_path / 'resources.json', spec, apply=True)
    profile = {'isolated_instance_reviewed': True, 'retrieval_count': 2,
               'max_retrieval_chars': 4000, 'max_source_chars': 12000}
    if fault == 'review':
        profile['isolated_instance_reviewed'] = False
    elif fault == 'limit':
        profile['max_source_chars'] = True
    elif fault == 'missing':
        del profile['max_source_chars']
    elif fault == 'unknown':
        profile['unreviewed'] = True
    elif fault == 'scope':
        original['retrieval_mode'] = 'blueprint'
    elif fault == 'collision':
        admin.tools['sa_reader_test_retrieval'] = {'id': 'sa_reader_test_retrieval'}
    else:
        admin.models[original['splits']['test']['model_id']]['info']['meta']['knowledge'] = []
    admin.calls.clear()
    with pytest.raises(StopRun):
        integration.bind(original, tmp_path / 'reader.json', 'sa-reader', 'recorded/teacher',
                         apply=True, retrieval_spec=profile)
    assert all(method == 'GET' for method, _, _ in admin.calls)
