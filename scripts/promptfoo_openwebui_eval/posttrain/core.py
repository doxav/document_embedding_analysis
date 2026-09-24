"""Pure data contracts. Ambiguous data is rejected, never silently repaired."""
from __future__ import annotations
import copy
import hashlib
import json
import math
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any
from . import SCHEMA

HARD = {'capture_integrity', 'within_budget', 'retrieval_qrel_hit'}
QUALITY = {'correctness', 'groundedness'}
METRICS = HARD | QUALITY
REASONING_DELIMITERS = (
    ('<think>', '</think>'), ('<thinking>', '</thinking>'),
    ('<reason>', '</reason>'), ('<reasoning>', '</reasoning>'),
    ('<thought>', '</thought>'), ('<Thought>', '</Thought>'),
    ('<|begin_of_thought|>', '<|end_of_thought|>'), ('◁think▷', '◁/think▷'),
    ('[THINK]', '[/THINK]'),
)
REASONING_MARKUP = re.compile('|'.join(
    re.escape(tag[:-1]) + r'(?=\W|$)' for pair in REASONING_DELIMITERS for tag in pair
))

class StopRun(RuntimeError):
    """Stop before an unsafe action or an unverified data transformation."""

def require(ok: bool, why: str) -> None:
    """Stop when an explicit safety or data contract is violated."""
    if not ok:
        raise StopRun(why)

def canonical(value: Any) -> str:
    """Serialize finite JSON deterministically for artifact identity."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)

def digest(value: Any) -> str:
    """Hash the canonical JSON representation of an artifact."""
    return hashlib.sha256(canonical(value).encode()).hexdigest()

def sha(path: str | Path) -> str:
    """Hash file bytes without loading the complete file into memory."""
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1048576), b''):
            h.update(part)
    return h.hexdigest()

def strict_json(text: Any) -> Any:
    """Parse JSON while rejecting duplicate keys and non-finite numbers."""
    def bad(v: Any) -> Any:
        """Reject a non-finite JSON constant."""
        raise StopRun('Non-finite JSON')
    def unique(pairs: Any) -> Any:
        """Build an object only when every JSON key is unique."""
        out = {}
        for k, v in pairs:
            require(k not in out, f'Duplicate JSON key: {k}')
            out[k] = v
        return out
    return json.loads(text, parse_constant=bad, object_pairs_hook=unique)

def read(path: str | Path) -> Any:
    """Read one artifact using the strict JSON parser."""
    return strict_json(Path(path).read_text(encoding='utf-8'))

def rows(path: str | Path) -> list[dict[str, Any]]:
    """Read JSONL objects without silently discarding malformed records."""
    result = [strict_json(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]
    require(all(isinstance(x, dict) for x in result), 'JSONL rows must be objects')
    return result

def write_text(path: str | Path, text: Any) -> None:
    """Atomically replace a private artifact and flush its bytes to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix='.sa-')
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(text); f.flush(); os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)

def write(path: str | Path, value: Any) -> None:
    """Persist a finite JSON artifact with private permissions."""
    write_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

def write_rows(path: str | Path, values: Any) -> None:
    """Persist canonical JSONL records atomically."""
    write_text(path, ''.join(canonical(x) + '\n' for x in values))

def safe_id(value: Any) -> str:
    """Validate an identifier before using it in a resource or artifact name."""
    require(isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', value), 'Unsafe identifier')
    return value

def inside(root: str | Path, relative: str) -> Path:
    """Resolve a relative artifact path without escaping its declared root."""
    require(not Path(relative).is_absolute(), 'Relative artifact path required')
    root = Path(root).resolve(); p = root / relative
    require(not p.is_symlink() and p.resolve().is_relative_to(root), 'Path escapes artifact root')
    return p

def text(value: Any) -> str:
    """Extract text content without silently discarding unsupported modalities."""
    if value is None: return ''
    if isinstance(value, str): return value
    require(isinstance(value, list), 'Unknown content format')
    require(all(isinstance(x, dict) and x.get('type') in {'text', 'input_text', 'output_text'} and isinstance(x.get('text'), str) for x in value),
            'Text-only pilot: images/audio/unknown content are not silently discarded')
    return ''.join(x['text'] for x in value)

def strip_reasoning(value: str) -> str:
    """Remove explicit leading assistant scratchpads; reject ambiguous reasoning markup."""
    require(isinstance(value, str), 'Assistant reasoning content must be text')
    while True:
        previous = value
        for op, cl in REASONING_DELIMITERS:
            if value.lstrip().startswith(op):
                start = value.index(op); end = value.find(cl, start + len(op))
                require(end >= 0, 'Unclosed reasoning block')
                require(REASONING_MARKUP.search(value[start + len(op):end]) is None,
                        'Nested or malformed assistant reasoning delimiters')
                value = value[end + len(cl):].lstrip('\n')
        if value == previous: break
    require(REASONING_MARKUP.search(value) is None, 'Ambiguous assistant reasoning delimiters')
    return value

def normalize(messages: Any) -> list[dict[str, Any]]:
    """Normalize messages and tool-call links while stripping assistant reasoning."""
    require(isinstance(messages, list) and messages, 'Empty conversation')
    out, ids, pending = [], {}, set()
    for raw in messages:
        role = raw.get('role')
        require(role in {'system', 'developer', 'user', 'assistant', 'tool'}, 'Unknown role')
        content = text(raw.get('content'))
        msg = {'role': role, 'content': strip_reasoning(content) if role == 'assistant' else content}
        calls = raw.get('tool_calls') or []
        if role == 'tool':
            old = raw.get('tool_call_id')
            require(old in pending, 'Orphan or duplicate tool result')
            msg['tool_call_id'] = ids[old]; pending.remove(old)
            if raw.get('name'): msg['name'] = raw['name']
        else:
            require(not pending, 'Missing tool observation before next non-tool message')
            require(not calls or role == 'assistant', 'Only assistant can call tools')
            if calls:
                msg['tool_calls'] = []
                for call in calls:
                    old = call.get('id'); fn = call.get('function') or {}
                    require(old and old not in ids and call.get('type', 'function') == 'function', 'Invalid tool call ID/type')
                    require(isinstance(fn.get('name'), str) and fn['name'], 'Missing tool name')
                    args = fn.get('arguments', '{}')
                    args = strict_json(args) if isinstance(args, str) else args
                    require(isinstance(args, dict), 'Arguments must be a JSON object')
                    ids[old] = f't{len(ids)+1:08d}'; pending.add(old)
                    msg['tool_calls'].append({'id': ids[old], 'type': 'function', 'function': {'name': fn['name'], 'arguments': canonical(args)}})
        out.append(msg)
    require(not pending or out[-1].get('tool_calls'), 'Missing tool observations at end')
    return out

def tools_canonical(tools: Any) -> list[dict[str, Any]]:
    """Validate tool schemas and serialize their parameter objects deterministically."""
    import jsonschema
    out = copy.deepcopy(tools or []); names = set()
    for tool in out:
        require(tool.get('type', 'function') == 'function', 'Only function tools')
        fn = tool.get('function') or {}; name = fn.get('name')
        require(name and name not in names, 'Missing/duplicate tool schema'); names.add(name)
        params = fn.get('parameters', {'type':'object'})
        params = strict_json(params) if isinstance(params, str) else params
        require(isinstance(params, dict), 'Invalid tool parameters')
        jsonschema.Draft202012Validator.check_schema(params)
        fn['parameters'] = canonical(params)
    return out

def validate_tools(messages: Any, tools: Any) -> None:
    """Check every captured tool call against its declared schema."""
    import jsonschema
    schemas = {x['function']['name']: strict_json(x['function']['parameters']) for x in tools_canonical(tools)}
    for msg in messages:
        for call in msg.get('tool_calls') or []:
            fn = call['function']; require(fn['name'] in schemas, 'Unknown/unauthorized function')
            args = fn['arguments']; args = strict_json(args) if isinstance(args, str) else args
            try: jsonschema.validate(args, schemas[fn['name']])
            except jsonschema.ValidationError as exc: raise StopRun('Invalid tool arguments') from exc

def turn_example(record: Any, case: Any) -> dict[str, Any]:
    """Build one next-action example from a complete provider wire record."""
    require(record.get('schema') == SCHEMA and record.get('source') == 'upstream_wire', 'Exact v3 provider capture required')
    require(record.get('complete') is True and not record.get('error'), 'Incomplete capture')
    req, resp = record['request'], record['response']
    choices = resp.get('choices', [])
    require(len(choices) == 1 and choices[0].get('finish_reason') in {'stop', 'tool_calls'}, 'Incomplete or multiple choices')
    target = choices[0]['message']; require(target.get('role', 'assistant') == 'assistant', 'Invalid target role')
    messages = normalize(req['messages'] + [dict(target, role='assistant')]); tools = tools_canonical(req.get('tools'))
    require(messages[-1]['content'] or messages[-1].get('tool_calls'), 'Empty supervised target')
    validate_tools(messages, tools)
    for m in messages: m['train'] = False
    messages[-1]['train'] = True
    return {'messages':messages, 'tools':tools, 'case_id':case['case_id'], 'group_id':case['group_id'], 'split':case['split'],
            'capture_hash':digest(record)}

def captured_reasoning_content(message: dict[str, Any]) -> str:
    """Keep captured text reasoning once; reject conflicting or opaque representations."""
    content = text(message.get('content'))
    answer = strip_reasoning(content)
    sources: list[str] = []
    for key in ('reasoning', 'reasoning_content', 'thinking'):
        value = message.get(key)
        require(value is None or isinstance(value, str), 'Reasoning must be captured text')
        if value:
            sources.append(value)
    details = message.get('reasoning_details')
    require(details is None or isinstance(details, list), 'Reasoning details must be a list')
    if details:
        require(isinstance(details, list) and all(isinstance(part, dict)
                and part.get('type') == 'reasoning.text' and isinstance(part.get('text'), str)
                for part in details), 'Opaque/nontext reasoning cannot become a text SFT target')
        sources.append(''.join(part['text'] for part in details))
    inline = content.lstrip().startswith('<think>')
    require(answer == content or inline, 'Reasoning variant requires reviewed <think> delimiters')
    if inline:
        sources.append(content.split('<think>', 1)[1].split('</think>', 1)[0])
        require(not REASONING_MARKUP.search(content.split('</think>', 1)[1]),
                'Multiple inline reasoning blocks require review')
    if not sources:
        return content
    require(all(source.strip() == sources[0].strip() for source in sources),
            'Captured reasoning representations disagree')
    require(not REASONING_MARKUP.search(sources[0]), 'Ambiguous captured reasoning delimiters')
    if inline:
        return content
    return '<think>\n' + sources[0] + '\n</think>\n\n' + content if sources[0].strip() else content


def reasoning_sft(sft: list[dict[str, Any]], captures: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Add wire reasoning to the already selected SFT rows without changing episode selection."""
    records = {digest(record): record for capture in captures.values() for record in capture['records']}
    result = []
    missing_targets = []
    assistant_messages = 0
    with_reasoning = 0
    for index, example in enumerate(sft):
        require(example.get('split') == 'train', 'Reasoning SFT must exclude holdouts')
        record = records.get(example.get('capture_hash'))
        require(record is not None and turn_example(record, example) == example,
                'Reasoning variant must match the selected original wire example')
        raw = record['request']['messages'] + [record['response']['choices'][0]['message']]
        enriched = copy.deepcopy(example)
        for position, (source, target) in enumerate(zip(raw, enriched['messages'])):
            if target['role'] != 'assistant':
                continue
            assistant_messages += 1
            target['content'] = captured_reasoning_content(source)
            require(strip_reasoning(target['content']) == example['messages'][position]['content'],
                    'Adding reasoning changed the original assistant answer')
            present = target['content'] != example['messages'][position]['content']
            with_reasoning += present
            if position == len(raw) - 1 and not present:
                missing_targets.append(index)
        result.append(enriched)
    return result, {'sft_hash': digest(sft), 'sft_reasoning_hash': digest(result),
                    'examples': len(result), 'assistant_messages': assistant_messages,
                    'assistant_messages_with_reasoning': with_reasoning,
                    'target_indices_without_reasoning': missing_targets,
                    'required_chat_template_kwargs': {'enable_thinking': True},
                    'actual_axolotl_validation': 'PENDING'}

def source_marker(dataset: Any, page: Any) -> str:
    """Derive a stable source marker from dataset and page identity."""
    return 'vsrc_' + digest([dataset, str(page)])[:20]

def evidence(capture: Any) -> list[dict[str, Any]]:
    # Only what the FINAL provider call actually saw: not the oracle and not
    # discarded earlier observations. A tool adapter must retain source markers.
    """Return only tool observations received by the final provider request."""
    records = capture.get('records') or []; require(records, 'Empty episode')
    return [copy.deepcopy(m) for m in records[-1]['request']['messages'] if m.get('role') == 'tool']

def retrieval(capture: Any, case: Any) -> dict[str, Any]:
    """Check required source markers in the evidence actually received."""
    if not case.get('requires_retrieval', True):
        return {'pass':True, 'score':1.0, 'reason':'retrieval not required', 'recall':None}
    wanted = set(case.get('relevant_source_markers') or [])
    require(wanted, 'Missing qrel mapping')
    observed = set(re.findall(r'\bvsrc_[0-9a-f]{20}\b', canonical(evidence(capture))))
    hit = wanted & observed
    # Policy, not a claim about qrel score semantics: one hit is a necessary
    # pilot gate, not proof of answer sufficiency; grounding remains independent.
    recall = len(hit)/len(wanted)
    return {'pass':bool(hit), 'score':float(bool(hit)), 'recall':recall, 'hits':sorted(hit), 'observed':sorted(observed)}

def metrics(capture: dict[str, Any]) -> dict[str, Any]:
    """Sum every provider turn; optional usage details remain unknown when omitted."""
    rs = capture.get('records') or []; usages = [r['response'].get('usage') or {} for r in rs]
    known = bool(rs) and all(type(u.get('completion_tokens')) is int and u['completion_tokens'] >= 0 for u in usages)
    total_known = known and all(type(u.get('prompt_tokens')) is int and u['prompt_tokens'] >= 0
        and type(u.get('total_tokens')) is int and u['total_tokens'] == u['prompt_tokens'] + u['completion_tokens']
        for u in usages)
    def detail_sum(container: str, field: str, parent: str) -> int | None:
        """Return a measured subset only when every call reports a valid count."""
        values = [u[container].get(field) if isinstance(u.get(container), dict) else None for u in usages]
        if not total_known or not all(type(v) is int and 0 <= v <= u[parent] for u, v in zip(usages, values)):
            return None
        return sum(values)
    return {'llm_calls':len(rs), 'tool_calls':sum(len(r['response']['choices'][0]['message'].get('tool_calls') or []) for r in rs),
            'completion_tokens':sum(u['completion_tokens'] for u in usages) if known else None,
            'prompt_tokens':sum(u['prompt_tokens'] for u in usages) if total_known else None,
            'total_tokens':sum(u['total_tokens'] for u in usages) if total_known else None,
            'total_usage_known':total_known,
            'cached_input_tokens':detail_sum('prompt_tokens_details', 'cached_tokens', 'prompt_tokens'),
            'reasoning_tokens':detail_sum('completion_tokens_details', 'reasoning_tokens', 'completion_tokens'),
            'seconds':capture.get('duration_seconds'), 'usage_known':known}

def task_result_row(case: dict[str, Any], candidate: dict[str, Any], capture: dict[str, Any]) -> dict[str, Any]:
    """Keep original task evaluation variables beside measured whole-episode performance."""
    usage = metrics(capture)
    costs = [(record['response'].get('usage') or {}).get('cost') for record in capture.get('records', [])]
    cost = sum(costs) if costs and all(type(value) in (int, float) and math.isfinite(value)
                                     and value >= 0 for value in costs) else None
    return {**case.get('task_metadata', {}), 'benchmark_id': candidate['benchmark_id'],
            'candidate_answer': candidate['candidate_answer'], 'generation_cost': cost,
            'openwebui_pipe_model': candidate['model'],
            **{'generation_' + target: usage.get(source) for target, source in (
                ('prompt_tokens', 'prompt_tokens'), ('completion_tokens', 'completion_tokens'),
                ('total_tokens', 'total_tokens'), ('duration_seconds', 'seconds'),
                ('llm_requests', 'llm_calls'), ('tool_calls', 'tool_calls'))}}


def validate_cases(cases: Any) -> None:
    """Reject invalid cases and document or group leakage across splits."""
    ids, groups, docs = set(), {}, {}
    for c in cases:
        require(c['case_id'] not in ids, 'Duplicate case'); ids.add(c['case_id'])
        require(c['split'] in {'train','dev','test'} and c['question'] and c['reference_answer'], 'Invalid QA')
        require(groups.setdefault(c['group_id'], c['split']) == c['split'], 'Group/translation leakage')
        require(c.get('document_ids'), 'No source document')
        for doc in c['document_ids']:
            require(docs.setdefault((c['dataset'],doc), c['split']) == c['split'], 'Document leakage')

def import_grades(payload: dict[str, Any], candidates: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Import explicit assertion verdicts while rejecting provider and grading infrastructure failures."""
    rr = payload.get('results'); rr = rr.get('results') if isinstance(rr, dict) else rr
    require(isinstance(rr,list), 'Unknown Promptfoo export schema')
    out, seen = [], set()
    for row in rr:
        vv = (row.get('testCase') or {}).get('vars') or row.get('vars') or {}; bid = vv.get('benchmark_id')
        require(bid in candidates and bid not in seen, 'Missing/duplicate result')
        failure_reason = row.get('failureReason')
        # Promptfoo 0.122 exports ASSERT=1 with error=gradingResult.reason; ERROR=2 is infrastructure.
        if 'failureReason' in row:
            require(type(failure_reason) is int and failure_reason in (0,1), 'Unknown failure reason or infrastructure error')
        response = row.get('response')
        require(response is None or isinstance(response,dict), 'Unknown provider response schema')
        require(not (response or {}).get('error'), 'Provider infrastructure error')
        seen.add(bid); cand = candidates[bid]
        require(vv.get('candidate_answer') == cand['candidate_answer'], 'Stale evaluation')
        grade = row.get('gradingResult') or {}; parts = {}
        require(not grade.get('error') and not any(str(value).startswith('SA_INFRASTRUCTURE:')
                for value in (row.get('error',''),grade.get('reason',''))), 'Grading infrastructure error')
        if row.get('error'):
            require(failure_reason == 1 and isinstance(row['error'],str)
                    and row['error'] == grade.get('reason') and grade.get('pass') is False,
                    'Unexplained result error; not a validated assertion failure')
        for part in grade.get('componentResults') or []:
            require(not part.get('error') and not str(part.get('reason','')).startswith('SA_INFRASTRUCTURE:'), 'Judge infrastructure error')
            name = (part.get('assertion') or {}).get('metric') or part.get('metric')
            if name not in METRICS: continue
            require(name not in parts, 'Duplicate metric')
            s, p = part.get('score'), part.get('pass')
            require(type(p) is bool and type(s) in (float,int) and math.isfinite(s) and 0 <= s <= 1, 'Explicit Boolean pass and finite score required')
            parts[name] = {'pass':p, 'score':float(s), 'reason':part.get('reason','')}
        require(parts.keys() == METRICS, 'Missing assertion')
        passed = all(p['pass'] for p in parts.values())
        require(grade.get('pass') is passed and row.get('success',passed) is passed, 'Aggregate disagrees with assertions')
        if 'failureReason' in row:
            require(failure_reason == (0 if passed else 1), 'Failure reason disagrees with assertions')
        out.append({'benchmark_id':bid, 'pass':passed, 'hard_ok':all(parts[k]['pass'] for k in HARD), 'metrics':parts,
                    'score':min(parts[k]['score'] for k in QUALITY), 'capture_hash':cand['capture_hash'], 'answer_hash':digest(cand['candidate_answer'])})
    require(seen == candidates.keys(), 'Partial export is not a complete evaluation')
    return out

def build_data(cases: Any, candidates: Any, grades: Any, captures: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Select positive next-action targets and score-separated same-state preferences."""
    validate_cases(cases); cm = {c['case_id']:c for c in cases}; gm = {g['benchmark_id']:g for g in grades}
    require(len(gm)==len(grades) and len({c['benchmark_id'] for c in candidates})==len(candidates), 'Duplicate candidates/grades')
    best, states = {}, defaultdict(list)
    for cand in candidates:
        c = cm[cand['case_id']]
        if c['split'] != 'train': continue
        bid = cand['benchmark_id']; g = gm[bid]; cap = captures[bid]
        require(digest(cap)==g['capture_hash']==cand['capture_hash'] and digest(cand['candidate_answer'])==g['answer_hash'], 'Artifacts changed after grading')
        if not g['hard_ok']: continue
        require(cap.get('complete') is True, 'Incomplete episode')
        examples = [turn_example(r,c) for r in cap['records']]; final = examples[-1]
        require(not final['messages'][-1].get('tool_calls') and final['messages'][-1]['content'].strip()==cand['candidate_answer'].strip(), 'Evaluation not on wire target')
        if g['pass'] and cand.get('kind') == 'episode':
            key = (-g['score'], len(canonical(examples)), bid)
            if c['case_id'] not in best or key < best[c['case_id']][0]: best[c['case_id']] = (key,examples)
        state = digest({'messages':final['messages'][:-1], 'tools':final['tools']})
        states[state].append((g,final))
    sft = [ex for _,exs in best.values() for ex in exs]
    # Keep seed weighting explicit; exact duplicate observations are not new tasks.
    sft = list({digest({'messages':ex['messages'],'tools':ex['tools']}):ex for ex in sft}.values())
    pairs=[]
    for state, entries in states.items():
        good=sorted([x for x in entries if x[0]['pass']],key=lambda x:-x[0]['score'])
        bad=sorted([x for x in entries if not x[0]['pass']],key=lambda x:-x[0]['score'])
        if not good or not bad: continue
        g,p = good[0]
        for b,n in bad:
            if g['score']-b['score'] < .1: continue
            require(p['group_id']==n['group_id'],'Pair crosses groups')
            chosen={k:v for k,v in p['messages'][-1].items() if k!='train'}
            rejected={k:v for k,v in n['messages'][-1].items() if k!='train'}
            if chosen == rejected: continue
            pairs.append({'messages':p['messages'][:-1], 'tools':p['tools'],'chosen':chosen,'rejected':rejected,
                          'group_id':p['group_id'],'split':'train','state_hash':state})
            break
    report={'sft_examples':len(sft),'seed_cases':len(best),'pairs':len(pairs),'pair_groups':len({x['group_id'] for x in pairs}),
            'sft_hash':digest(sft),'pairs_hash':digest(pairs),'simpo_ready':len(pairs)>=20 and len({x['group_id'] for x in pairs})>=10}
    return sft,pairs,report
