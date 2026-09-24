"""Reproducible text derivative: original QA/qrels, no teacher-generated gold."""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import re
from typing import Any
from . import SCHEMA
from .core import require, digest, sha, rows, read, write, write_rows, write_text, source_marker, validate_cases, inside

CORPORA={'vidore/vidore_v3_energy':'energy','vidore/vidore_v3_finance_en':'finance',
         'vidore/vidore_v3_finance_fr':'finance','vidore/vidore_v3_hr':'hr','vidore/vidore_v3_pharmaceuticals':'pharmaceuticals'}
LANG={'french':'fr','fr':'fr','français':'fr','english':'en','en':'en'}


def fetch(revisions: Any=None) -> Any:
    """Resolve dataset revisions and download only the required text columns."""
    from huggingface_hub import HfApi
    from datasets import load_dataset
    snapshots=[]
    for name in CORPORA:
        revision=HfApi().dataset_info(name,revision=(revisions or {}).get(name,'main')).sha
        require(revision and re.fullmatch('[0-9a-f]{40}',revision),'Cannot pin dataset revision')
        def selected(config: Any,columns: Any) -> Any:
            # Project out image blobs. Schema changes are errors, never silently
            # coerced by inventing alternative field names.
            """Stream the selected columns from one immutable dataset configuration."""
            return list(load_dataset(name,name=config,split='test',revision=revision,streaming=True,columns=columns))
        snapshots.append({'dataset':name,'revision':revision,
          'queries':selected('queries',['query_id','query','language','answer','raw_answers','content_type']),
          'qrels':selected('qrels',['query_id','corpus_id','score']),
          'corpus':selected('corpus',['corpus_id','doc_id','markdown','page_number_in_doc'])})
    return snapshots


def reviewed_exclusions(snapshots: list[dict[str, Any]], exclusions: list[dict[str, str]] | None) -> list[dict[str, str]]:
    """Validate reviewed query exclusions against their immutable source snapshots."""
    if exclusions is None:
        return []
    require(isinstance(exclusions, list), 'Review exclusions must be a list')
    known = {snap['dataset']: snap for snap in snapshots}
    validated = []
    seen = set()
    for item in exclusions:
        require(isinstance(item, dict) and set(item) == {'dataset', 'revision', 'query_id', 'reason'},
                'Review exclusion requires exactly dataset, revision, query_id and reason')
        require(all(isinstance(value, str) and value.strip() for value in item.values()),
                'Review exclusion fields must be nonempty strings')
        dataset, revision, query_id = item['dataset'], item['revision'], item['query_id']
        require(dataset in known, 'Review exclusion dataset is not in the source snapshots')
        require(re.fullmatch('[0-9a-f]{40}', revision) and revision == known[dataset]['revision'],
                'Review exclusion revision does not match the immutable source snapshot')
        require(sum(str(query['query_id']) == query_id for query in known[dataset]['queries']) == 1,
                'Review exclusion query_id must identify exactly one source query')
        key = (dataset, revision, query_id)
        require(key not in seen, 'Duplicate review exclusion')
        seen.add(key)
        validated.append(dict(item))
    return sorted(validated, key=lambda item: (item['dataset'], item['revision'], item['query_id']))


def reviewed_split_plan(snapshots: list[dict[str, Any]], plan: dict[str, Any] | None,
                        exclusions: list[dict[str, str]]) -> tuple[dict[str, list[str]], dict[tuple[str, str, str], str]]:
    """Keep document splits fixed, with explicit dev QA repairs and an unchanged test."""
    if plan is None:
        return {}, {}
    fields = {'baseline', 'baseline_manifest_sha256', 'documents', 'reserved_documents'}
    require(isinstance(plan, dict) and fields <= set(plan)
            and set(plan) <= fields | {'replace_excluded_dev_queries'}, 'Split plan fields invalid')
    replace_dev = plan.get('replace_excluded_dev_queries', False)
    require(type(replace_dev) is bool, 'Split plan dev replacement option must be boolean')
    require(isinstance(plan['baseline'], str) and Path(plan['baseline']).is_absolute(),
            'Split plan baseline must be an absolute dataset path')
    baseline = Path(plan['baseline'])
    require(sha(baseline / 'manifest.json') == plan['baseline_manifest_sha256'], 'Split plan baseline manifest drift')
    previous = load(baseline)
    known = {snap['dataset']: snap for snap in snapshots}
    selections: dict[str, dict[str, list[str]]] = {name: {s: [] for s in ('train', 'dev', 'test')} for name in known}
    assigned: dict[tuple[str, str], str] = {}
    reserved: set[tuple[str, str]] = set()
    for field in ('documents', 'reserved_documents'):
        require(isinstance(plan[field], list), 'Split plan documents must be lists')
        for item in plan[field]:
            require(isinstance(item, dict) and set(item) == {'dataset', 'revision', 'doc_id', 'split'}
                    and all(isinstance(v, str) and v.strip() for v in item.values()), 'Split plan document fields invalid')
            name, doc, split = item['dataset'], item['doc_id'], item['split']
            require(name in known and item['revision'] == known[name]['revision'], 'Split plan revision mismatch')
            require(doc in {str(p['doc_id']) for p in known[name]['corpus']}, 'Split plan document absent from snapshot')
            require(split in ('dev', 'test') if field == 'reserved_documents' else split in ('train', 'dev', 'test'),
                    'Split plan split invalid')
            key = (name, doc)
            require(key not in assigned and key not in reserved, 'Split plan duplicate or reserved document collision')
            if field == 'reserved_documents':
                reserved.add(key)
            else:
                assigned[key] = split
                selections[name][split].append(doc)
    for selection in selections.values():
        require({s: len(docs) for s, docs in selection.items()} == {'train': 3, 'dev': 1, 'test': 1},
                'Split plan requires 3/1/1 documents per corpus')
    inherited = read(baseline / 'manifest.json').get('split_plan', {}).get('reserved_documents', [])
    require({(d['dataset'], d['doc_id']) for d in inherited} <= reserved,
            'Split plan cannot forget previously reserved documents')
    heldout: dict[tuple[str, str, str], str] = {}
    excluded = {(item['dataset'], item['query_id']) for item in exclusions}
    for case in previous:
        name, doc = case['dataset'], case['document_ids'][0]
        require(case['revision'] == known[name]['revision'], 'Split plan baseline revision mismatch')
        if case['split'] != 'train':
            require(assigned.get((name, doc)) == case['split'], 'Split plan cannot move or replace a baseline holdout')
            replaced = case['split'] == 'dev' and replace_dev and (name, case['query_id']) in excluded
            require(replaced or (name, case['query_id']) not in excluded, 'Split plan cannot exclude a sealed holdout query')
            queries = [q for q in known[name]['queries'] if str(q['query_id']) == case['query_id']]
            require(len(queries) == 1 and queries[0]['query'] == case['question']
                    and queries[0]['answer'] == case['reference_answer']
                    and queries[0].get('raw_answers') == case.get('raw_answers'),
                    'Split plan sealed holdout reference drift')
            if not replaced:
                heldout[(name, doc, case['language'])] = case['query_id']
        elif (name, doc) in assigned:
            require(assigned[(name, doc)] == 'train', 'Split plan cannot promote a baseline train document to holdout')
    return {name: [doc for split in ('train', 'dev', 'test') for doc in sorted(selection[split])]
            for name, selection in selections.items()}, heldout


def build(snapshots: list[dict[str, Any]], directory: Path, seed: int = 42,
          exclusions: list[dict[str, str]] | None = None,
          split_plan: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build the unchanged deterministic derivative after any explicit reviewed exclusions."""
    require(not directory.exists(),'Dataset path exists; do not overwrite/resplit')
    require(len(snapshots)==5 and {x['dataset'] for x in snapshots}==set(CORPORA),'All five corpus snapshots required')
    review_exclusions = reviewed_exclusions(snapshots, exclusions)
    fixed_documents, heldout_queries = reviewed_split_plan(snapshots, split_plan, review_exclusions)
    excluded_queries = {(item['dataset'], item['query_id']) for item in review_exclusions}
    cases=[];documents=[];excluded=defaultdict(int)
    # Prepare all eligibility before creating output: missing languages stops
    # instead of leaving a partially valid 50-question manifest.
    selections=[]
    for snap in snapshots:
        name=snap['dataset'];require(re.fullmatch('[0-9a-f]{40}',snap['revision']),'Immutable dataset revision required')
        pages={str(p['corpus_id']):p for p in snap['corpus']};require(len(pages)==len(snap['corpus']),'Duplicate page ids')
        qrels=defaultdict(set)
        for rel in snap['qrels']:
            if float(rel['score'])>0:
                pid=str(rel['corpus_id']);require(pid in pages,'Qrel refers to absent page');qrels[str(rel['query_id'])].add(pid)
        bydoc=defaultdict(lambda:defaultdict(list))
        for q in snap['queries']:
            if (name, str(q['query_id'])) in excluded_queries:
                excluded['review_exclusion'] += 1
                continue
            lang=LANG.get(str(q.get('language','')).lower());rel=qrels.get(str(q['query_id']),set())
            if not lang or not rel or not isinstance(q.get('answer'),str) or not q['answer'].strip():excluded['missing_qa_language']+=1;continue
            if re.search(r'\b(chart|figure|image|graph|diagram)s?\b',str(q.get('content_type','')).lower()):excluded['visual_dependency']+=1;continue
            docs={str(pages[x]['doc_id']) for x in rel}
            if len(docs)!=1:excluded['multi_document_query']+=1;continue
            if any(not str(pages[x].get('markdown') or '').strip() for x in rel):excluded['missing_text']+=1;continue
            bydoc[next(iter(docs))][lang].append((q,sorted(rel)))
        eligible=sorted((doc for doc,v in bydoc.items() if v['fr'] and v['en']),key=lambda d:digest([seed,name,d]))
        require(len(eligible)>=5,f'{name}: fewer than five disjoint bilingual document groups; review dataset, no silent substitution')
        chosen = fixed_documents.get(name, eligible[:5])
        require(all(doc in eligible for doc in chosen), f'{name}: fixed document lacks bilingual eligible QA; no resplit')
        for doc in chosen:
            for lang in ('fr', 'en'):
                query_id = heldout_queries.get((name, doc, lang))
                if query_id is not None:
                    matches = [pair for pair in bydoc[doc][lang] if str(pair[0]['query_id']) == query_id]
                    require(len(matches) == 1, 'Split plan sealed holdout query unavailable or ambiguous')
                    bydoc[doc][lang] = matches
        selections.append((snap,pages,bydoc,chosen))
    directory.mkdir(parents=True);directory.chmod(0o700)
    for snap,pages,bydoc,chosen in selections:
        name=snap['dataset']
        for i,doc in enumerate(chosen):
            split='train' if i<3 else 'dev' if i==3 else 'test';group=digest([name,doc])[:24];paths=[]
            for page in sorted((p for p in pages.values() if str(p['doc_id'])==doc),key=lambda p:int(p['page_number_in_doc'])):
                pid=str(page['corpus_id']);marker=source_marker(name,pid);relpath=f'documents/{group}/{marker}.md'
                body=f'Source-ID: {marker}\nDocument-ID: {doc}\nPage-as-published: {page["page_number_in_doc"]}\n\n{page.get("markdown") or ""}\n'
                write_text(directory/relpath,body);paths.append(relpath)
                documents.append({'path':relpath,'sha256':sha(directory/relpath),'dataset':name,'revision':snap['revision'],
                                  'corpus_id':pid,'doc_id':doc,'group_id':group,'split':split,'source_marker':marker})
            for lang in ('fr','en'):
                q,relevant=min(bydoc[doc][lang],key=lambda x:digest([seed,str(x[0]['query_id']),lang]))
                cases.append({'schema':SCHEMA,'case_id':'v3-'+group+'-'+lang,'group_id':group,'split':split,
                    'dataset':name,'revision':snap['revision'],'domain':CORPORA[name],'language':lang,'query_id':str(q['query_id']),
                    'question':q['query'],'reference_answer':q['answer'],'raw_answers':q.get('raw_answers'),
                    'document_ids':[doc],'source_paths':paths,'requires_retrieval':True,
                    'relevant_source_markers':[source_marker(name,p) for p in relevant],
                    'relevant_corpus_ids':relevant,'reference_kind':'original ViDoRe answer; human/raw annotations retained'})
    validate_cases(cases);require(len(cases)==50,'Expected 50 QA')
    manifest={'schema':SCHEMA,'cases_hash':digest(cases),'documents_hash':digest(documents),'seed':seed,
              'splits':{s:sum(c['split']==s for c in cases) for s in ('train','dev','test')},
              'languages':{s:sum(c['language']==s for c in cases) for s in ('fr','en')},'excluded':dict(excluded),
              'warning':'Public test derivative, text RAG only; not an official visual benchmark; review source copyright/markdown fidelity.'}
    if review_exclusions:
        manifest['review_exclusions'] = review_exclusions
        manifest['review_exclusions_hash'] = digest(review_exclusions)
    if split_plan is not None:
        manifest['split_plan'] = split_plan
        manifest['split_plan_hash'] = digest(split_plan)
    write_rows(directory/'cases.jsonl',cases);write(directory/'documents.json',documents);write(directory/'manifest.json',manifest)
    return manifest


def load(directory: Path) -> Any:
    """Load and validate the pinned source artifacts before using them."""
    cases=rows(directory/'cases.jsonl');docs=read(directory/'documents.json');manifest=read(directory/'manifest.json')
    if 'review_exclusions' in manifest:
        require(digest(manifest['review_exclusions']) == manifest.get('review_exclusions_hash'), 'Review exclusions manifest drift')
    if 'split_plan' in manifest:
        require(digest(manifest['split_plan']) == manifest.get('split_plan_hash'), 'Split plan manifest drift')
        expected = {(d['dataset'], d['revision'], d['doc_id'], d['split']) for d in manifest['split_plan']['documents']}
        require({(d['dataset'], d['revision'], d['doc_id'], d['split']) for d in docs} == expected,
                'Split plan document assignments drift')
    require(digest(cases)==manifest['cases_hash'] and digest(docs)==manifest['documents_hash'],'Dataset/reference manifest drift')
    for doc in docs:require(sha(inside(directory,doc['path']))==doc['sha256'],'Document/page bytes changed')
    known={d['path']:d for d in docs}
    for case in cases:
        for rel in case['source_paths']:require(rel in known and known[rel]['split']==case['split'],'Cross-split source reference')
        case['source_paths']=[str(inside(directory,p).resolve()) for p in case['source_paths']]
    validate_cases(cases);return cases
