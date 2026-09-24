"""Collect, evaluate and export wire datasets; no GPU provisioning or training commands."""
from __future__ import annotations

import argparse
from pathlib import Path

from .core import StopRun, canonical, read, require, rows, write, write_rows


def parser() -> argparse.ArgumentParser:
    """Expose the established data CLI with explicit mutation flags."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--env-file', type=Path)
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('preflight')
    a = sub.add_parser('budget-init')
    a.add_argument('--path', type=Path, required=True)
    a.add_argument('--usd', type=float, required=True)
    a.add_argument('--tokens', type=int, default=8000000)
    a.add_argument('--calls', type=int, default=800)
    a.add_argument('--execute', action='store_true')
    a = sub.add_parser('budget-status'); a.add_argument('--path', type=Path, required=True)
    a = sub.add_parser('vidore')
    a.add_argument('--output', type=Path, required=True)
    a.add_argument('--snapshot', type=Path)
    a.add_argument('--download', action='store_true')
    a.add_argument('--revisions', type=Path)
    a.add_argument('--exclusions', type=Path)
    a.add_argument('--split-plan', type=Path)
    a.add_argument('--execute', action='store_true')
    a = sub.add_parser('import-csv')
    for flag in ('csv', 'source-root', 'assignments', 'output'):
        a.add_argument('--' + flag, type=Path, required=True)
    a.add_argument('--execute', action='store_true')
    for name in ('provision', 'collect', 'branch'):
        a = sub.add_parser(name)
        a.add_argument('--dataset', type=Path, required=True)
        a.add_argument('--spec', type=Path, required=True)
        a.add_argument('--execute', action='store_true')
        if name == 'provision':
            a.add_argument('--output', type=Path, required=True)
        else:
            a.add_argument('--run', type=Path, required=True)
        if name == 'collect':
            a.add_argument('--resources', type=Path, required=True)
        if name == 'branch':
            a.add_argument('--source-runs', nargs='+', type=Path, required=True)
    a = sub.add_parser('bind-student')
    for flag in ('resources', 'output'):
        a.add_argument('--' + flag, type=Path, required=True)
    a.add_argument('--namespace', required=True); a.add_argument('--model-id', required=True)
    a.add_argument('--retrieval-spec', type=Path); a.add_argument('--execute', action='store_true')
    for name in ('score', 'import-grades'):
        a = sub.add_parser(name); a.add_argument('--run', type=Path, required=True)
        a.add_argument('--execute', action='store_true')
        if name == 'score': a.add_argument('--promptfoo', default='promptfoo')
    a = sub.add_parser('catalog')
    a.add_argument('--runs', nargs='+', type=Path, required=True)
    a.add_argument('--reviews', type=Path, required=True)
    a.add_argument('--output', type=Path, required=True)
    a.add_argument('--execute', action='store_true')
    a = sub.add_parser('build')
    a.add_argument('--dataset', type=Path, required=True)
    a.add_argument('--runs', nargs='+', type=Path, required=True)
    a.add_argument('--reviewed-catalog', type=Path)
    a.add_argument('--output', type=Path, required=True)
    a.add_argument('--with-reasoning', action='store_true')
    a.add_argument('--execute', action='store_true')
    a = sub.add_parser('compare')
    a.add_argument('--baseline', type=Path, required=True)
    a.add_argument('--candidate', type=Path, required=True)
    a.add_argument('--output', type=Path); a.add_argument('--execute', action='store_true')
    return p


def main() -> int:
    """Dispatch preparation and execution without altering production defaults."""
    args = parser().parse_args()
    from . import integration
    if args.env_file:
        integration.load_env(args.env_file)
    if args.cmd == 'preflight':
        result = integration.audit()
        print(canonical(result))
        return 0 if result['ok'] else 2
    if args.cmd in {'budget-init', 'budget-status'}:
        from .budget import Ledger
        if args.cmd == 'budget-init':
            limits = {'usd': args.usd, 'tokens': args.tokens, 'calls': args.calls, 'seconds': 36000}
            result = Ledger.initialize(args.path, limits).snapshot() if args.execute else {'limits': limits, 'execute': False}
        else:
            result = Ledger(args.path).snapshot()
    elif args.cmd == 'vidore':
        from .vidore import build, fetch
        require(bool(args.snapshot) != args.download, 'Choose --snapshot OR --download')
        if not args.execute:
            result = {'execute': False, 'command': 'vidore'}
        else:
            snapshots = read(args.snapshot) if args.snapshot else fetch(read(args.revisions) if args.revisions else None)
            result = build(snapshots, args.output, exclusions=read(args.exclusions) if args.exclusions else None,
                           split_plan=read(args.split_plan) if args.split_plan else None)
    elif args.cmd == 'import-csv':
        from .datasets import import_csv
        result = import_csv(args.csv, args.source_root, read(args.assignments), args.output, args.execute)
    elif args.cmd in {'provision', 'bind-student', 'collect', 'branch'}:
        from .vidore import load
        require(integration.audit()['ok'], 'Original bundle changed; review integration before live use')
        if args.cmd == 'provision':
            result = integration.provision(load(args.dataset), args.output, read(args.spec), apply=args.execute)
        elif args.cmd == 'bind-student':
            result = integration.bind(read(args.resources), args.output, args.namespace, args.model_id,
                                      apply=args.execute, retrieval_spec=read(args.retrieval_spec) if args.retrieval_spec else None)
        elif not args.execute:
            result = {'execute': False, 'command': args.cmd}
        elif args.cmd == 'collect':
            result = integration.collect(load(args.dataset), read(args.resources), args.run, read(args.spec))
        else:
            result = integration.branch(load(args.dataset), args.source_runs, args.run, read(args.spec))
    elif args.cmd in {'score', 'import-grades'}:
        from .evaluation import score, import_run
        # Historical score() writes preparation files. Reserve those writes too for --execute.
        result = (score(args.run, execute=True, executable=args.promptfoo) if args.cmd == 'score'
                  else import_run(args.run)) if args.execute else {'execute': False, 'command': args.cmd}
    elif args.cmd == 'catalog':
        from .datasets import catalog_runs
        require(not args.output.exists(), 'Catalog already exists')
        values = catalog_runs(args.runs, rows(args.reviews))
        if args.execute: write_rows(args.output, values)
        result = {'episodes': len(values), 'execute': args.execute}
    elif args.cmd == 'build':
        from .datasets import build_reviewed
        require(args.reviewed_catalog is not None, '--reviewed-catalog required; automatic grades do not replace review')
        result = build_reviewed(args.dataset, args.runs, args.reviewed_catalog, args.output,
                                args.with_reasoning, args.execute)
    else:
        from .evaluation import compare
        result = compare(args.baseline, args.candidate)
        if args.execute and args.output:
            require(not args.output.exists(), 'Comparison output already exists')
            write(args.output, result)
    print(canonical(result))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except StopRun as exc:
        raise SystemExit(str(exc)) from None
