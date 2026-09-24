"""Durable reservation accounting. Ambiguous operations retain their reservation."""
from __future__ import annotations
from typing import Any
import contextlib
import math
import os
import sqlite3
import time
import uuid
from pathlib import Path
from .core import require, canonical, strict_json

class Ledger:
    def __init__(self, path: str | Path) -> None:
        """Initialize local state without starting a provider request."""
        self.path=Path(path)
        require(self.path.is_file(), 'Initialize the budget explicitly before any paid action')

    @staticmethod
    def initialize(path: str | Path, limits: Any) -> Any:
        """Create a new private ledger without resetting an existing budget."""
        path=Path(path); require(not path.exists(), 'Existing budget cannot be reset')
        require(set(limits)=={'seconds','usd','tokens','calls'}, 'Expected seconds/usd/tokens/calls limits')
        require(all(type(v) in (float,int) and math.isfinite(v) and v>0 for v in limits.values()),'Invalid budget')
        require(limits['seconds']<=36000, 'GPU allocation budget maximum is ten hours in this pilot')
        path.parent.mkdir(parents=True,exist_ok=True)
        fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.close(fd)
        with sqlite3.connect(path) as db:
            db.execute('CREATE TABLE limits (payload TEXT NOT NULL)')
            db.execute('INSERT INTO limits VALUES (?)',(canonical(limits),))
            db.execute('CREATE TABLE tickets (id TEXT PRIMARY KEY, started REAL, deadline REAL, kind TEXT, reserved TEXT, actual TEXT, status TEXT, metadata TEXT)')
        return Ledger(path)

    @contextlib.contextmanager
    def transaction(self) -> Any:
        """Serialize budget mutations in a rollback-safe SQLite transaction."""
        db=sqlite3.connect(self.path,timeout=30)
        try:
            db.execute('BEGIN IMMEDIATE');yield db;db.commit()
        except BaseException:
            db.rollback();raise
        finally: db.close()

    def snapshot(self) -> dict[str, Any]:
        """Read cumulative usage and outstanding reservations."""
        with self.transaction() as db: return self._snapshot(db)

    def _snapshot(self, db: Any) -> dict[str, Any]:
        """Account for settled usage and unresolved reservations in one transaction."""
        lim=strict_json(db.execute('SELECT payload FROM limits').fetchone()[0]); used={k:0.0 for k in lim}
        tickets=[]
        for row in db.execute('SELECT id,started,deadline,kind,reserved,actual,status,metadata FROM tickets'):
            rec=dict(zip(('id','started','deadline','kind','reserved','actual','status','metadata'),row))
            for k in ('reserved','metadata'):rec[k]=strict_json(rec[k])
            rec['actual']=strict_json(rec['actual']) if rec['actual'] else None
            amount=rec['actual'] if rec['actual'] is not None else rec['reserved']
            for k,v in amount.items():used[k]+=v
            tickets.append(rec)
        return {'limits':lim,'used':used,'remaining':{k:lim[k]-used[k] for k in lim},'tickets':tickets}

    def reserve(self, kind: str, amounts: Any, *, metadata: Any=None) -> tuple[str, float]:
        """Reserve bounded resources before a potentially charged operation."""
        require(kind in {'api','local','runpod'},'Unknown billing kind')
        require(amounts and all(k in {'seconds','usd','tokens','calls'} and type(v) in (int,float) and math.isfinite(v) and v>=0 for k,v in amounts.items()),'Invalid reservation')
        with self.transaction() as db:
            state=self._snapshot(db)
            for k,v in amounts.items(): require(v<=state['remaining'][k],f'Budget exhausted: {k}')
            if kind!='api':require(not any(x['kind']!='api' and x['status']!='settled' for x in state['tickets']),'Reconcile prior GPU allocation before another one')
            ticket=uuid.uuid4().hex;now=time.time(); deadline=now+amounts.get('seconds',3600)
            db.execute('INSERT INTO tickets VALUES (?,?,?,?,?,?,?,?)',(ticket,now,deadline,kind,canonical(amounts),None,'reserved',canonical(metadata or {})))
        return ticket,deadline

    def update(self, ticket: str, metadata: Any) -> None:
        """Attach reconciliation metadata to an outstanding reservation."""
        with self.transaction() as db:
            row=db.execute('SELECT metadata,status FROM tickets WHERE id=?',(ticket,)).fetchone()
            require(row and row[1]!='settled','Unknown/settled ticket')
            data=strict_json(row[0]);data.update(metadata)
            db.execute('UPDATE tickets SET metadata=? WHERE id=?',(canonical(data),ticket))

    def settle(self,ticket: str,actual: Any=None) -> None:
        """Record observed usage or retain the conservative reserved amount."""
        with self.transaction() as db:
            row=db.execute('SELECT reserved,status FROM tickets WHERE id=?',(ticket,)).fetchone()
            require(row and row[1]!='settled','Unknown/already settled ticket')
            reserved=strict_json(row[0]); actual=reserved if actual is None else actual
            require(actual.keys()==reserved.keys() and all(type(v) in (float,int) and math.isfinite(v) and v>=0 for v in actual.values()),'Invalid observed usage')
            db.execute('UPDATE tickets SET actual=?,status=? WHERE id=?',(canonical(actual),'settled',ticket))

def api_ledger() -> Ledger:
    """Open the explicitly configured API ledger."""
    p=os.environ.get('SA_API_BUDGET_DB');require(p,'SA_API_BUDGET_DB required')
    return Ledger(p)
