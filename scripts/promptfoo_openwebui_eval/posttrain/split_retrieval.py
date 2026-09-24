"""Render a private Open WebUI retrieval tool with one immutable split scope."""
from __future__ import annotations

import re
from collections.abc import Mapping
from textwrap import dedent, indent

from .core import require, safe_id


def render_tool(
    knowledge_id: str,
    file_sources: Mapping[str, str],
    count: int = 3,
    max_output_chars: int = 6000,
    max_source_chars: int | None = None,
) -> str:
    """Bind native retrieval to uploaded file provenance without exposing IDs as inputs."""
    safe_id(knowledge_id)
    require(isinstance(file_sources, Mapping) and file_sources, 'A nonempty file/source mapping is required')
    sources: dict[str, str] = {}
    for file_id, marker in file_sources.items():
        safe_id(file_id)
        require(isinstance(marker, str) and re.fullmatch(r'vsrc_[0-9a-f]{20}', marker), 'Invalid source marker')
        sources[file_id] = marker
    require(len(set(sources.values())) == len(sources), 'Source markers must identify distinct uploaded pages')
    require(type(count) is int and 1 <= count <= 10, 'Retrieval count must be an integer from 1 to 10')
    require(type(max_output_chars) is int and 1 <= max_output_chars <= 100000, 'Invalid retrieval output character limit')
    require(max_source_chars is None or type(max_source_chars) is int and 1 <= max_source_chars <= 100000,
            'Invalid source reader output character limit')
    source = dedent(f'''\
        """Private retrieval tool. Scope and provenance are fixed at provisioning."""
        from __future__ import annotations

        import json
        import math
        import re
        from typing import Any, NoReturn

        from fastapi import Request
        from open_webui.tools.builtin import query_knowledge_files

        _KNOWLEDGE_ID = {knowledge_id!r}
        _FILE_SOURCES = {dict(sorted(sources.items()))!r}
        _MAX_COUNT = {count!r}
        _MAX_OUTPUT_CHARS = {max_output_chars!r}
        _MAX_QUERY_BYTES = 2048


        def _unique_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            """Reject ambiguous duplicate fields in a native tool response."""
            result: dict[str, Any] = {{}}
            for key, value in pairs:
                if key in result:
                    raise ValueError("Duplicate retrieval response field")
                result[key] = value
            return result


        def _reject_constant(value: str) -> NoReturn:
            """Reject non-finite values accepted by Python's permissive JSON decoder."""
            raise ValueError("Non-finite retrieval response value")


        def _encode_evidence(chunks: Any, count: int, max_chars: int) -> str:
            """Validate intact evidence and provenance for both native read operations."""
            if not isinstance(chunks, list) or not chunks or len(chunks) > count:
                raise RuntimeError("Native retrieval returned an error or an invalid evidence count")
            result: list[dict[str, Any]] = []
            for chunk in chunks:
                if not isinstance(chunk, dict):
                    raise RuntimeError("Native retrieval returned an invalid evidence chunk")
                file_id = chunk.get('file_id')
                if not isinstance(file_id, str) or file_id not in _FILE_SOURCES:
                    raise RuntimeError("Retrieved evidence is outside the assigned file scope")
                content = chunk.get('content')
                source = chunk.get('source')
                if not isinstance(content, str) or not content.strip() or not isinstance(source, str) or not source.strip():
                    raise RuntimeError("Retrieved evidence lacks content or source provenance")
                marker = _FILE_SOURCES[file_id]
                embedded = re.findall(r'^Source-ID: ([^\\s]+)\\s*$', content, flags=re.MULTILINE)
                if any(value != marker for value in embedded):
                    raise RuntimeError("Retrieved evidence has inconsistent source provenance")
                item: dict[str, Any] = {{'content': content, 'source': source, 'file_id': file_id, 'source_marker': marker}}
                if 'distance' in chunk:
                    distance = chunk['distance']
                    if type(distance) not in (int, float) or not math.isfinite(distance):
                        raise RuntimeError("Retrieved evidence has an invalid distance")
                    item['distance'] = distance
                result.append(item)
            output = json.dumps(result, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
            if len(output) > max_chars:
                raise RuntimeError("Retrieved evidence exceeds its output budget; no truncation was applied")
            return output


        class Tools:
            """Expose one search function with a fixed knowledge and file allowlist."""

            async def search(
                self,
                query: str,
                count: int = {count!r},
                __request__: Request | None = None,
                __user__: dict[str, Any] | None = None,
            ) -> str:
                """Search the assigned corpus and return intact evidence with stable source IDs.

                :param query: Nonempty search query, at most 2048 UTF-8 bytes.
                :param count: Maximum number of chunks, between 1 and {count}.
                :return: JSON evidence chunks with stable source markers.
                """
                if not isinstance(query, str) or not query.strip():
                    raise ValueError("Retrieval query must be nonempty text")
                try:
                    query_bytes = query.encode('utf-8')
                except UnicodeEncodeError:
                    raise ValueError("Retrieval query must contain valid Unicode") from None
                if len(query_bytes) > _MAX_QUERY_BYTES:
                    raise ValueError("Retrieval query exceeds its input budget")
                if type(count) is not int or not 1 <= count <= _MAX_COUNT:
                    raise ValueError("Retrieval count exceeds its allowed range")
                if __request__ is None or not isinstance(__user__, dict) or not __user__.get('id'):
                    raise ValueError("Authenticated retrieval context is required")
                try:
                    raw = await query_knowledge_files(
                        query=query,
                        knowledge_ids=[_KNOWLEDGE_ID],
                        count=count,
                        __request__=__request__,
                        __user__=__user__,
                        __model_knowledge__=[{{'type': 'collection', 'id': _KNOWLEDGE_ID}}],
                    )
                except Exception:
                    raise RuntimeError("Native retrieval failed; review the episode") from None
                if not isinstance(raw, str) or len(raw) > _MAX_OUTPUT_CHARS:
                    raise RuntimeError("Native retrieval response exceeds its output budget or has an invalid type")
                try:
                    chunks = json.loads(raw, object_pairs_hook=_unique_fields, parse_constant=_reject_constant)
                except (ValueError, TypeError):
                    raise RuntimeError("Native retrieval returned invalid JSON") from None
                return _encode_evidence(chunks, count, _MAX_OUTPUT_CHARS)
        ''')
    if max_source_chars is not None:
        source += indent(dedent(f'''

            async def read_source(
                self,
                source_marker: str,
                __request__: Request | None = None,
                __user__: dict[str, Any] | None = None,
            ) -> str:
                """Read one complete source page identified in search evidence, within the assigned corpus.

                :param source_marker: Exact stable source_marker returned by search.
                :return: Intact source page with provenance; oversized pages fail without truncation.
                """
                if not isinstance(source_marker, str) or source_marker not in _FILE_SOURCES.values():
                    raise ValueError("Source marker is outside the assigned file scope")
                if __request__ is None or not isinstance(__user__, dict) or not __user__.get('id'):
                    raise ValueError("Authenticated retrieval context is required")
                file_id = next(key for key, value in _FILE_SOURCES.items() if value == source_marker)
                try:
                    from open_webui.tools.builtin import view_knowledge_file
                    raw = await view_knowledge_file(file_id=file_id, __request__=__request__, __user__=__user__)
                except Exception:
                    raise RuntimeError("Native source read failed; review the episode") from None
                if not isinstance(raw, str) or len(raw) > {max_source_chars!r}:
                    raise RuntimeError("Native source response exceeds its output budget or has an invalid type")
                try:
                    page = json.loads(raw, object_pairs_hook=_unique_fields, parse_constant=_reject_constant)
                except (ValueError, TypeError):
                    raise RuntimeError("Native source reader returned invalid JSON") from None
                if not isinstance(page, dict) or 'error' in page or page.get('id') != file_id:
                    raise RuntimeError("Native source reader returned an error or mismatched file identity")
                return _encode_evidence([{{'file_id': file_id, 'content': page.get('content'),
                                          'source': page.get('filename')}}], 1, {max_source_chars!r})
            '''), '    ')
    return source
