import sys
from pathlib import Path


BUNDLE_ROOT = Path(__file__).resolve().parents[1] / "scripts" / "promptfoo_openwebui_eval"
if str(BUNDLE_ROOT) not in sys.path:
    sys.path.insert(0, str(BUNDLE_ROOT))

from lib.bundle_common import build_openwebui_user_prompt


def test_openwebui_prompt_includes_tool_file_and_kb_blocks():
    prompt = build_openwebui_user_prompt(
        {
            "request_prompt": "Summarize the supplied documents.",
            "tool_parameters_json": '{"emit_diagnostics": false}',
            "algorithm": "kohaku",
            "target_length": "long",
            "structure": "sectioned",
            "kb_ids_json": '["kb-lit-review"]',
        },
        file_ids=["file-1"],
    )

    assert "Summarize the supplied documents." in prompt
    assert "<tool_parameters>" in prompt
    assert "<algorithm>kohaku</algorithm>" in prompt
    assert "<target_length>long</target_length>" in prompt
    assert "<structure>sectioned</structure>" in prompt
    assert "<emit_diagnostics>false</emit_diagnostics>" in prompt
    assert "<files_list>" in prompt
    assert '"file-1"' in prompt
    assert "<kb_list>" in prompt
    assert '"kb-lit-review"' in prompt
