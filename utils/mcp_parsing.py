from __future__ import annotations

import json
from typing import Any, Dict


def parse_mcp_result(result: Any) -> Dict[str, Any]:
    if result is None:
        return {}

    if isinstance(result, dict):
        if result.get("type") == "text" and isinstance(result.get("text"), str):
            return json.loads(result["text"])

        if "text" in result and isinstance(result["text"], str):
            txt = result["text"].strip()
            if txt.startswith("{") or txt.startswith("["):
                return json.loads(txt)
        return result

    if isinstance(result, str):
        return json.loads(result)

    if isinstance(result, list) and result:
        first = result[0]
        if isinstance(first, dict):
            if first.get("type") == "text" and isinstance(first.get("text"), str):
                return json.loads(first["text"])
            if "text" in first and isinstance(first["text"], str):
                txt = first["text"].strip()
                if txt.startswith("{") or txt.startswith("["):
                    return json.loads(txt)
            return first

        if isinstance(first, str):
            return json.loads(first)

        if hasattr(first, "text"):
            return json.loads(first.text)

    if hasattr(result, "text"):
        return json.loads(result.text)

    raise ValueError(f"Unexpected MCP result type: {type(result)}")