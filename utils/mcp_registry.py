from __future__ import annotations
import os
print(os.getenv("OPENAI_API_KEY"))
import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict

from langchain_mcp_adapters.client import MultiServerMCPClient


@dataclass
class ToolRegistry:
    client: MultiServerMCPClient
    tools_cache: Dict[str, Any] = field(default_factory=dict)
    _init_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def ensure_discovered(self, timeout: float = 600.0) -> None:
        if self.tools_cache:
            return
        async with self._init_lock:
            if self.tools_cache:
                return
            tools = await asyncio.wait_for(self.client.get_tools(), timeout=timeout)
            for t in tools:
                self.tools_cache[t.name] = t
            if not self.tools_cache:
                raise RuntimeError("No tools discovered from MCP servers.")

    async def get_tool(self, target_name: str) -> Any:
        await self.ensure_discovered()

        if target_name in self.tools_cache:
            return self.tools_cache[target_name]

        for full_name, tool_obj in self.tools_cache.items():
            if full_name.endswith(f"_{target_name}") or full_name.endswith(f"-{target_name}"):
                return tool_obj

        for full_name, tool_obj in self.tools_cache.items():
            if target_name in full_name:
                return tool_obj

        raise ValueError(f"Tool '{target_name}' not found. Available: {list(self.tools_cache.keys())}")