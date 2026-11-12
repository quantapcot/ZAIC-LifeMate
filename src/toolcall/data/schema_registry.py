"""Lossless access to official LifeMate catalog definitions."""
import copy
import json
from pathlib import Path


class ToolRegistry:
    def __init__(self, catalog):
        if not isinstance(catalog, dict) or not isinstance(catalog.get("apps"), list):
            raise ValueError("catalog must contain an apps list")
        self.version = catalog.get("schema_version")
        self._tools = {}
        self._groups = {}
        for app in catalog["apps"]:
            if not isinstance(app, dict) or not isinstance(app.get("id"), str) or not isinstance(app.get("tools"), list):
                raise ValueError("malformed catalog app")
            for tool in app["tools"]:
                if not isinstance(tool, dict) or not isinstance(tool.get("name"), str) or not tool["name"]:
                    raise ValueError("missing tool name")
                name = tool["name"]
                if name in self._tools:
                    raise ValueError(f"duplicate tool: {name}")
                schema = tool.get("parameters")
                if not isinstance(schema, dict) or schema.get("type") != "object" or not isinstance(schema.get("properties"), dict) or not isinstance(schema.get("required"), list):
                    raise ValueError(f"malformed parameter schema: {name}")
                if any(not isinstance(k, str) or k not in schema["properties"] for k in schema["required"]) or any(not isinstance(v, dict) for v in schema["properties"].values()):
                    raise ValueError(f"malformed parameter definitions: {name}")
                self._tools[name] = copy.deepcopy(tool)
                self._groups[name] = (app["id"], app.get("group"))
        if not self._tools:
            raise ValueError("catalog contains no tools")

    @classmethod
    def from_catalog(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def get_tool(self, name):
        return copy.deepcopy(self._tools[name])

    def get_schema(self, name):
        return self.get_tool(name)["parameters"]

    def list_tools(self, app=None, category=None):
        return sorted(name for name, (a, g) in self._groups.items() if (app is None or a == app) and (category is None or g == category))

    def arguments(self, name):
        schema = self.get_schema(name)
        required = set(schema["required"])
        return {"required": sorted(required), "optional": sorted(set(schema["properties"]) - required)}
