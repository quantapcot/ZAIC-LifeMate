"""Lossless access to official LifeMate catalog definitions."""
import copy
import json
from pathlib import Path


def _check_schema(schema):
    if not isinstance(schema, dict):
        raise ValueError("schema must be an object")
    if "type" in schema:
        value = schema["type"]
        valid = {"object", "array", "string", "number", "integer", "boolean", "null"}
        if isinstance(value, str): value = [value]
        if not isinstance(value, list) or not value or any(not isinstance(x, str) or x not in valid for x in value):
            raise ValueError("invalid schema type")
    for field in ("description", "format"):
        if field in schema and not isinstance(schema[field], str): raise ValueError("invalid schema field")
    if "enum" in schema and not isinstance(schema["enum"], list): raise ValueError("invalid enum")
    if "properties" in schema:
        if not isinstance(schema["properties"], dict): raise ValueError("invalid properties")
        for name, child in schema["properties"].items():
            if not isinstance(name, str) or not name: raise ValueError("invalid property name")
            _check_schema(child)
    if "required" in schema:
        required = schema["required"]
        if not isinstance(required, list) or any(not isinstance(k, str) for k in required) or len(set(required)) != len(required): raise ValueError("invalid required")
        if any(k not in schema.get("properties", {}) for k in required): raise ValueError("unknown required property")
    if "items" in schema: _check_schema(schema["items"])
    for key in ("oneOf", "anyOf", "allOf"):
        if key in schema:
            if not isinstance(schema[key], list) or not schema[key]: raise ValueError("invalid schema alternatives")
            for child in schema[key]: _check_schema(child)


class ToolRegistry:
    def __init__(self, catalog):
        if not isinstance(catalog, dict) or not isinstance(catalog.get("apps"), list):
            raise ValueError("catalog must contain an apps list")
        self.version = catalog.get("schema_version")
        self._tools = {}
        self._groups = {}
        app_ids = set()
        for app in catalog["apps"]:
            if not isinstance(app, dict) or not isinstance(app.get("id"), str) or not isinstance(app.get("tools"), list):
                raise ValueError("malformed catalog app")
            if not app["id"].strip() or app["id"] in app_ids or ("group" in app and not isinstance(app["group"], str)):
                raise ValueError("invalid or duplicate application")
            app_ids.add(app["id"])
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
                _check_schema(schema)
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
