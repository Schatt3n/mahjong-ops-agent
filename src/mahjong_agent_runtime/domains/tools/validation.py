"""Small JSON-schema subset used to validate model tool arguments."""

from __future__ import annotations

from typing import Any


def validate_schema(arguments: dict[str, Any], schema: dict[str, Any]) -> str | None:
    return validate_value("arguments", arguments, schema)


def validate_object(key: str, value: dict[str, Any], schema: dict[str, Any]) -> str | None:
    for required_key in schema.get("required") or []:
        if required_key not in value:
            return f"missing required argument: {required_key}"
    properties = schema.get("properties") or {}
    if schema.get("additionalProperties") is False:
        for item_key in value:
            if item_key not in properties:
                return f"unexpected argument: {item_key}"
    for item_key, item_value in value.items():
        prop = properties.get(item_key)
        if not prop:
            continue
        error = validate_value(item_key, item_value, prop)
        if error:
            return error
    return None


def validate_value(key: str, value: Any, schema: dict[str, Any]) -> str | None:
    expected = schema.get("type")
    if expected == "object" and not isinstance(value, dict):
        return f"{key} must be object"
    if expected == "object":
        return validate_object(key, value, schema)
    if expected == "array":
        if not isinstance(value, list):
            return f"{key} must be array"
        if "minItems" in schema and len(value) < int(schema["minItems"]):
            return f"{key} must contain at least {schema['minItems']} item(s)"
        if "maxItems" in schema and len(value) > int(schema["maxItems"]):
            return f"{key} must contain at most {schema['maxItems']} item(s)"
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                error = validate_value(f"{key}[{index}]", item, item_schema)
                if error:
                    return error
        return None
    if expected == "string":
        if not isinstance(value, str):
            return f"{key} must be string"
        if "minLength" in schema and len(value.strip()) < int(schema["minLength"]):
            return f"{key} must have length >= {schema['minLength']}"
        if "enum" in schema and value not in set(str(item) for item in schema["enum"]):
            return f"{key} must be one of: {', '.join(str(item) for item in schema['enum'])}"
        return None
    if expected == "boolean" and not isinstance(value, bool):
        return f"{key} must be boolean"
    if expected == "integer":
        if not isinstance(value, int):
            return f"{key} must be integer"
        if "minimum" in schema and value < int(schema["minimum"]):
            return f"{key} must be >= {schema['minimum']}"
        if "maximum" in schema and value > int(schema["maximum"]):
            return f"{key} must be <= {schema['maximum']}"
    if expected == "number":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return f"{key} must be number"
        if "minimum" in schema and float(value) < float(schema["minimum"]):
            return f"{key} must be >= {schema['minimum']}"
        if "maximum" in schema and float(value) > float(schema["maximum"]):
            return f"{key} must be <= {schema['maximum']}"
    return None
