"""Inspect schema-bearing nodes without modifying official definitions."""


def unsupported_constraints(schema):
    """Return deterministic unsupported codes and schema paths; do not inspect data values."""
    found = []
    direct = {'x-ref': 'UNSUPPORTED_X_REF', 'x-match': 'UNSUPPORTED_X_MATCH',
              'x-one-of': 'UNSUPPORTED_CONDITIONAL_SCHEMA',
              'oneOf': 'UNSUPPORTED_CONDITIONAL_SCHEMA', 'anyOf': 'UNSUPPORTED_CONDITIONAL_SCHEMA',
              'allOf': 'UNSUPPORTED_CONDITIONAL_SCHEMA', 'if': 'UNSUPPORTED_CONDITIONAL_SCHEMA',
              'then': 'UNSUPPORTED_CONDITIONAL_SCHEMA', 'else': 'UNSUPPORTED_CONDITIONAL_SCHEMA',
              'not': 'UNSUPPORTED_CONDITIONAL_SCHEMA', 'dependentSchemas': 'UNSUPPORTED_CONDITIONAL_SCHEMA'}
    maps = ('properties', 'patternProperties', '$defs', 'definitions', 'dependentSchemas')
    singles = ('items', 'additionalProperties', 'contains', 'propertyNames', 'if', 'then', 'else', 'not')
    arrays = ('oneOf', 'anyOf', 'allOf', 'prefixItems')
    def walk(node, path):
        if not isinstance(node, dict): return
        for key, code in direct.items():
            if key in node: found.append({'code': code, 'path': path + '.' + key})
        for key in maps:
            value = node.get(key)
            if isinstance(value, dict):
                for name in sorted(value): walk(value[name], path + '.' + key + '.' + name)
        for key in singles:
            value = node.get(key)
            if isinstance(value, dict): walk(value, path + '.' + key)
            elif key == 'items' and isinstance(value, list):
                for i, child in enumerate(value): walk(child, f'{path}.{key}[{i}]')
        for key in arrays:
            if isinstance(node.get(key), list):
                for i, child in enumerate(node[key]): walk(child, f'{path}.{key}[{i}]')
    walk(schema, '$')
    return sorted(found, key=lambda issue: (issue['code'], issue['path']))


def catalog_eligibility(registry):
    """Counts over catalog schemas, not over eligible train records."""
    from collections import Counter
    reasons = Counter()
    eligible = 0
    for name in registry.list_tools():
        issues = unsupported_constraints(registry.get_schema(name))
        if issues:
            reasons.update(set(issue['code'] for issue in issues))
        else:
            eligible += 1
    return {'catalog_tools': len(registry.list_tools()), 'schema_eligible': eligible,
            'excluded_by_reason': dict(sorted(reasons.items()))}
