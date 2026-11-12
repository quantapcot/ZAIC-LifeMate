"""Conservative deterministic checks; semantic correctness remains unverified."""
import math
from src.toolcall.data.contracts import VerificationFeedback
from .errors import issue


def feedback(stage, issues, unverified=()):
    normalized = [item if isinstance(item, dict) else issue(item[1], item[0], retryable=not item[1].startswith('unsupported')) for item in issues]
    return VerificationFeedback(stage=stage, valid=not normalized, status='provisional' if not normalized else 'rejected',
        checks=[{'name': name, 'status':'not_verified'} for name in unverified],
        issues=normalized, retry_count=0, max_retries=0,
        failure_reasons=sorted({item['error_code'] for item in normalized}))


def validate_ground(ground, registry):
    errors = []
    names = set(registry.list_tools())
    candidates, targets = ground.candidate_tools, ground.target_tools
    if ground.scenario_type not in ('simple', 'multiple'): errors.append(('scenario_type', 'unsupported_scenario'))
    if len(targets) != 1: errors.append(('target_tools', 'invalid_call_count'))
    if not candidates or len(set(candidates)) != len(candidates): errors.append(('candidate_tools', 'invalid_candidates'))
    if any(x not in names for x in candidates + targets): errors.append(('tools', 'unknown_tool'))
    if not set(targets) <= set(candidates): errors.append(('target_tools', 'target_not_candidate'))
    if ground.scenario_type == 'simple' and len(candidates) != 1: errors.append(('candidate_tools','invalid_simple'))
    if ground.scenario_type == 'multiple' and len(candidates) < 2: errors.append(('candidate_tools','missing_distractor'))
    return feedback(1, errors, ('semantic_coherence', 'independence'))


def validate_query(query, ground):
    errors=[]
    if not isinstance(query, dict) or set(query) != {'user_query'} or not isinstance(query.get('user_query'), str) or not query['user_query'].strip(): errors.append(('user_query','invalid_query'))
    return feedback(2, errors, ('vietnamese_language', 'semantic_alignment', 'argument_grounding', 'training_query_copy'))


def _schema(value, schema, path, issues, unverified):
    typ = schema.get('type')
    checks = {'object': lambda x:isinstance(x, dict), 'array':lambda x:isinstance(x, list), 'string':lambda x:isinstance(x, str), 'integer':lambda x:type(x) is int, 'number':lambda x:type(x) in (int,float) and math.isfinite(x), 'boolean':lambda x:type(x) is bool, 'null':lambda x:x is None}
    types = typ if isinstance(typ, list) else [typ]
    if typ is not None and (any(t not in checks for t in types) or not any(checks[t](value) for t in types)):
        issues.append(issue('INVALID_ARGUMENT_TYPE', path, message='Expected ' + str(typ) + '; received ' + type(value).__name__ + '.')); return
    if 'enum' in schema and not any(type(value) is type(x) and value == x for x in schema['enum']): issues.append((path,'invalid_enum'))
    if type(value) in (int,float):
        for op in ('minimum','maximum','exclusiveMinimum','exclusiveMaximum'):
            if op in schema and not {'minimum':lambda a,b:a>=b,'maximum':lambda a,b:a<=b,'exclusiveMinimum':lambda a,b:a>b,'exclusiveMaximum':lambda a,b:a<b}[op](value,schema[op]): issues.append((path,'invalid_range'))
    if isinstance(value,dict):
        props = schema.get('properties', {})
        for key in sorted(value.keys() - props.keys()): issues.append((path+'.'+key,'unexpected_argument'))
        for key in schema.get('required', []):
            if key not in value or value[key] is None or value[key] == '' or value[key] == [] or value[key] == {}: issues.append((path+'.'+key,'missing_required'))
        for key in sorted(value.keys() & props.keys()): _schema(value[key], props[key], path+'.'+key, issues, unverified)
        groups=schema.get('x-one-of', [])
        if groups and not any(all(k in value and value[k] not in (None,'',[],{}) for k in group) for group in groups): issues.append((path,'x_one_of_unsatisfied'))
    if isinstance(value,list) and isinstance(schema.get('items'),dict):
        for i,item in enumerate(value): _schema(item,schema['items'],f'{path}[{i}]',issues,unverified)
    for key in ('x-ref','x-match','format','oneOf','anyOf','allOf'):
        if key in schema: unverified.add(key)


def validate_calls(raw, ground, registry):
    issues=[]; unverified={'semantic_alignment','argument_grounding','sdk_runtime_validation'}
    if not isinstance(raw, dict) or set(raw) != {'tool_calls'} or not isinstance(raw.get('tool_calls'),list):
        return feedback(3, [('tool_calls','invalid_structure')], unverified)
    calls=raw['tool_calls']
    if len(calls) != len(ground.target_tools): issues.append(('tool_calls','invalid_call_count'))
    for idx, call in enumerate(calls):
        if not isinstance(call,dict) or set(call) != {'name','arguments'} or not isinstance(call.get('name'),str) or not isinstance(call.get('arguments'),dict):
            issues.append((f'tool_calls[{idx}]','invalid_structure'));continue
        name=call['name']
        if name not in registry.list_tools(): issues.append((f'tool_calls[{idx}].name','unknown_tool'));continue
        if name not in ground.candidate_tools: issues.append((f'tool_calls[{idx}].name','not_candidate'));continue
        _schema(call['arguments'],registry.get_schema(name),f'tool_calls[{idx}].arguments',issues,unverified)
    if [c.get('name') if isinstance(c,dict) else None for c in calls] != ground.target_tools: issues.append(('tool_calls','target_mismatch'))
    return feedback(3, issues, sorted(unverified))
