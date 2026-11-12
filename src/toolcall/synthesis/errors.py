"""Stable, content-free issues shared by generation and deterministic checks."""
MESSAGES = {
    'TRANSPORT_FAILURE': 'Local inference transport failed.',
    'TIMEOUT': 'Local inference request timed out.',
    'INVALID_JSON': 'Model content is not valid JSON.',
    'MALFORMED_ENVELOPE': 'Model response envelope is malformed.',
    'OUTPUT_SCHEMA_MISMATCH': 'Model output does not match the response schema.',
    'INVALID_TOOL_NAME': 'Tool name is not available in the candidate catalog.',
    'UNEXPECTED_ARGUMENT': 'Argument is not declared by the tool schema.',
    'MISSING_REQUIRED_ARGUMENT': 'Required argument is missing or empty.',
    'INVALID_ARGUMENT_TYPE': 'Argument does not match the expected schema type.',
    'INVALID_ENUM': 'Argument is not an allowed enum value.',
    'INVALID_NUMERIC_RANGE': 'Argument violates a numeric bound.',
    'UNSUPPORTED_SCHEMA_CONSTRAINT': 'Schema constraint cannot be verified in this phase.',
    'TARGET_MISMATCH': 'Generated tool sequence differs from the selected target.',
    'QUERY_COPY': 'Generated query matches a normalized training query.',
}
ALIASES = {'unknown_tool': 'INVALID_TOOL_NAME', 'not_candidate':'INVALID_TOOL_NAME',
           'unexpected_argument':'UNEXPECTED_ARGUMENT', 'missing_required':'MISSING_REQUIRED_ARGUMENT',
           'invalid_type':'INVALID_ARGUMENT_TYPE', 'invalid_enum':'INVALID_ENUM',
           'invalid_range':'INVALID_NUMERIC_RANGE', 'target_mismatch':'TARGET_MISMATCH',
           'copies_training_query':'QUERY_COPY'}


def issue(code, field, *, source='deterministic_validator', retryable=True, message=None):
    code = ALIASES.get(code, code.upper())
    return {'error_code':code, 'field':field, 'message':message or MESSAGES.get(code, code.replace('_',' ').capitalize()+'.'),
            'retryable':retryable, 'source':source, 'reason':code}
