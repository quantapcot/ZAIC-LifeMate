"""Fail-closed, versioned rubric for static single-action synthesis."""
POLICY_VERSION = 'lifemate-static-v1'
REQUIRED = {
    1: ('coherence','feasibility','distractor_relevance','no_state_dependency'),
    2: ('vietnamese_naturalness','intent_consistency','required_parameter_coverage','no_invented_state'),
    3: ('tool_selection','action_completeness','no_extra_actions','argument_consistency','no_invented_state'),
}
