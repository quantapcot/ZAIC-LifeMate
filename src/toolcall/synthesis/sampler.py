"""Sample only single-call targets from the verified training view."""
import random
from src.toolcall.data.contracts import GroundTruth, Metadata


def sample(record, registry, scenario, seed):
    if scenario not in ('simple','multiple'):
        raise ValueError('unsupported_scenario')
    calls=record['label']['tool_calls']
    if len(calls) != 1: raise ValueError('unsupported_train_example')
    target=calls[0]['name']
    installed = record['context'].get('installed_apps',{})
    allowed = {tool for app in installed.values() if isinstance(app,dict) for tool in app.get('tools',[]) if isinstance(tool,str)}
    if target not in registry.list_tools() or target not in allowed or target.startswith('sys.'):
        raise ValueError('unsupported_target')
    schema=registry.get_schema(target)
    # State-derived identifiers and conditional argument alternatives need later trajectory verification.
    if schema.get('x-one-of') or any(any(k in spec for k in ('x-ref','x-match')) for spec in schema['properties'].values()):
        raise ValueError('state_dependent_target')
    candidates=[target]
    if scenario == 'multiple':
        app=target.split('.',1)[0]
        distractors=sorted(set(registry.list_tools(app=app)) & allowed - {target})
        if not distractors: raise ValueError('no_coherent_distractor')
        candidates.append(random.Random(seed).choice(distractors))
        random.Random(seed+1).shuffle(candidates)
    return GroundTruth(scenario,candidates,[target],Metadata(str(registry.version),'train_split','pending'))
