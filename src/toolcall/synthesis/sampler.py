"""Sample only single-call targets from the verified training view."""
import random
from src.toolcall.data.contracts import GroundTruth, Metadata
from .schema_inspection import unsupported_constraints


class UnsupportedSchema(ValueError):
    def __init__(self, tool_name, issues):
        self.tool_name = tool_name
        self.issues = issues
        super().__init__(issues[0]['code'])


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
    issues = unsupported_constraints(schema)
    if issues:
        raise UnsupportedSchema(target, issues)
    candidates=[target]
    if scenario == 'multiple':
        app=target.split('.',1)[0]
        distractors=sorted(set(registry.list_tools(app=app)) & allowed - {target})
        if not distractors: raise ValueError('no_coherent_distractor')
        candidates.append(random.Random(seed).choice(distractors))
        random.Random(seed+1).shuffle(candidates)
    return GroundTruth(scenario,candidates,[target],Metadata(str(registry.version),'train_split','pending'))
