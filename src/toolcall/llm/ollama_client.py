"""Local, non-streaming Ollama chat adapter; never follows redirects or proxies."""
import json
import time
from urllib import error, request
from urllib.parse import urlsplit

MODEL = "qwen3:4b-instruct-2507-q4_K_M"


class TransportError(RuntimeError): pass
class InvalidModelOutput(ValueError): pass


def _reject_constant(value):
    raise ValueError("invalid JSON constant")


def _check_output(value, schema):
    """Validate the JSON-schema subset used by our structured response envelopes."""
    kind = schema.get("type")
    predicates = {"object": lambda x: isinstance(x, dict), "array": lambda x: isinstance(x, list),
                  "string": lambda x: isinstance(x, str), "boolean": lambda x: type(x) is bool,
                  "integer": lambda x: type(x) is int, "number": lambda x: type(x) in (int, float)}
    if kind not in predicates or not predicates[kind](value):
        raise InvalidModelOutput("structured response has an invalid type")
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        if any(k not in value for k in schema.get("required", [])):
            raise InvalidModelOutput("structured response is missing a field")
        if schema.get("additionalProperties") is False and any(k not in properties for k in value):
            raise InvalidModelOutput("structured response has an unexpected field")
        for key, child in properties.items():
            if key in value: _check_output(value[key], child)
    if isinstance(value, list) and "items" in schema:
        for child in value: _check_output(child, schema["items"])
    if "enum" in schema and value not in schema["enum"]:
        raise InvalidModelOutput("structured response violates enum")


class NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise TransportError("local server returned a redirect")


class OllamaClient:
    def __init__(self, model=MODEL, base_url="http://127.0.0.1:11434", seed=42,
                 temperature=0.1, context_length=4096, max_tokens=512, timeout=180):
        url = urlsplit(base_url)
        if url.scheme != "http" or url.hostname not in ("127.0.0.1", "localhost", "::1") or url.username or url.password or url.path not in ("", "/") or url.query or url.fragment:
            raise ValueError("Ollama URL must be a local HTTP origin")
        if not model or not 1 <= context_length <= 16384 or max_tokens < 1 or timeout <= 0 or not 0 <= temperature <= 2:
            raise ValueError("invalid inference configuration")
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.options = {"seed": seed, "temperature": temperature, "num_ctx": context_length, "num_predict": max_tokens}
        self.timeout = timeout
        self.opener = request.build_opener(request.ProxyHandler({}), NoRedirect())

    def model_digest(self):
        try:
            with self.opener.open(self.base_url + "/api/tags", timeout=self.timeout) as response:
                payload = json.loads(response.read(4 * 1024 * 1024))
            for item in payload.get("models", []):
                if item.get("name") == self.model:
                    return item.get("digest")
        except (error.URLError, TimeoutError, OSError, ValueError, AttributeError):
            return None
        return None

    def generate_json(self, system_prompt, user_prompt, output_schema):
        payload = {"model": self.model, "stream": False, "format": output_schema,
                   "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], "options": self.options}
        req = request.Request(self.base_url + "/api/chat", data=json.dumps(payload, ensure_ascii=False).encode('utf-8'), headers={"Content-Type": "application/json"}, method="POST")
        start = time.perf_counter()
        try:
            with self.opener.open(req, timeout=self.timeout) as response:
                raw = response.read(4 * 1024 * 1024 + 1)
            if len(raw) > 4 * 1024 * 1024: raise InvalidModelOutput("response exceeds limit")
        except (error.URLError, TimeoutError, OSError) as exc:
            raise TransportError("Ollama request failed or timed out") from exc
        elapsed = time.perf_counter() - start
        try:
            envelope = json.loads(raw, parse_constant=_reject_constant)
            if not isinstance(envelope, dict) or envelope.get('done') is not True or not isinstance(envelope.get('message'), dict): raise ValueError()
            content = envelope['message'].get('content')
            if not isinstance(content, str): raise ValueError()
            result = json.loads(content, parse_constant=_reject_constant)
            if not isinstance(result, dict): raise ValueError()
        except (ValueError, UnicodeDecodeError, TypeError) as exc:
            raise InvalidModelOutput("malformed response or invalid JSON") from exc
        _check_output(result, output_schema)
        return result, {"latency_seconds": elapsed, "prompt_tokens": envelope.get('prompt_eval_count'), "output_tokens": envelope.get('eval_count'), "model": self.model, "options": dict(self.options)}
