"""Local, non-streaming Ollama chat adapter; never follows redirects or proxies."""
import json
import time
import socket
import math
from urllib import error, request
from urllib.parse import urlsplit

MODEL = "qwen3:4b-instruct-2507-q4_K_M"


class TransportError(RuntimeError):
    def __init__(self, message, code="TRANSPORT_FAILURE"):
        super().__init__(message)
        self.code = code
        self.field = "request"
        self.raw_output = None


class InvalidModelOutput(ValueError):
    def __init__(self, message, code="OUTPUT_SCHEMA_MISMATCH", field="response", raw_output=None):
        super().__init__(message)
        self.code = code
        self.field = field
        self.raw_output = raw_output



class PromptBudgetError(ValueError):
    def __init__(self, budget):
        super().__init__("PROMPT_TOO_LARGE")
        self.budget = dict(budget)


def _reject_constant(value):
    raise ValueError("invalid JSON constant")


def _check_output(value, schema, path="$"):
    """Validate the JSON-schema subset used by our structured response envelopes."""
    kind = schema.get("type")
    predicates = {"object": lambda x: isinstance(x, dict), "array": lambda x: isinstance(x, list),
                  "string": lambda x: isinstance(x, str), "boolean": lambda x: type(x) is bool,
                  "integer": lambda x: type(x) is int, "number": lambda x: type(x) in (int, float)}
    if kind not in predicates or not predicates[kind](value):
        raise InvalidModelOutput("structured response has an invalid type", field=path)
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        missing = [k for k in schema.get("required", []) if k not in value]
        if missing:
            raise InvalidModelOutput("structured response is missing a field", field=path + "." + missing[0])
        extra = sorted(k for k in value if k not in properties) if schema.get("additionalProperties") is False else []
        if extra:
            raise InvalidModelOutput("structured response has an unexpected field", field=path + "." + extra[0])
        for key, child in properties.items():
            if key in value: _check_output(value[key], child, path + "." + key)
    if isinstance(value, list) and "items" in schema:
        for i, child in enumerate(value): _check_output(child, schema["items"], f"{path}[{i}]")
    if "enum" in schema and value not in schema["enum"]:
        raise InvalidModelOutput("structured response violates enum", field=path)


class NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise TransportError("local server returned a redirect")


class OllamaClient:
    def __init__(self, model=MODEL, base_url="http://127.0.0.1:11434", seed=42,
                 temperature=0.1, context_length=4096, max_tokens=512, timeout=180, keep_alive=None, think=None, max_payload_bytes=262144, token_estimate_factor=1.0, confirm_unloaded=False):
        url = urlsplit(base_url)
        if url.scheme != "http" or url.hostname not in ("127.0.0.1", "localhost", "::1") or url.username or url.password or url.path not in ("", "/") or url.query or url.fragment:
            raise ValueError("Ollama URL must be a local HTTP origin")
        if not model or not 1 <= context_length <= 16384 or max_tokens < 1 or timeout <= 0 or not 0 <= temperature <= 2:
            raise ValueError("invalid inference configuration")
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.options = {"seed": seed, "temperature": temperature, "num_ctx": context_length, "num_predict": max_tokens}
        self.timeout = timeout
        self.keep_alive = keep_alive
        self.think = think
        self.confirm_unloaded = confirm_unloaded
        if max_payload_bytes < 1024 or not math.isfinite(token_estimate_factor) or token_estimate_factor <= 0 or max_tokens + 128 >= context_length:
            raise ValueError("invalid prompt budget configuration")
        self.max_payload_bytes = max_payload_bytes
        self.token_estimate_factor = token_estimate_factor
        self.last_budget = None
        self._model_digest = None
        self.opener = request.build_opener(request.ProxyHandler({}), NoRedirect())

    def model_digest(self):
        try:
            with self.opener.open(self.base_url + "/api/tags", timeout=self.timeout) as response:
                payload = json.loads(response.read(4 * 1024 * 1024))
            for item in payload.get("models", []):
                if item.get("name") == self.model:
                    self._model_digest = item.get("digest")
                    return self._model_digest
        except (error.URLError, TimeoutError, OSError, ValueError, AttributeError):
            return None
        return None

    def assert_unloaded(self):
        """Fail closed if the server does not confirm that this model is unloaded."""
        for _ in range(20):
            try:
                with self.opener.open(self.base_url + "/api/ps", timeout=self.timeout) as response:
                    payload=json.loads(response.read(1024*1024))
                if not isinstance(payload,dict) or not isinstance(payload.get('models'),list): raise ValueError()
                if not any(m.get('name')==self.model or m.get('model')==self.model or (self._model_digest and m.get('digest')==self._model_digest) for m in payload['models']): return
            except (error.URLError,TimeoutError,OSError,ValueError,TypeError,AttributeError) as exc:
                raise TransportError('Unable to confirm model unload',code='UNLOAD_UNCONFIRMED') from exc
            time.sleep(0.2)
        raise TransportError('Model did not unload',code='UNLOAD_UNCONFIRMED')

    def unload(self):
        """Ask the local Ollama server to unload this exact model."""
        payload = json.dumps({"model":self.model,"keep_alive":0,"prompt":"","stream":False}).encode("utf-8")
        req = request.Request(self.base_url + "/api/generate", data=payload, headers={"Content-Type":"application/json"}, method="POST")
        try:
            with self.opener.open(req, timeout=self.timeout) as response:
                response.read(1024 * 1024)
            if self.confirm_unloaded: self.assert_unloaded()
        except (error.URLError, TimeoutError, OSError) as exc:
            raise TransportError("Ollama unload failed") from exc

    def generate_json(self, system_prompt, user_prompt, output_schema):
        payload = {"model": self.model, "stream": False, "format": output_schema,
                   "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], "options": self.options}
        if self.keep_alive is not None: payload["keep_alive"] = self.keep_alive
        if self.think is not None: payload["think"] = self.think
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        # Conservative heuristic, not a tokenizer: count Unicode characters in messages
        # and schema as at most one estimated token each, plus fixed chat overhead.
        prompt_content = json.dumps({"messages":payload["messages"],"format":output_schema},ensure_ascii=False,separators=(",",":"))
        estimated = math.ceil(len(prompt_content) * self.token_estimate_factor) + 128
        available = self.options['num_ctx'] - self.options['num_predict']
        self.last_budget = {"estimated_input_tokens":estimated,"reserved_output_tokens":self.options['num_predict'],
                            "context_tokens":self.options['num_ctx'],"payload_bytes":len(encoded),
                            "max_payload_bytes":self.max_payload_bytes,"estimation":"unicode_codepoints_plus_overhead"}
        if len(encoded) > self.max_payload_bytes or estimated > available:
            raise PromptBudgetError(self.last_budget)
        req = request.Request(self.base_url + "/api/chat", data=encoded, headers={"Content-Type": "application/json"}, method="POST")
        start = time.perf_counter()
        try:
            with self.opener.open(req, timeout=self.timeout) as response:
                raw = response.read(4 * 1024 * 1024 + 1)
            if self.confirm_unloaded and self.keep_alive == 0: self.assert_unloaded()
            if len(raw) > 4 * 1024 * 1024: raise InvalidModelOutput("response exceeds limit", code="RESPONSE_TOO_LARGE")
        except error.HTTPError as exc:
            try:
                detail = exc.read(4096).decode("utf-8", errors="replace").casefold()
            except OSError:
                detail = ""
            code = "OUT_OF_MEMORY" if "out of memory" in detail or "cuda malloc" in detail else "TRANSPORT_FAILURE"
            raise TransportError("Ollama HTTP request failed", code=code) from exc
        except (error.URLError, TimeoutError, OSError) as exc:
            reason = exc.reason if isinstance(exc, error.URLError) else exc
            code = "TIMEOUT" if isinstance(reason, (TimeoutError, socket.timeout)) else "TRANSPORT_FAILURE"
            raise TransportError("Ollama transport failed", code=code) from exc
        elapsed = time.perf_counter() - start
        try:
            envelope = json.loads(raw, parse_constant=_reject_constant)
            if not isinstance(envelope, dict) or envelope.get('done') is not True or not isinstance(envelope.get('message'), dict):
                raise ValueError()
            content = envelope['message'].get('content')
            if not isinstance(content, str): raise ValueError()
        except (ValueError, UnicodeDecodeError, TypeError) as exc:
            raise InvalidModelOutput("malformed response envelope", code="MALFORMED_ENVELOPE", field="response") from exc
        try:
            result = json.loads(content, parse_constant=_reject_constant)
        except (ValueError, TypeError) as exc:
            raise InvalidModelOutput("model content is invalid JSON", code="INVALID_JSON", field="message.content", raw_output=content[:4096]) from exc
        try:
            _check_output(result, output_schema)
        except InvalidModelOutput as exc:
            exc.raw_output = content[:4096]
            raise
        return result, {"latency_seconds": elapsed, "prompt_tokens": envelope.get('prompt_eval_count'), "output_tokens": envelope.get('eval_count'), "model": self.model, "options": dict(self.options), "input_budget": dict(self.last_budget)}
