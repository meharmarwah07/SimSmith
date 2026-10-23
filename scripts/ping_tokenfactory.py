"""Standalone Token Factory check (stdlib only).

Usage:
  set -a; source .env; set +a
  python scripts/ping_tokenfactory.py              # lists models, filters to nemotron
  python scripts/ping_tokenfactory.py <model-id>   # also sends one test chat call
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

base = os.environ.get("TOKEN_FACTORY_BASE_URL", "").rstrip("/")
key = os.environ.get("TOKEN_FACTORY_API_KEY", "")
if not base or not key:
    sys.exit("Set TOKEN_FACTORY_BASE_URL and TOKEN_FACTORY_API_KEY first.")

HEADERS = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def call(path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(base + path, data=data, headers=HEADERS,
                                 method="POST" if payload is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        sys.exit(f"HTTP {e.code} on {path}: {e.read().decode(errors='ignore')[:500]}")
    except urllib.error.URLError as e:
        sys.exit(f"Network error on {path}: {e}")


models = call("/models")
ids = sorted(m.get("id", "") for m in models.get("data", []))
print(f"{len(ids)} models visible to this key.")
nemo = [i for i in ids if "nemotron" in i.lower()]
print("\nNemotron models:" if nemo else "\nNo model id contains 'nemotron'. All ids:")
for i in (nemo or ids):
    print("  ", i)

if len(sys.argv) > 1:
    model = sys.argv[1]
    t0 = time.time()
    out = call("/chat/completions", {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly: ok"}],
        "max_tokens": 20, "temperature": 0})
    print(f"\n{model} replied in {time.time() - t0:.1f}s:")
    print("  ", out["choices"][0]["message"]["content"])
    print("   usage:", out.get("usage"))
