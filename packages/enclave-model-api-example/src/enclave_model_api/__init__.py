"""Pre-packaged LLM inference pipeline for enclaves.

The inference docker image runs a small FastAPI server (see ``server.py``)
next to the regular enclave runner. Model weights arrive through syftbox
(shared by the model owner as a private dataset); inference request logs are
written to a dataset on the enclave's own datasite whose private data never
leaves the enclave.

``backend.py`` is the only module that imports model dependencies (the
DeepMind ``gemma`` package) — everything else stays importable in the plain
test environment.
"""
