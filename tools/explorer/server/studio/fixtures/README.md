# Studio mock fixtures

Record/replay fixtures for the `mock` provider dialect (plan §7). One JSON file
per request fingerprint: `<model>__<fp16>.json`, holding the request echo and the
`ProviderResult` to replay.

- **Replay**: `ProviderClient("mock", ...)` reads the fixture whose fingerprint
  matches the incoming `ProviderRequest` and streams its text back deterministically.
- **Record**: any real dialect constructed with `record=True` writes its final
  result here as it streams — that captures one cheap live run into fixtures so
  the $0 test suite can replay it forever.

Fixtures are git-tracked (they are the test corpus). They contain no API keys —
keys are never part of a request fingerprint or a fixture payload.
