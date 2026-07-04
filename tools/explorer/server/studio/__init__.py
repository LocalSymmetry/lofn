"""Lofn Creative Studio backend package (v2 of the Prompt Explorer).

The *experiment bench* half of the Two-Truths doctrine (plan §0): Canon is
production (Claude reading prose); Studio is the lab (the engine interpreting a
FlowSpec via provider APIs). Everything here is ADDITIVE — it never mutates
skills/** or vault/gates.yaml. The studio owns tools/explorer/flows/** (git) and
output/studio/** (engine).

WS0 froze the contracts this package is built around:
  schemas/flowspec.schema.json · schemas/knobs.schema.json · schemas/run.schema.json
  flowspec.py  — load/validate/version FlowSpecs + the canon importer
"""
