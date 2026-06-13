"""Pure reconstruction core — model-agnostic.

Depends ONLY on ports (app/core/ports). Must not import FastAPI, torch, or any
concrete model adapter. This is the hexagonal boundary (handover §3, §6).
"""
