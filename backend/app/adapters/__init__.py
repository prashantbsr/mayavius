"""Driven adapters — concrete model integrations implementing ReconstructionPort.

Each adapter is the ONLY place a specific model/SDK is imported. The core never
imports these. The MVP ships at least one working adapter; the default combo is
locked by the model-selection task (handover §6) and recorded in
spec/03-decisions-locked.md + decisions/decision-log.md.
"""
