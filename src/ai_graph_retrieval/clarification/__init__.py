"""Clarifying-questions (human-in-the-loop) flow.

Ask the user targeted clarifying questions before answering, modelled as a durable
two-turn conversation over Redis (not an in-process pause), so it works on a stateless
multi-worker service. Turn 1 detects ambiguity + emits questions; turn 2 (`/resume`)
consumes the stored session and answers.
"""
