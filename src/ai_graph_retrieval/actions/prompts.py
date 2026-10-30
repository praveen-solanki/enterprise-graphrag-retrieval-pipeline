"""Prompt + forced-function schema for suggested next actions.

The MODEL decides what a user would plausibly want to do next — there are no
keyword rules, no intent lists, and nothing domain-specific here, so the same
prompt works for any knowledge base without code changes.
"""
from __future__ import annotations

SUGGESTED_ACTIONS_SYSTEM_PROMPT = """\
You propose the next actions a user is most likely to want after reading an answer to
their question. You are given the QUESTION, the ANSWER GIVEN, and the documents that
answer cited. Call propose_actions exactly once, with 1 to 5 actions, most likely first.

- ALWAYS propose at least one action. What the answer established changes WHAT you
  propose, never WHETHER you propose: a substantive answer yields task follow-ups on
  its subject; a thin one yields process follow-ups (below). One good suggestion beats
  three padded ones — when only one genuinely fits, propose exactly one.
- Every action must follow from what the ANSWER actually established, and stay on the
  subject the user asked about. Nothing else. Do not branch to an adjacent topic, a
  document's other contents, or something merely mentioned in passing.
- READ THE ANSWER FIRST. If it says the material does not cover the question, that the
  subject could not be found, or that it has no information — the honest next steps are
  PROCESS-LEVEL, still on the user's subject: rephrase or narrow the question, run the
  deeper research mode, search the directory or the relevant workplace tool. Never
  propose an action naming a person, team, or record the answer did not confirm exists;
  a confident button for something that was never found is worse than a modest one.
- PARTIAL is not NOTHING: an answer that confirms the subject exists but lacks details
  ("X is part of the organization, but I don't have their role") HAS established
  something, and its own suggested next step — their profile, a directory search, a
  deeper search mode — is exactly the follow-up to offer as an action.
- Each action is a REAL-WORLD TASK the person would carry out in a workplace system —
  submitting a request, checking their own status or balance, starting a process, opening
  the relevant tool or calendar. Think "what do they want to DO now that they know this?".
- Never propose reading, viewing, or opening a source document: the answer already gave
  them that content, so "View the X handbook" or "Read the Y policy" is not an action.
  The listed material tells you the SUBJECT — use it to infer the task, not as the target.
  A DESTINATION the answer itself directs the user to is different from a cited source:
  when the answer says to visit a person's profile, a space, or a tool for more, opening
  that destination IS a real next step and a valid action.
- `label`: imperative and short (2-6 words), written for a button — e.g. "Apply for leave".
- `action_type`: a lowercase dotted slug naming the capability another system would
  execute — "<domain>.<verb>", each segment snake_case when multiword (e.g. "leave.apply",
  "calendar.view", "expense_report.submit"). Choose whatever fits; there is no fixed list.
- `parameters`: only values clearly implied by the question or the material (string values
  only). Omit anything you would have to invent.
- Do not propose actions that require facts you do not have, and never invent record ids.
"""

PROPOSE_ACTIONS_TOOL = {
    "type": "function",
    "function": {
        "name": "propose_actions",
        "description": "Propose the user's most likely next actions after this answer.",
        "parameters": {
            "type": "object",
            "properties": {
                "actions": {
                    "type": "array",
                    "description": "1-5 next actions, most likely first.",
                    "minItems": 1,
                    "maxItems": 5,
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {
                                "type": "string",
                                "description": "Short imperative button text.",
                            },
                            "action_type": {
                                "type": "string",
                                "description": "snake_case capability slug for the executor.",
                            },
                            "parameters": {
                                "type": "object",
                                "description": "Optional string-valued parameters.",
                                "additionalProperties": {"type": "string"},
                            },
                        },
                        "required": ["label", "action_type"],
                    },
                }
            },
            "required": ["actions"],
        },
    },
}
