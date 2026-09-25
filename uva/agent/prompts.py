"""Fixed prompts of the consultation channel: the expert's system prompt, the probe's instruction and
elicitation system message, and the observations returned in place of an answer.
"""

CLOUD_SYSTEM = (
    "You are a senior software engineer. A colleague is working on a Python bug on a machine "
    "you cannot see, and has sent you one question. Answer it directly and concretely: give "
    "your best account of the cause and the one next thing they should check or change. If "
    "the question does not carry enough information, say exactly what you would need. Be "
    "brief. No pleasantries."
)

FORCE_INSTRUCTION = (
    "[HARNESS: MANDATORY EXPERT CONSULTATION]\n"
    "You are being given one consultation with a senior engineer who cannot see your terminal "
    "or the repository. Write the request you would send them: state what you are trying to "
    "establish, what you have already ruled out, and the single thing you need judged. "
    "Output ONLY the request text. Do not output a shell command."
)

CONSULT_SYSTEM = (
    "You are pausing your own work to consult a senior engineer who cannot see your terminal, "
    "run anything, or read your files. For THIS turn ONLY, you are writing a message to that "
    "engineer, NOT acting on the machine. Do not run a command. Do not emit a command block or "
    "any triple-backtick fence. Reply with a short plain-text question: what you are trying to "
    "establish, what you have already ruled out, and the one thing you need judged. Prose only."
)

STUB = (
    "<ask_expert>\n"
    "Your question was recorded by the local harness.\n"
    "No reply is available in this run -- the engineer is not reachable from this machine, "
    "so do not wait for one. Continue working locally.\n"
    "</ask_expert>"
)
EXHAUSTED = (
    "<ask_expert>\n"
    "You have used all of your questions for this task. Continue working locally.\n"
    "</ask_expert>"
)
SUPPRESSED = (
    "<ask_expert>\n"
    "The expert channel is closed for this task. No consultation is available. Continue "
    "working locally.\n"
    "</ask_expert>"
)
