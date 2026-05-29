# Invoice Reconciliation Agent

A command-line agent that reads a client billing email, looks up the relevant invoice and contract
through tool calls, identifies any discrepancy between what was charged and what was agreed, and
drafts a professional reply. Built on a raw LLM SDK so the tool-calling loop, reasoning, and failure
handling are fully visible rather than hidden behind a framework.

## Project layout

```
invoice-reconciliation-agent/
|- agent.py                  Main script: tools, prompt, and the agent loop
|- requirements.txt          Python dependencies
|- .env.example              Template for your API key
|- README.md                 This file
|- execution_trace.json      Generated each run: the run narrative
|- raw_message_history.json  Generated each run: exact per-call API inputs
```

## Requirements

- Python 3.10 or newer
- A Groq API key (free tier works)

## Get a Groq API key

1. Sign in at https://console.groq.com/keys
2. Click **Create API Key**, name it, and copy the value (you only see it once)

## Setup

```bash
cd invoice-reconciliation-agent
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Copy the env template and paste in your key:

```bash
cp .env.example .env
```

Then open `.env` and set `GROQ_API_KEY` to the key you copied. The `.env` file is git-ignored and
must never be committed.

## Running

```bash
python agent.py
```

This feeds a hardcoded client email to the agent, runs the tool-calling loop, prints the drafted
reply, and regenerates the two trace files below.

### Trying the different paths

Behavior is driven by the invoice ID in the test email. Three IDs are wired into the mock:

| Invoice ID    | Mock returns                          | Agent should                                                |
|---------------|---------------------------------------|-------------------------------------------------------------|
| `INV-100`     | A valid invoice record                | Reconcile against the contract and explain the discrepancy  |
| `INV-999`     | `ERROR 502: BILLING DATABASE TIMEOUT` | Acknowledge the outage and escalate to billing for review   |
| anything else | `{"status": "not_found"}`             | Tell the client no invoice matches and ask them to verify   |

To exercise a different path, change the invoice number in the `TEST_EMAIL` string near the top of
`agent.py` and run again.

## Output files

- `execution_trace.json` -- the initial prompt, every tool call (name and arguments), each tool
  result, token usage, and the final drafted email. The readable narrative of the run.
- `raw_message_history.json` -- the exact message arrays sent to the model, one snapshot per API
  call, each deep-copied at the moment of the request. Because snapshots are taken right before each
  call, the final assistant reply (an output, never an input) does not appear. This is the literal
  wire input, not a reconstruction.

## Why the system prompt is structured the way it is

The prompt is split into ordered sections: persona, available tools, reconciliation steps, failure
handling, and output format.

- **Persona first.** A specific role ("a billing operations agent") anchors tone and keeps the final
  email out of generic chatbot phrasing.
- **Tools described in plain language.** The schema tells the model *how* to call a function; the
  prompt tells it *when*, which reduces answering from assumption instead of calling the tool.
- **Explicit reconciliation steps.** Spelling out "look up the invoice, look up the contract, compare
  them, explain the gap" turns a vague request into a checklist that produces a real comparison.
- **Two distinct failure cases.** A missing record and a backend outage are different problems: a
  not-found result asks the client to verify the number, while an error or timeout is escalated for
  manual review. Both share one hard rule -- never fabricate data to fill a gap.
- **Output format last.** The final answer must be only the customer-facing email; stating this at
  the end keeps it fresh when the model composes its reply.

The principle: enough structure to be reliable and safe, without scripting it so tightly that the
tool-calling reasoning stops being the model's own.

## Given more time: deploying as a shadow-mode pipeline

Shadow mode runs the agent against real traffic without letting it touch the customer. Every billing
email is processed, but output is reviewed internally instead of sent, so quality and safety can be
measured on real data before granting any autonomy.

1. **Replace the mock tools with real integrations.** `get_invoice_details` and `get_client_contract`
   call the real billing and CRM systems behind a thin, typed data-access layer with timeouts and
   retries, so a slow backend degrades gracefully.
2. **Trigger on real inbound email.** A queue worker picks up emails, runs the agent, and stores the
   draft as a suggestion on the ticket -- never auto-sent.
3. **Human-in-the-loop review.** Agents see the draft alongside the original email and the data
   pulled, then accept, edit, or reject. Every decision is logged as labeled data.
4. **Keep and extend the tracing.** The per-call message history and execution trace already form the
   observability backbone; in production they stream to a tracing backend so every run is auditable.
5. **Track metrics that matter.** Draft acceptance rate, edit distance, tool-call accuracy,
   hallucination incidents, and cost per ticket decide whether the agent can graduate.
6. **Graduate gradually.** Allow auto-send only for the lowest-risk cases (a clean invoice-contract
   match) while anything ambiguous stays with a human. Autonomy is earned per category, not all at
   once.
