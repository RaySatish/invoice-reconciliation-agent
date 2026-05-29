"""
Invoice Reconciliation Agent

Reads a client email about a billing dispute, looks up invoice and contract
data through tool calls, reconciles the two, and drafts a response.
"""

import os
import json
import copy
import logging
from datetime import datetime, timezone

from dotenv import load_dotenv
from groq import Groq, BadRequestError, RateLimitError

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# Mock tools. These stand in for real billing and CRM APIs.

def get_invoice_details(invoice_id: str) -> dict:
    if invoice_id == "INV-999":
        return "ERROR 502: BILLING DATABASE TIMEOUT"

    # INV-100 is the only invoice that exists in this mock.
    if invoice_id == "INV-100":
        return {"invoice_id": "INV-100", "total_billed": 500, "line_items": ["Platform Fee: $300", "Overage: $200"]}

    # Any other ID is a genuine not-found: the lookup succeeded but no record matches.
    return {"status": "not_found", "invoice_id": invoice_id}


def get_client_contract(client_email: str) -> dict:
    return {"client_email": "hello@acmecorp.com", "plan_type": "Pro", "monthly_platform_fee": 300, "overage_status": "Waived"}


TOOL_FUNCTIONS = {
    "get_invoice_details": get_invoice_details,
    "get_client_contract": get_client_contract,
}


# The JSON schemas that tell the model which tools exist and how to call them.

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_invoice_details",
            "description": (
                "Retrieve the full details of an invoice given its ID. "
                "Returns the total billed and a breakdown of line items. "
                "Use this when the client references a specific invoice number."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "invoice_id": {
                        "type": "string",
                        "description": "The invoice identifier, e.g. 'INV-100'.",
                    },
                },
                "required": ["invoice_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_client_contract",
            "description": (
                "Retrieve contract details for a client given their email. "
                "Returns plan type, monthly fee, and overage status. "
                "Use this to verify what the client is entitled to."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "client_email": {
                        "type": "string",
                        "description": "The client's email address, e.g. 'hello@acmecorp.com'.",
                    },
                },
                "required": ["client_email"],
            },
        },
    },
]


# The system prompt: persona, the tools available, and the reconciliation steps.

SYSTEM_PROMPT = """You are an invoice reconciliation agent working in a billing support team. Your job is to investigate billing disputes raised by clients, determine whether discrepancies exist, and draft a clear, professional response.

You have access to two tools:

1. get_invoice_details(invoice_id) -- Retrieves the full breakdown of an invoice including total amount and line items.
2. get_client_contract(client_email) -- Retrieves the client's contract terms including plan type, monthly fee, and overage status.

When a client emails about a billing concern, follow these steps:

Step 1: Read the email carefully. Identify the client's name, company, the invoice they reference, and their specific concern.

Step 2: Call get_invoice_details with the invoice ID from the email.

Step 3: Call get_client_contract with the client's email address. If no email is provided explicitly, infer it from the company name (e.g. hello@companyname.com).

Step 4: Compare the invoice against the contract. Identify which charges match and which do not. Be specific.

Step 5: Draft a response to the client that:
- Addresses them by name.
- Acknowledges their concern.
- States exactly what you found -- which charges are correct, which are not, and why.
- If there is a billing error, confirms it and explains the corrective action.
- If there is no error, explains why the charges are valid.
- Maintains a respectful, helpful tone.

Constraints:
- Only use data returned by the tools. Never fabricate amounts, line items, or contract terms.
- If a tool returns a not-found result (for example {"status": "not_found"}), tell the client that no invoice matches the number they referenced and ask them to verify the invoice number. Do not escalate this case and do not invent a record.
- If a tool returns an error string or an error code (for example a database timeout), tell the client that billing is temporarily unavailable and that their case has been escalated to the billing team for manual review.
- In either case, never guess or fabricate invoice or contract data.
- If the billing system times out or returns an error code, do not retry. Inform the client that there is a temporary system issue and their case has been flagged for follow-up.
- Do not assume anything about the contract that was not returned by get_client_contract.
- Your final answer must contain only the customer-facing email itself, starting with the greeting (for example "Dear Daniel,"). Do not include any internal reasoning, status notes, or preamble such as "I will escalate this" before the email. Keep that reasoning out of the draft entirely.
- Be precise with dollar amounts."""


# The sample client email the agent works from.

TEST_EMAIL = "Hi, I am Daniel from Acme Corp. Why is invoice INV-100 for $500? We have overages waived on our Pro plan."


# Thin wrapper around the Groq call that retries on rate limits.

def create_with_retry(client, *, model, messages, tools, max_attempts=3):
    """
    Call Groq with a small retry budget.

    llama-3.3-70b-versatile sometimes emits a malformed tool-call token that Groq rejects
    with a 400 'tool_use_failed'. Generation is non-deterministic, so a retry
    usually clears it. Only that failure is retried; anything else is raised.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )
        except BadRequestError as err:
            if "tool_use_failed" in str(err) and attempt < max_attempts:
                log.warning(
                    "Model produced a malformed tool call (attempt %d/%d), retrying",
                    attempt,
                    max_attempts,
                )
                continue
            raise


# The agent loop: send the conversation, run any tool the model asks for, repeat
# until it returns a final answer.

def run_agent(email_body: str) -> dict:
    """
    Run the reconciliation agent on a given email.

    Returns a dict containing the final response, full message history,
    and a structured execution trace.
    """
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    model = "llama-3.3-70b-versatile"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Client email:\n\n{email_body}"},
    ]

    # Snapshot of the messages array exactly as it is sent to each API call.
    # Each entry is the literal input for one request, deep-copied so later
    # appends to `messages` do not mutate it after the fact.
    call_inputs = []

    trace = {
        "run_id": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "input_email": email_body,
        "steps": [],
        "token_usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
        "final_response": None,
    }

    log.info("Starting agent run")
    log.info("Input: %s", email_body)

    max_iterations = 10
    final_text = None

    for iteration in range(1, max_iterations + 1):
        log.info("Iteration %d", iteration)

        call_inputs.append(copy.deepcopy(messages))

        response = create_with_retry(
            client,
            model=model,
            messages=messages,
            tools=TOOLS,
        )

        choice = response.choices[0]
        message = choice.message

        if response.usage:
            trace["token_usage"]["prompt_tokens"] += response.usage.prompt_tokens
            trace["token_usage"]["completion_tokens"] += response.usage.completion_tokens
            trace["token_usage"]["total_tokens"] += response.usage.total_tokens

        # No tool calls means the model is done reasoning and has a final answer.
        if not message.tool_calls:
            final_text = message.content or ""
            trace["final_response"] = final_text
            trace["steps"].append({
                "iteration": iteration,
                "type": "final_response",
                "content": final_text,
            })
            messages.append({"role": "assistant", "content": final_text})
            log.info("Agent produced final response")
            break

        # The model wants to call one or more tools. Record the assistant
        # message, then execute each tool and feed results back.
        assistant_msg = {
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ],
        }
        messages.append(assistant_msg)

        for tool_call in message.tool_calls:
            fn_name = tool_call.function.name

            try:
                fn_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                fn_args = {}
                log.warning("Bad arguments for %s: %s", fn_name, tool_call.function.arguments)

            log.info("Tool call: %s(%s)", fn_name, fn_args)

            fn = TOOL_FUNCTIONS.get(fn_name)
            if fn is None:
                result = f"Unknown tool: {fn_name}"
                log.warning("Unknown tool requested: %s", fn_name)
            else:
                result = fn(**fn_args)
            result_str = json.dumps(result) if isinstance(result, dict) else str(result)
            log.info("Result: %s", result_str)

            trace["steps"].append({
                "iteration": iteration,
                "type": "tool_call",
                "function": fn_name,
                "arguments": fn_args,
                "result": result if isinstance(result, (dict, list)) else result_str,
            })

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result_str,
            })

    if final_text is None:
        log.warning("Hit iteration cap (%d) without a final response", max_iterations)
        trace["final_response"] = "[Agent did not produce a response within the iteration limit]"

    return {
        "response": trace["final_response"],
        "messages": messages,
        "call_inputs": call_inputs,
        "trace": trace,
    }


# Entry point.

def main():
    try:
        result = run_agent(TEST_EMAIL)
    except RateLimitError:
        # Groq free tier has a daily token cap. Exit cleanly rather than
        # dumping a traceback; there is nothing to retry against a daily limit.
        log.error("Groq rate limit reached. Wait for the quota to reset or upgrade the tier, then rerun.")
        return

    print()
    print("AGENT RESPONSE")
    print("-" * 40)
    print(result["response"])

    base_dir = os.path.dirname(__file__)

    with open(os.path.join(base_dir, "execution_trace.json"), "w") as f:
        json.dump(result["trace"], f, indent=2)
    log.info("Wrote execution_trace.json")

    with open(os.path.join(base_dir, "raw_message_history.json"), "w") as f:
        json.dump(result["call_inputs"], f, indent=2)
    log.info("Wrote raw_message_history.json")


if __name__ == "__main__":
    main()
