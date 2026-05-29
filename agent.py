"""
Invoice Reconciliation Agent

Reads a client email about a billing dispute, looks up invoice and contract
data through tool calls, reconciles the two, and drafts a response.
"""

import os
import json
import logging
from datetime import datetime, timezone

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# -- Mock tools ---------------------------------------------------------------
# These are the exact mock functions provided in the assessment.
# In production, these would call real billing and CRM APIs.

def get_invoice_details(invoice_id: str) -> dict:
    if invoice_id == "INV-999":
        return "ERROR 502: BILLING DATABASE TIMEOUT"

    return {"invoice_id": "INV-100", "total_billed": 500, "line_items": ["Platform Fee: $300", "Overage: $200"]}


def get_client_contract(client_email: str) -> dict:
    return {"client_email": "hello@acmecorp.com", "plan_type": "Pro", "monthly_platform_fee": 300, "overage_status": "Waived"}


TOOL_FUNCTIONS = {
    "get_invoice_details": get_invoice_details,
    "get_client_contract": get_client_contract,
}


# -- Tool schemas (for the Groq API) -----------------------------------------

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


# -- System prompt ------------------------------------------------------------

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
- If a tool returns an error or indicates a record was not found, acknowledge the failure honestly. Tell the client the issue is being escalated to the billing team for manual review. Do not guess or make up data.
- If the billing system times out or returns an error code, do not retry. Inform the client that there is a temporary system issue and their case has been flagged for follow-up.
- Do not assume anything about the contract that was not returned by get_client_contract.
- Be precise with dollar amounts."""


# -- Test email ---------------------------------------------------------------

TEST_EMAIL = "Hi, I am Daniel from Acme Corp. Why is invoice INV-100 for $500? We have overages waived on our Pro plan."


# -- Execution loop -----------------------------------------------------------

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

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
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
        "trace": trace,
    }


# -- Main ---------------------------------------------------------------------

def main():
    result = run_agent(TEST_EMAIL)

    print()
    print("AGENT RESPONSE")
    print("-" * 40)
    print(result["response"])

    base_dir = os.path.dirname(__file__)

    with open(os.path.join(base_dir, "execution_trace.json"), "w") as f:
        json.dump(result["trace"], f, indent=2)
    log.info("Wrote execution_trace.json")

    with open(os.path.join(base_dir, "raw_message_history.json"), "w") as f:
        json.dump(result["messages"], f, indent=2)
    log.info("Wrote raw_message_history.json")


if __name__ == "__main__":
    main()
