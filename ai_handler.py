import json
import os
import httpx

OPENAI_API_URL = "https://api.openai.com/v1/responses"
MODEL = "gpt-5"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


async def _call_openai(prompt: str) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}",
    }

    payload = {
        "model": MODEL,
        "instructions": "Return only valid JSON. Do not include markdown fences or extra text.",
        "input": prompt,
        "max_output_tokens": 1200,
        "reasoning": {
            "effort": "minimal"
        },
        "text": {
            "format": {
                "type": "text"
            }
        }
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(
            OPENAI_API_URL,
            headers=headers,
            json=payload,
            timeout=30,
        )

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError:
            try:
                err = response.json()
            except Exception:
                err = response.text
            raise RuntimeError(f"OpenAI API error: {err}")

        data = response.json()

        if data.get("status") == "incomplete":
            raise RuntimeError(f"OpenAI response incomplete: {data}")

        if data.get("output_text"):
            return data["output_text"].strip()

        text_blocks = []
        for item in data.get("output", []):
            if item.get("type") == "message":
                for content in item.get("content", []):
                    if content.get("type") in ("output_text", "text") and "text" in content:
                        text_blocks.append(content["text"])

        if not text_blocks:
            raise RuntimeError(f"No text returned from OpenAI: {data}")

        return "\n".join(text_blocks).strip()


async def classify_message(text: str) -> dict:
    """
    Classify an incoming message as:
    - "issue"  → a new problem being reported
    - "update" → a follow-up, resolution, or status update on an existing issue

    "other" has been removed. Noise filtering is handled upstream by
    has_incident_signals() in agent.py before this function is called.

    IMPORTANT: The presence of a ticket ID (e.g. INC25177727) or device name
    always takes priority over the tone or phrasing of the message.
    A message that opens with a greeting or delegation phrase but contains
    incident identifiers must still be classified as "issue" or "update".
    """
    prompt = f"""Analyze this Teams chat message and classify its intent.

Message: "{text}"

Respond ONLY with a JSON object, no markdown, no explanation:
{{
  "type": "issue" | "update",
  "description": "concise one-line summary of the issue or update"
}}

Rules:
- "issue" = a new problem being reported, even if phrased as a request to check or review
- "update" = any follow-up, status update, resolution, or comment about an existing problem
- IMPORTANT: If a ticket ID (e.g. INC25177727) or device name is present, ALWAYS classify
  as "issue" or "update" — never let the opening phrase override this signal
- If the message mentions a ticket ID alongside a problem description, classify as "issue"
- If the message mentions a ticket ID alongside a resolution or status, classify as "update"

Examples of "issue":
- "INC25177727 traffic drain failed on RTCEEUZ22225B"
- "SH not triggering on device RTDPUS16390A"
- "Hi team please review INC25067637"
- "Jong, Yansen kindly check INC12345 OSPF cost 2000 tunnel299 costed up"
- "please check ticket INC25177727, drain failed"

Examples of "update":
- "Just checked, still not working"
- "INC25199999 has been fixed, tunnel is back up"
- "We're looking into it"
- "drain should have still went in, just need to check the script output"
- "INC12345 normalised successfully"
"""
    raw = await _call_openai(prompt)
    try:
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception as e:
        print("classify_message parse failed:", e)
        print("raw response:", raw)
        # Safe fallback — treat as issue so it's not silently dropped
        return {"type": "issue", "description": text}


async def classify_issue_status(thread: str) -> dict:
    """
    Given the full thread of an issue (original description + all comments
    including the latest message already appended), determine if the issue
    is resolved or still open.

    Fallback: if uncertain, classify as still open.

    Returns:
    {
        "status": "resolved" | "open",
        "note": "one-line summary of why",
        "confidence": 0-100
    }
    """
    prompt = f"""You are reviewing the full conversation thread of a tracked issue.
Determine whether the issue has been resolved or is still open.

Thread:
{thread}

Respond ONLY with a JSON object, no markdown, no explanation:
{{
  "status": "resolved" | "open",
  "note": "one-line summary of the current state",
  "confidence": <integer between 0 and 100>
}}

Rules:
- "resolved" means someone has explicitly confirmed the issue is fixed or working
- "open" means the issue is still being investigated, monitored, or has no confirmation of fix
- If you are not confident (confidence below 70), always return "open"
- When in doubt, return "open" — it is always safer to leave an issue open than to wrongly close it

Examples that lead to "resolved":
- "confirmed fixed, tunnel is back up"
- "INC25199999 normalised successfully"
- "working now, can close the ticket"

Examples that lead to "open":
- "still checking"
- "we're looking into it"
- "just checked, still not working"
- "okay noted"
- "drain should have still went in, just need to check the script output"
"""
    raw = await _call_openai(prompt)
    try:
        raw = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)
        # Safety net: if confidence is below 70, force open
        if result.get("confidence", 0) < 70:
            result["status"] = "open"
        return result
    except Exception as e:
        print("classify_issue_status parse failed:", e)
        print("raw response:", raw)
        return {"status": "open", "note": "Could not determine status", "confidence": 0}


async def find_duplicate(new_description: str, open_issues: list) -> dict:
    """
    Check if a newly reported issue is a duplicate of an existing open issue.
    Only matches on hard entity signals: same ticket ID or same device name.
    """
    if not open_issues:
        return {"is_duplicate": False, "matched_issue_id": None, "reason": "No open issues"}

    issues_text = "\n".join(
        [f"Issue #{i['id']} (raised by {i['raised_by']}): {i['description']}" for i in open_issues]
    )

    prompt = f"""You are checking if a new issue report is a duplicate of existing open issues.

New issue: "{new_description}"

Existing open issues:
{issues_text}

First extract the signature of the new issue:
- What ticket IDs are mentioned? (e.g. INC25177727)
- What device names are mentioned? (e.g. RTCEEUZ22225B)

Then compare strictly:
- ONLY mark as duplicate if the same ticket ID appears in both
- ONLY mark as duplicate if the same device name appears in both
- NEVER mark as duplicate based on shared keywords alone (e.g. "SH", "drain", "tunnel")
- If unsure, return is_duplicate: false — it is always safer to create a new issue

Respond ONLY with a JSON object, no markdown, no explanation:
{{
  "is_duplicate": true | false,
  "matched_issue_id": <issue id as integer, or null>,
  "reason": "brief explanation"
}}
"""
    raw = await _call_openai(prompt)
    try:
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception as e:
        print("find_duplicate parse failed:", e)
        print("raw response:", raw)
        return {"is_duplicate": False, "matched_issue_id": None, "reason": "Could not determine"}