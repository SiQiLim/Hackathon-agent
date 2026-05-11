from dotenv import load_dotenv
load_dotenv()

import re
from datetime import datetime, timezone

from microsoft_agents.hosting.core import (
    AgentApplication,
    TurnState,
    TurnContext,
    MemoryStorage,
)
from microsoft_agents.hosting.core.authorization import (
    AnonymousTokenProvider,
    ClaimsIdentity,
    AgentAuthConfiguration,
)
from microsoft_agents.hosting.aiohttp import CloudAdapter
from start_server import start_server
from issue_store import (
    add_issue,
    get_open_issues,
    get_resolved_issues,
    resolve_issue,
    get_issue_by_id,
    get_issue_by_source_activity_id,
    get_open_issues_by_sender,
    append_issue_comment,
    link_activity_to_issue,
    get_issue_by_any_activity_id,
    get_full_thread,
)
from ai_handler import classify_message, classify_issue_status, find_duplicate

RECENCY_HOURS = 24

# Keywords that signal a network/ops incident
INCIDENT_KEYWORDS = [
    "fail", "failed", "failing", "down", "outage", "error", "issue",
    "not working", "unreachable", "timeout", "drain", "tunnel", "ospf",
    "bgp", "ebgp", "cost", "normalise", "normalized", "check", "review",
    "trigger", "triggered", "alert", "incident", "inc", "ticket",
]


class LocalTokenProvider(AnonymousTokenProvider):
    async def get_access_token(self, resource_url: str, scopes: list, force_refresh: bool = False) -> str:
        return "local-dev-token"


class AnonymousConnections:
    def __init__(self):
        self._provider = LocalTokenProvider()

    def get_connection(self, connection_name: str):
        return self._provider

    def get_default_connection(self):
        return self._provider

    def get_token_provider(self, claims_identity: ClaimsIdentity, service_url: str):
        return self._provider

    def get_default_connection_configuration(self):
        return AgentAuthConfiguration()


connection_manager = AnonymousConnections()

AGENT_APP = AgentApplication[TurnState](
    storage=MemoryStorage(),
    adapter=CloudAdapter(connection_manager=connection_manager),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_issue(issue: dict) -> str:
    status = "✅ Resolved" if issue["status"] == "resolved" else "🔴 Open"
    lines = [
        f"**#{issue['id']}** — {issue['description']}",
        f"Raised by: {issue['raised_by']} | {status}",
    ]
    if issue["status"] == "resolved":
        lines.append(f"Fix: {issue['resolution']} (by {issue['resolved_by']})")
    return "\n".join(lines)


def get_sender_name(context: TurnContext) -> str:
    try:
        return context.activity.from_property.name or "Someone"
    except Exception:
        return "Someone"


def get_reply_to_id(context: TurnContext) -> str | None:
    activity = context.activity
    return getattr(activity, "reply_to_id", None) or getattr(activity, "replyToId", None)


def get_activity_id(context: TurnContext) -> str | None:
    return getattr(context.activity, "id", None)


def get_conversation_id(context: TurnContext) -> str | None:
    conversation = getattr(context.activity, "conversation", None)
    return getattr(conversation, "id", None) if conversation else None


def extract_ticket_id(text: str) -> str | None:
    """Extract a normalised ticket ID like INC25177727 from text."""
    lower = text.lower()
    m = re.search(r"\b(inc[- ]?\d+)\b", lower)
    if m:
        return m.group(1).replace(" ", "").replace("-", "").upper()
    m = re.search(r"\bticket\s*#?\s*([a-z]*\d+)\b", lower)
    if m:
        return m.group(1).replace("-", "").upper()
    return None


def extract_all_ticket_ids(text: str) -> list[str]:
    """Extract ALL ticket IDs from a message, not just the first one."""
    matches = re.findall(r"\b(inc[- ]?\d+)\b", text.lower())
    return [m.replace(" ", "").replace("-", "").upper() for m in matches]


def extract_device_name(text: str) -> str | None:
    """Extract device identifiers — uppercase alphanumeric, 8+ chars."""
    m = re.search(r"\b([A-Z]{2,}[A-Z0-9]{6,})\b", text)
    return m.group(1) if m else None


def hours_since(iso_timestamp: str) -> float:
    """Return how many hours have passed since an ISO timestamp."""
    try:
        created = datetime.fromisoformat(iso_timestamp)
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        return (now - created).total_seconds() / 3600
    except Exception:
        return 0.0


def extract_issue_reference(text: str) -> int | None:
    """Detect explicit 'issue #N' references in text."""
    m = re.search(r"issue\s*#(\d+)", text.lower())
    return int(m.group(1)) if m else None


def has_incident_signals(text: str) -> bool:
    """
    Pre-classification noise filter — replaces the old 'other' category.

    Returns True if the message contains at least one hard incident signal:
    - A ticket ID (INC12345)
    - A device name (RTCEEUZ22225B)
    - A known incident keyword

    Returns False for pure noise (greetings, thanks, questions with no
    incident context) so they are dropped before the LLM is even called.

    Examples that return False (dropped):
      "Hi team", "Thanks!", "Good morning", "👍", "Okay noted"

    Examples that return True (processed):
      "Hi team please review INC25067637"   ← ticket ID
      "OSPF fail on RTCEEUZ22225B"          ← device name + keyword
      "drain failed"                        ← keyword
      "tunnel is back up"                   ← keyword
    """
    # Hard signals — ticket ID or device name always wins
    if extract_all_ticket_ids(text):
        return True
    if extract_device_name(text):
        return True

    # Soft signals — known incident keywords
    lower = text.lower()
    for keyword in INCIDENT_KEYWORDS:
        if keyword in lower:
            return True

    return False


# ---------------------------------------------------------------------------
# Branch B: shortlist candidate issues for a no-reply_to_id message
# ---------------------------------------------------------------------------

def shortlist_candidate_issues(text: str, sender: str) -> dict:
    """
    Returns the best matching open issue for a message with no reply_to_id,
    or None if no confident match is found.

    Strategy:
    1. Get all open issues raised by this sender (sorted most recent first)
    2. Tier 1 — hard entity match (ticket ID or device name):
       - If found, use that issue regardless of age
    3. Tier 2 — soft match (most recent issue by sender within RECENCY_HOURS):
       - Only if no hard entity match
    4. No match → return None (treat as new issue)
    """
    sender_issues = get_open_issues_by_sender(sender)

    if not sender_issues:
        return {"issue": None, "match_type": None, "reason": "No open issues by this sender"}

    new_ticket = extract_ticket_id(text)
    new_device = extract_device_name(text)

    # Tier 1: hard entity match — ticket ID or device name
    for issue in sender_issues:
        issue_text = issue["description"]
        if new_ticket and extract_ticket_id(issue_text) == new_ticket:
            return {
                "issue": issue,
                "match_type": "entity",
                "reason": f"Ticket ID {new_ticket} matched Issue #{issue['id']}",
            }
        if new_device and extract_device_name(issue_text) == new_device:
            return {
                "issue": issue,
                "match_type": "entity",
                "reason": f"Device name {new_device} matched Issue #{issue['id']}",
            }

    # Tier 2: soft match — most recent issue by sender within recency window
    most_recent = sender_issues[0]
    age_hours = hours_since(most_recent["created_at"])

    if age_hours <= RECENCY_HOURS:
        return {
            "issue": most_recent,
            "match_type": "recency",
            "reason": f"Most recent open issue by {sender} is {age_hours:.1f}h old (within {RECENCY_HOURS}h window)",
        }

    return {
        "issue": None,
        "match_type": None,
        "reason": f"Most recent issue by {sender} is {age_hours:.1f}h old — too old for soft match, no entity match found",
    }


# ---------------------------------------------------------------------------
# Shared: apply resolved/open outcome to an issue
# ---------------------------------------------------------------------------

async def _apply_status(context: TurnContext, issue: dict, text: str, sender: str):
    """
    Append the new message to the issue thread, then ask the LLM whether
    the issue is resolved or still open. Act accordingly.
    """
    updated_issue = append_issue_comment(issue["id"], text, author=sender)
    if not updated_issue:
        return

    activity_id = get_activity_id(context)
    if activity_id:
        link_activity_to_issue(activity_id, issue["id"])

    thread = get_full_thread(updated_issue)
    try:
        result = await classify_issue_status(thread)
    except Exception as e:
        await context.send_activity(f"⚠️ Status classification failed: {str(e)}")
        return

    status = result.get("status", "open")
    note = result.get("note", "")

    if status == "resolved":
        resolved = resolve_issue(issue["id"], note, sender)
        if resolved:
            await context.send_activity(
                f"✅ **Issue #{resolved['id']}** has been resolved!\n"
                f"**Fix logged:** {note}\n\n"
                f"Use `/resolved` to see all resolved issues."
            )
    else:
        await context.send_activity(
            f"📝 **Issue #{updated_issue['id']}** updated, still open.\n"
            f"**Update:** {note}\n\n"
            f"Use `/issues` to see all open issues."
        )


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

async def _welcome(context: TurnContext, _: TurnState):
    await context.send_activity(
        "👋 **Issue Tracker Bot** is here!\n\n"
        "I automatically detect issues raised in this chat and track their resolutions.\n\n"
        "**Commands:**\n"
        "• `/issues` — list all open issues\n"
        "• `/resolved` — list resolved issues\n"
        "• `/issue <description>` — manually log an issue\n"
        "• `/resolve <id> <resolution>` — manually mark an issue as resolved\n"
        "• `/help` — show this message"
    )


async def _list_open(context: TurnContext, _: TurnState):
    issues = get_open_issues()
    if not issues:
        await context.send_activity("✅ No open issues right now!")
        return
    lines = ["**🔴 Open Issues:**\n"] + [format_issue(i) for i in issues]
    await context.send_activity("\n\n".join(lines))


async def _list_resolved(context: TurnContext, _: TurnState):
    issues = get_resolved_issues()
    if not issues:
        await context.send_activity("No resolved issues yet.")
        return
    lines = ["**✅ Resolved Issues:**\n"] + [format_issue(i) for i in issues]
    await context.send_activity("\n\n".join(lines))


async def _manual_issue(context: TurnContext, _: TurnState):
    text = context.activity.text or ""
    description = text[len("/issue"):].strip()
    if not description:
        await context.send_activity("Usage: `/issue <description>`")
        return

    sender = get_sender_name(context)
    open_issues = get_open_issues()

    dup = await find_duplicate(description, open_issues)
    if dup["is_duplicate"] and dup["matched_issue_id"]:
        original = get_issue_by_id(dup["matched_issue_id"])
        await context.send_activity(
            f"⚠️ This looks similar to **Issue #{original['id']}** raised by **{original['raised_by']}**:\n"
            f"> {original['description']}\n\n"
            f"Issue logged anyway as a new entry."
        )

    issue = add_issue(
        description=description,
        raised_by=sender,
        source_activity_id=get_activity_id(context),
        conversation_id=get_conversation_id(context),
    )
    await context.send_activity(f"📝 Logged **Issue #{issue['id']}**: {description}")


async def _manual_resolve(context: TurnContext, _: TurnState):
    text = context.activity.text or ""
    parts = text[len("/resolve"):].strip().split(" ", 1)
    if len(parts) < 2 or not parts[0].isdigit():
        await context.send_activity("Usage: `/resolve <id> <resolution description>`")
        return

    issue_id = int(parts[0])
    resolution = parts[1].strip()
    sender = get_sender_name(context)

    issue = resolve_issue(issue_id, resolution, sender)
    if not issue:
        await context.send_activity(f"❌ Issue #{issue_id} not found.")
        return

    await context.send_activity(
        f"✅ **Issue #{issue_id}** marked as resolved!\n"
        f"**Fix:** {resolution}"
    )


AGENT_APP.conversation_update("membersAdded")(_welcome)
AGENT_APP.message("/help")(_welcome)
AGENT_APP.message("/issues")(_list_open)
AGENT_APP.message("/resolved")(_list_resolved)
AGENT_APP.message("/issue")(_manual_issue)
AGENT_APP.message("/resolve")(_manual_resolve)


# ---------------------------------------------------------------------------
# Main message handler
# ---------------------------------------------------------------------------

@AGENT_APP.activity("message")
async def on_message(context: TurnContext, _: TurnState):
    text = context.activity.text or ""

    if text.startswith("/"):
        return

    sender = get_sender_name(context)
    activity_id = get_activity_id(context)
    reply_to_id = get_reply_to_id(context)

    # ----------------------------------------------------------------
    # Pre-classification noise filter (replaces "other" category)
    # Drop the message early if it has zero incident signals.
    # This avoids calling the LLM for greetings, thanks, and noise.
    # ----------------------------------------------------------------
    if not reply_to_id and not has_incident_signals(text):
        return   # pure noise — silently ignore

    # Classify the intent of the message (issue / update only — no "other")
    try:
        classification = await classify_message(text)
    except Exception as e:
        await context.send_activity(f"⚠️ AI classification failed: {str(e)}")
        return

    # ----------------------------------------------------------------
    # Branch A: message is a Teams reply (reply_to_id is known)
    # ----------------------------------------------------------------
    if reply_to_id:
        linked_issue = (
            get_issue_by_source_activity_id(reply_to_id)
            or get_issue_by_any_activity_id(reply_to_id)
        )

        if linked_issue and linked_issue["status"] == "open":
            await _apply_status(context, linked_issue, text, sender)
            return

    # ----------------------------------------------------------------
    # Branch B: no reply_to_id (or reply not linked to a known issue)
    # ----------------------------------------------------------------

    description = classification.get("description") or text

    # Step 1: explicit "issue #N" reference
    referenced_issue_id = extract_issue_reference(text)
    if referenced_issue_id is not None:
        referenced_issue = get_issue_by_id(referenced_issue_id)
        if referenced_issue and referenced_issue["status"] == "open":
            await _apply_status(context, referenced_issue, text, sender)
            return

    # Step 2: shortlist candidate issues by sender + entity + recency
    candidate = shortlist_candidate_issues(text, sender)

    if candidate["issue"] is not None:
        await _apply_status(context, candidate["issue"], text, sender)
        return

    # Step 3: no candidate match
    if classification["type"] == "update":
        await context.send_activity(
            "I detected a possible update but couldn't confidently match it to an open issue. "
            "If this relates to an existing issue, please reply directly to that message."
        )
        return

    # Step 4: duplicate check before creating new issue
    open_issues = get_open_issues()
    try:
        dup = await find_duplicate(description, open_issues)
    except Exception as e:
        await context.send_activity(f"⚠️ Duplicate check failed: {str(e)}")
        return

    if dup["is_duplicate"] and dup["matched_issue_id"]:
        original = get_issue_by_id(dup["matched_issue_id"])
        if original and original["status"] == "open":
            await _apply_status(context, original, text, sender)
            return

    # Brand new issue
    issue = add_issue(
        description=description,
        raised_by=sender,
        source_activity_id=activity_id,
        conversation_id=get_conversation_id(context),
    )
    if activity_id:
        link_activity_to_issue(activity_id, issue["id"])

    await context.send_activity(
        f"📝 Issue detected and logged as **#{issue['id']}**:\n"
        f"> {description}\n\n"
        f"Use `/issues` to see all open issues."
    )


@AGENT_APP.activity("installationUpdate")
async def on_installation_update(context: TurnContext, _: TurnState):
    pass


if __name__ == "__main__":
    try:
        print("Starting Issue Tracker Bot on http://localhost:3978")
        start_server(AGENT_APP, None)
    except Exception as error:
        print("FATAL ERROR:", repr(error))
        raise