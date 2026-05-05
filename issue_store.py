from datetime import datetime

# In-memory issue store
_issues = []
_next_issue_id = 1

# Maps any activity_id → issue_id, not just the original source message.
# This allows reply chains to be traced back to an issue even if the reply
# is to a question or comment, not the original issue report message.
_activity_to_issue: dict[str, int] = {}


def _now_iso() -> str:
    return datetime.utcnow().isoformat()


def add_issue(
    description: str,
    raised_by: str,
    source_activity_id: str | None = None,
    conversation_id: str | None = None,
) -> dict:
    global _next_issue_id

    issue = {
        "id": _next_issue_id,
        "description": description,
        "raised_by": raised_by,
        "status": "open",           # only two states: "open" or "resolved"
        "resolution": None,
        "resolved_by": None,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "comments": [],
        "source_activity_id": source_activity_id,
        "conversation_id": conversation_id,
    }

    _issues.append(issue)

    if source_activity_id:
        _activity_to_issue[source_activity_id] = _next_issue_id

    _next_issue_id += 1
    return issue


def link_activity_to_issue(activity_id: str, issue_id: int) -> None:
    """Link any Teams message activity_id to an issue so future replies
    in this thread can be traced back to the issue."""
    _activity_to_issue[activity_id] = issue_id


def get_issue_by_any_activity_id(activity_id: str) -> dict | None:
    """Look up an issue by ANY activity_id linked to it,
    not just the original source message that created the issue."""
    issue_id = _activity_to_issue.get(activity_id)
    if issue_id is None:
        return None
    return get_issue_by_id(issue_id)


def get_open_issues() -> list:
    return [issue for issue in _issues if issue["status"] == "open"]


def get_resolved_issues() -> list:
    return [issue for issue in _issues if issue["status"] == "resolved"]


def get_issue_by_id(issue_id: int) -> dict | None:
    for issue in _issues:
        if issue["id"] == issue_id:
            return issue
    return None


def get_issue_by_source_activity_id(activity_id: str) -> dict | None:
    for issue in _issues:
        if issue.get("source_activity_id") == activity_id:
            return issue
    return None


def resolve_issue(issue_id: int, resolution: str, resolved_by: str) -> dict | None:
    issue = get_issue_by_id(issue_id)
    if not issue:
        return None

    issue["status"] = "resolved"
    issue["resolution"] = resolution
    issue["resolved_by"] = resolved_by
    issue["updated_at"] = _now_iso()
    return issue


def get_open_issues_by_sender(sender: str) -> list:
    """Return all open issues raised by a specific sender,
    sorted by created_at descending (most recent first)."""
    sender_issues = [
        issue for issue in _issues
        if issue["status"] == "open" and issue["raised_by"] == sender
    ]
    return sorted(sender_issues, key=lambda i: i["created_at"], reverse=True)


def append_issue_comment(issue_id: int, comment: str, author: str | None = None) -> dict | None:
    issue = get_issue_by_id(issue_id)
    if not issue:
        return None

    issue["comments"].append({
        "text": comment,
        "author": author,
        "timestamp": _now_iso(),
    })

    author_prefix = f"{author}: " if author else ""
    issue["description"] = f"{issue['description']}\nUpdate: {author_prefix}{comment}"
    issue["updated_at"] = _now_iso()
    return issue


def get_full_thread(issue: dict) -> str:
    """Build a full readable thread for an issue to pass to the LLM."""
    lines = [f"Issue #{issue['id']}: {issue['description']}"]
    for comment in issue.get("comments", []):
        author = comment.get("author") or "Unknown"
        lines.append(f"{author}: {comment['text']}")
    return "\n".join(lines)


def reset_issues():
    global _issues, _next_issue_id, _activity_to_issue
    _issues = []
    _next_issue_id = 1
    _activity_to_issue = {}
