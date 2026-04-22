"""Generate bug_scrub.md from Issues sheet + action_Type classification."""
from __future__ import annotations

import re
import textwrap
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

import openpyxl

REPO_ROOT = Path(__file__).resolve().parents[7]
EXCEL = REPO_ROOT / "opencode/issue_triage/result/torch_xpu_ops_issues.xlsx"
OUT   = REPO_ROOT / "opencode/issue_triage/result/bug_scrub.md"
REPO  = "intel/torch-xpu-ops"
TODAY = datetime(2026, 4, 21, tzinfo=timezone.utc)
RECENT_CUTOFF = TODAY - timedelta(days=7)

# display order for the two macro-groups
DEV_SECTIONS = ["NEED_ACTION", "NEEDS_OWNER", "TRACK_PR", "IMPLEMENT",
                "RETRIAGE_PRS", "ROOT_CAUSE"]
QA_SECTIONS  = ["CLOSE", "VERIFY_AND_CLOSE", "AWAIT_REPLY", "SKIP",
                "MONITOR", "NOT_TARGET_CLOSE", "CHECK_CASES"]

# priority ordering for "primary category" selection when a row has combos
PRIMARY_ORDER = [
    "CLOSE", "NOT_TARGET_CLOSE", "VERIFY_AND_CLOSE", "TRACK_PR",
    "IMPLEMENT", "RETRIAGE_PRS", "WAIT_EXTERNAL",
    "ROOT_CAUSE", "FILE_ISSUE", "MONITOR",
    "NEEDS_OWNER", "NEED_ACTION",
    "AWAIT_REPLY", "CHECK_CASES", "SKIP",
]

SECTION_TITLES = {
    "NEED_ACTION":      "NEED_ACTION — no PR and no decision; owner must start investigation",
    "NEEDS_OWNER":      "NEEDS_OWNER — awaiting triage-lead to assign an owner",
    "TRACK_PR":         "TRACK_PR — identified PR is open; wait for / push to merge",
    "IMPLEMENT":        "IMPLEMENT — new code / new PR must be written",
    "RETRIAGE_PRS":     "RETRIAGE_PRS — prior PRs dead or cross-refs unverified; re-evaluate path",
    "ROOT_CAUSE":       "ROOT_CAUSE — owner actively debugging this specific failure",
    "CLOSE":            "CLOSE — terminal close (CI passing, duplicate, or confirmed gap acceptable)",
    "VERIFY_AND_CLOSE": "VERIFY_AND_CLOSE — fix merged; validate then close",
    "AWAIT_REPLY":      "AWAIT_REPLY — open questions in thread; owner must respond",
    "SKIP":             "SKIP — labeled not-target/wontfix at intake",
    "MONITOR":          "MONITOR — long-running tracker / maintenance / scoping",
    "NOT_TARGET_CLOSE": "NOT_TARGET_CLOSE — authoritative not-target decision (full or partial)",
    "CHECK_CASES":      "CHECK_CASES — XPU test case missing in repo; QA must verify case existence before action",
    "UNCLASSIFIED":     "UNCLASSIFIED — Phase 4b produced no verdict; needs manual triage",
}

PRIO_RANK = {"P0": 0, "P1": 1, "P2": 2, "P3": 3, "": 9, None: 9}


# -------- load -------------------------------------------------------------
wb  = openpyxl.load_workbook(EXCEL)
ws  = wb["Issues"]
hdr = [c.value for c in ws[1]]
def col(n): return hdr.index(n)
C = {k: col(k) for k in [
    "Issue ID","Title","Status","Assignee","Reporter","Labels","Created Time",
    "Category","Priority","Fix Approach","action_TBD","action_reason",
    "owner_transferred","duplicated_issue","Dependency","action_Type",
]}

rows = [tuple(c.value for c in r) for r in ws.iter_rows(min_row=2)]
print(f"loaded {len(rows)} rows")


# -------- helpers ----------------------------------------------------------
def clean(v) -> str:
    if v is None: return ""
    s = str(v).strip()
    if s.lower() == "none": return ""
    return s

def owner(r) -> str:
    a = clean(r[C["Assignee"]])
    if a: return a
    o = clean(r[C["owner_transferred"]])
    return o

def esc(s: str, max_len: int = 0) -> str:
    s = s.replace("\r", " ").replace("\n", " ").replace("|", "\\|")
    s = re.sub(r"\s+", " ", s)
    if max_len and len(s) > max_len:
        s = s[: max_len - 1] + "…"
    return s

def fmt_list(v) -> str:
    """action_TBD / action_reason cells are JSON arrays; join with '; '."""
    s = clean(v)
    if not s:
        return ""
    if s.startswith("["):
        try:
            import json as _j
            items = _j.loads(s)
            if isinstance(items, list):
                return "; ".join(str(x) for x in items)
        except Exception:
            pass
    return s


def wrap_cell(s, width: int = 80) -> str:
    """Soft-wrap a cell to `width` chars per visual line using <br>.
    Escapes pipes, collapses whitespace, word-wraps at word boundaries."""
    s = clean(s).replace("\r", " ").replace("\n", " ").replace("|", "\\|")
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return ""
    return "<br>".join(
        textwrap.wrap(s, width=width, break_long_words=False,
                      break_on_hyphens=False)
    )


DUP_TOKEN = re.compile(r"#?(\d+)")


def parse_dup_ids(dup_cell, action_tbd) -> list[int]:
    """Extract duplicate issue IDs from duplicated_issue cell, with
    action_TBD 'duplicate of …' fallback. Forward refs only."""
    ids: set[int] = set()
    for tok in re.split(r"[,;\s]+", clean(dup_cell)):
        m = DUP_TOKEN.fullmatch(tok)
        if m:
            ids.add(int(m.group(1)))
    if not ids:
        s = clean(action_tbd).lower()
        idx = s.find("duplicate of")
        if idx >= 0:
            for m in DUP_TOKEN.finditer(s[idx:idx+200]):
                ids.add(int(m.group(1)))
    return sorted(ids)


BACK = '_[↑ Back to Index](#sec-2)_'

def issue_link(iid) -> str:
    return f"[#{iid}](https://github.com/{REPO}/issues/{iid})"

def primary(action_type: str) -> str | None:
    if not action_type: return None
    parts = action_type.split("+")
    for cat in PRIMARY_ORDER:
        if cat in parts:
            return cat
    return parts[0]

def prio_key(r):
    p = clean(r[C["Priority"]]) or None
    return PRIO_RANK.get(p, 9)

def parse_dt(s) -> datetime | None:
    s = clean(s)
    if not s: return None
    try:
        return datetime.fromisoformat(s.replace("Z","+00:00"))
    except Exception:
        return None


# -------- bucket rows into sections ---------------------------------------
by_section: dict[str, list] = defaultdict(list)
per_cat = Counter()       # multi-label category counts
per_primary = Counter()   # primary-only category counts
per_prio = Counter()
per_status = Counter()
per_category_col = Counter()
empty_action = 0
check_cases_ids: list = []
check_cases_rows: list = []   # all rows with CHECK_CASES in action_Type (any position)
unclassified_rows: list = []  # rows with empty action_Type

for r in rows:
    at = clean(r[C["action_Type"]])
    if at:
        parts = at.split("+")
        for c in parts:
            per_cat[c] += 1
        prim = primary(at)
        per_primary[prim] += 1
        if prim in DEV_SECTIONS + QA_SECTIONS:
            # Exclude issues labeled 'random' from the CLOSE bucket — they are
            # flaky tests that happened to pass in one CI run, not true fixes.
            if prim == "CLOSE" and "random" in clean(r[C["Labels"]]).lower():
                pass
            else:
                by_section[prim].append(r)
        if "CHECK_CASES" in parts:
            check_cases_ids.append(r[C["Issue ID"]])
            check_cases_rows.append(r)
    else:
        empty_action += 1
        unclassified_rows.append(r)
    per_prio[clean(r[C["Priority"]]) or "(blank)"] += 1
    per_status[clean(r[C["Status"]]) or "(blank)"] += 1
    per_category_col[clean(r[C["Category"]]) or "(blank)"] += 1

# Duplicated: duplicated_issue non-empty OR action_TBD mentions "duplicate of"
dup_rows = [r for r in rows if clean(r[C["duplicated_issue"]]) or
            "duplicate of" in clean(r[C["action_TBD"]]).lower()]

# Dependency: non-blank AND not upstream-pytorch AND not SYCL kernel:* AND not CPU fallback
def dep_ok(r) -> bool:
    d = clean(r[C["Dependency"]]).lower()
    if not d: return False
    if d == "upstream-pytorch": return False
    if d.startswith("sycl kernel"): return False
    if d == "cpu fallback": return False
    return True
dep_rows = [r for r in rows if dep_ok(r)]

# Exclude terminal QA rows (CLOSE / VERIFY_AND_CLOSE / SKIP / NOT_TARGET_CLOSE
# — sections 4.1, 4.2, 4.4, 4.6) from §6 and §7.
TERMINAL_QA = {"CLOSE", "VERIFY_AND_CLOSE", "SKIP", "NOT_TARGET_CLOSE"}
def is_terminal(r) -> bool:
    at = clean(r[C["action_Type"]])
    return primary(at) in TERMINAL_QA if at else False
dep_rows    = [r for r in dep_rows    if not is_terminal(r)]

# New <=7 days (exclude terminal-QA rows)
recent_rows = [r for r in rows
               if (dt := parse_dt(r[C["Created Time"]])) and dt >= RECENT_CUTOFF
               and not is_terminal(r)]


# -------- render -----------------------------------------------------------
def render_table(row_list) -> str:
    """Standard table: Issue | Priority | Title | Owner | Fix Approach | action_TBD | action_reason | Reporter | Labels"""
    if not row_list:
        return "_No issues._\n"
    head = "| Issue | Priority | Title | Owner | Fix Approach | action_TBD | action_reason | Reporter | Labels |"
    sep  = "|---|---|---|---|---|---|---|---|---|"
    out = [head, sep]
    sorted_rows = sorted(row_list, key=lambda r: (
        prio_key(r), str(r[C["Issue ID"]])
    ))
    for r in sorted_rows:
        out.append("| " + " | ".join([
            issue_link(r[C["Issue ID"]]),
            esc(clean(r[C["Priority"]]), 6),
            esc(clean(r[C["Title"]]), 50),
            esc(owner(r), 25),
            wrap_cell(r[C["Fix Approach"]], 80),
            esc(fmt_list(r[C["action_TBD"]]), 100),
            esc(fmt_list(r[C["action_reason"]]), 140),
            esc(clean(r[C["Reporter"]]), 20),
            esc(clean(r[C["Labels"]]), 40),
        ]) + " |")
    return "\n".join(out) + "\n"


def slug(s: str) -> str:
    """GitHub-style anchor slug."""
    s = (s or "").lower().strip()
    s = s.replace("_", "-")
    s = re.sub(r"[^a-z0-9\s-]", "", s)
    s = re.sub(r"\s+", "-", s)
    return s


def render_section_by_category(row_list, section_num: str, cat_prefix: str) -> list[str]:
    """Split rows by their `Category` column; emit a `#### cat_prefix.N <Cat>` sub-heading with a table per group.
    Returns list of markdown lines and list of (anchor, label) for TOC."""
    buckets: dict[str, list] = defaultdict(list)
    for r in row_list:
        buckets[clean(r[C["Category"]]) or "(blank)"].append(r)
    out: list[str] = []
    toc: list[tuple[str, str]] = []
    for idx, cat in enumerate(sorted(buckets), start=1):
        rows_c = buckets[cat]
        anchor = f"sec-{section_num.replace('.','-')}-{idx}-{slug(cat)}"
        out.append(f'<a id="{anchor}"></a>')
        out.append(f"#### {section_num}.{idx} {cat}  ·  {len(rows_c)} issues")
        out.append("")
        out.append(BACK)
        out.append("")
        out.append(render_table(rows_c))
        out.append("")
        toc.append((anchor, f"{section_num}.{idx} {cat} ({len(rows_c)})"))
    return out, toc


def render_dep_table(row_list) -> str:
    head = "| Issue | Dependency | Priority | Title | Owner | Fix Approach | action_TBD | action_reason | Reporter | Labels |"
    sep  = "|---|---|---|---|---|---|---|---|---|---|"
    out = [head, sep]
    sorted_rows = sorted(row_list, key=lambda r: (
        prio_key(r), clean(r[C["Dependency"]]), str(r[C["Issue ID"]])
    ))
    for r in sorted_rows:
        out.append("| " + " | ".join([
            issue_link(r[C["Issue ID"]]),
            esc(clean(r[C["Dependency"]]), 30),
            esc(clean(r[C["Priority"]]), 6),
            esc(clean(r[C["Title"]]), 50),
            esc(owner(r), 25),
            wrap_cell(r[C["Fix Approach"]], 80),
            esc(fmt_list(r[C["action_TBD"]]), 100),
            esc(fmt_list(r[C["action_reason"]]), 140),
            esc(clean(r[C["Reporter"]]), 20),
            esc(clean(r[C["Labels"]]), 40),
        ]) + " |")
    return "\n".join(out) + "\n"


def render_recent(row_list) -> str:
    head = "| Issue | Created | Priority | Title | Owner | Fix Approach | action_TBD | action_reason | Reporter | Labels |"
    sep  = "|---|---|---|---|---|---|---|---|---|---|"
    out = [head, sep]
    sorted_rows = sorted(
        row_list,
        key=lambda r: (prio_key(r), -(parse_dt(r[C["Created Time"]]) or TODAY).timestamp(), str(r[C["Issue ID"]])),
    )
    for r in sorted_rows:
        dt = parse_dt(r[C["Created Time"]])
        created = dt.strftime("%Y-%m-%d") if dt else ""
        out.append("| " + " | ".join([
            issue_link(r[C["Issue ID"]]),
            created,
            esc(clean(r[C["Priority"]]), 6),
            esc(clean(r[C["Title"]]), 50),
            esc(owner(r), 25),
            wrap_cell(r[C["Fix Approach"]], 80),
            esc(fmt_list(r[C["action_TBD"]]), 100),
            esc(fmt_list(r[C["action_reason"]]), 140),
            esc(clean(r[C["Reporter"]]), 20),
            esc(clean(r[C["Labels"]]), 40),
        ]) + " |")
    return "\n".join(out) + "\n"


def render_dup_table(row_list) -> str:
    """§5 table: adds a `Duplicates` column after `Issue` with clickable issue links
    parsed from `duplicated_issue` (fallback: 'duplicate of …' clause in action_TBD)."""
    if not row_list:
        return "_No issues._\n"
    head = "| Issue | Duplicates | Priority | Title | Owner | Fix Approach | action_TBD | action_reason | Reporter | Labels |"
    sep  = "|---|---|---|---|---|---|---|---|---|---|"
    out = [head, sep]
    sorted_rows = sorted(row_list, key=lambda r: (
        prio_key(r), str(r[C["Issue ID"]])
    ))
    for r in sorted_rows:
        ids = parse_dup_ids(r[C["duplicated_issue"]], r[C["action_TBD"]])
        if ids:
            dup_cell = ", ".join(issue_link(i) for i in ids)
        else:
            dup_cell = esc(clean(r[C["duplicated_issue"]]), 30)
        out.append("| " + " | ".join([
            issue_link(r[C["Issue ID"]]),
            dup_cell,
            esc(clean(r[C["Priority"]]), 6),
            esc(clean(r[C["Title"]]), 50),
            esc(owner(r), 25),
            wrap_cell(r[C["Fix Approach"]], 80),
            esc(fmt_list(r[C["action_TBD"]]), 100),
            esc(fmt_list(r[C["action_reason"]]), 140),
            esc(clean(r[C["Reporter"]]), 20),
            esc(clean(r[C["Labels"]]), 40),
        ]) + " |")
    return "\n".join(out) + "\n"


# ---- assemble report ------------------------------------------------------
lines: list[str] = []
def w(s=""): lines.append(s)

w(f"# XPU Ops Bug Scrub Report")
w()
w(f"- **Repository**: `{REPO}`")
w(f"- **Generated**: {TODAY.strftime('%Y-%m-%d')} (cutoff for Section 7: {RECENT_CUTOFF.strftime('%Y-%m-%d')})")
w(f"- **Total issues in workbook**: {len(rows)}")
w(f"- **Classified (non-empty `action_Type`)**: {len(rows) - empty_action}")
w(f"- **Empty `action_TBD` (no verdict)**: {empty_action}")
w()

# -- Section 1: Summary ----------------------------------------------------
w("## 1. Summary")
w()
w(f"This report groups the {len(rows)} tracked torch-xpu-ops issues into action "
  f"buckets derived from the `action_Type` classification column of the triage "
  f"workbook. Each issue appears in at most one Action-Required or QA section, "
  f"chosen by its highest-priority category. Cross-cutting slices (duplicated "
  f"issues, external dependency blockers, newly filed issues) are listed "
  f"separately for visibility.")
w()
w("**Headline counts (primary category):**")
w()
w("| Bucket | Categories | Issues |")
w("|---|---|---:|")
dev_total = sum(per_primary[c] for c in DEV_SECTIONS)
qa_total  = sum(per_primary[c] for c in QA_SECTIONS)
w(f"| Developer action required | {', '.join(DEV_SECTIONS)} | {dev_total} |")
w(f"| QA action required | {', '.join(QA_SECTIONS)} | {qa_total} |")
w(f"| Duplicated | — | {len(dup_rows)} |")
w(f"| External dependency (non-upstream-pytorch, non-SYCL-kernel) | — | {len(dep_rows)} |")
w(f"| Filed within last 7 days | — | {len(recent_rows)} |")
w()

# -- Section 2: Index ------------------------------------------------------
w('<a id="sec-2"></a>')
w("## 2. Index")
w()
w('- [3. Action required (Developer)](#sec-3)')
w('  - [3.0 UNCLASSIFIED](#sec-3-0-unclassified)')
for i, c in enumerate(DEV_SECTIONS, start=1):
    w(f'  - [3.{i} {c}](#sec-3-{i}-{slug(c)})')
w('- [4. QA](#sec-4)')
for i, c in enumerate(QA_SECTIONS, start=1):
    w(f'  - [4.{i} {c}](#sec-4-{i}-{slug(c)})')
w('- [5. Duplicated issues](#sec-5)')
w('- [6. Dependency (external blockers)](#sec-6)')
w('- [7. New submitted issues (<7 days)](#sec-7)')
w('- [8. Statistics](#sec-8)')
w()

# -- Section 3: Developer --------------------------------------------------
w('<a id="sec-3"></a>')
w("## 3. Action required (Developer)")
w()
w(BACK)
w()
w("Issues in this section require developer work before they can progress. "
  "Each subsection is split by `Category` (existing taxonomy column); "
  "rows inside each category table are sorted by `Priority` (P0 → P3).")
w()

# §3.0 Unclassified — rows with empty action_Type (Phase 4b emitted no verb)
w('<a id="sec-3-0-unclassified"></a>')
w(f"### 3.0 UNCLASSIFIED  ·  {len(unclassified_rows)} issues")
w()
w(BACK)
w()
w(f"**{SECTION_TITLES['UNCLASSIFIED']}**")
w()
w(render_table(unclassified_rows))
w()

dev_toc: list[tuple[str, str]] = []  # [(anchor, label), ...] for extended TOC
for i, cat in enumerate(DEV_SECTIONS, start=1):
    section_num = f"3.{i}"
    anchor = f"sec-3-{i}-{slug(cat)}"
    w(f'<a id="{anchor}"></a>')
    w(f"### {section_num} {cat}  ·  {len(by_section[cat])} issues")
    w()
    w(f"**{SECTION_TITLES[cat]}**")
    w()
    dev_toc.append((anchor, f"{section_num} {cat}"))
    sub_lines, sub_toc = render_section_by_category(by_section[cat], section_num, cat)
    for line in sub_lines:
        w(line)
    # extend TOC with sub-category links
    dev_toc.extend([(a, "  " + lbl) for a, lbl in sub_toc])
w()

# -- Section 4: QA ---------------------------------------------------------
w('<a id="sec-4"></a>')
w("## 4. QA")
w()
w(BACK)
w()
w("Issues in this section are ready for QA action (close, verify, reply, etc.). "
  "Rows sorted by `Priority` (P0 → P3).")
w()
for i, cat in enumerate(QA_SECTIONS, start=1):
    section_num = f"4.{i}"
    anchor = f"sec-4-{i}-{slug(cat)}"
    # §4.7 CHECK_CASES shows ALL rows tagged CHECK_CASES (any position in
    # action_Type), not just those where CHECK_CASES is the primary bucket.
    bucket = check_cases_rows if cat == "CHECK_CASES" else by_section[cat]
    w(f'<a id="{anchor}"></a>')
    w(f"### {section_num} {cat}  ·  {len(bucket)} issues")
    w()
    w(f"**{SECTION_TITLES[cat]}**")
    w()
    w(render_table(bucket))
    w()

# -- Section 5: Duplicated -------------------------------------------------
w('<a id="sec-5"></a>')
w("## 5. Duplicated issues")
w()
w(BACK)
w()
w(f"Rows where `duplicated_issue` is set or `action_TBD` contains "
  f"\"duplicate of\".  —  {len(dup_rows)} issues.")
w()
w(render_dup_table(dup_rows))
w()

# -- Section 6: Dependency -------------------------------------------------
w('<a id="sec-6"></a>')
w("## 6. Dependency (external blockers)")
w()
w(BACK)
w()
w("Issues with a non-blank `Dependency` value, excluding `upstream-pytorch`, "
  "`CPU fallback`, and `SYCL kernel:*` (in-repo kernel pointers). "
  "Terminal-QA rows (CLOSE / VERIFY_AND_CLOSE / SKIP / NOT_TARGET_CLOSE) are "
  f"also excluded.  —  {len(dep_rows)} issues.")
w()
w(render_dep_table(dep_rows))
w()

# -- Section 7: New <=7 days -----------------------------------------------
w('<a id="sec-7"></a>')
w("## 7. New submitted issues (<7 days)")
w()
w(BACK)
w()
w(f"Issues created on or after {RECENT_CUTOFF.strftime('%Y-%m-%d')}, "
  "excluding terminal-QA rows.  —  "
  f"{len(recent_rows)} issues.")
w()
w(render_recent(recent_rows))
w()

# -- Section 8: Statistics -------------------------------------------------
w('<a id="sec-8"></a>')
w("## 8. Statistics")
w()
w(BACK)
w()
w(f"- Total rows: **{len(rows)}**")
w(f"- Classified (non-empty `action_Type`): **{len(rows) - empty_action}**")
w(f"- Empty `action_TBD` (no verdict yet): **{empty_action}**")
w(f"- Issues flagged for test-case existence check (`CHECK_CASES`): **{len(check_cases_ids)}**")
w()

w("### 8.1 Primary action_Type distribution (exclusive — one bucket per issue)")
w()
w(BACK)
w()
w("| Category | Issues |")
w("|---|---:|")
for c in DEV_SECTIONS + QA_SECTIONS + ["WAIT_EXTERNAL","FILE_ISSUE","CHECK_CASES"]:
    if per_primary[c]:
        w(f"| {c} | {per_primary[c]} |")
w()

w("### 8.2 action_Type distribution (multi-label — each category counted once per issue)")
w()
w(BACK)
w()
w("| Category | Issues |")
w("|---|---:|")
for c in PRIMARY_ORDER:
    if per_cat[c]:
        w(f"| {c} | {per_cat[c]} |")
w()

w("### 8.3 Priority distribution")
w()
w(BACK)
w()
w("| Priority | Issues |")
w("|---|---:|")
for p in ["P0","P1","P2","P3","(blank)"]:
    if per_prio.get(p):
        w(f"| {p} | {per_prio[p]} |")
w()

w("### 8.4 Status distribution")
w()
w(BACK)
w()
w("| Status | Issues |")
w("|---|---:|")
for s, n in per_status.most_common():
    w(f"| {s} | {n} |")
w()

w("### 8.5 Category column distribution (top 20)")
w()
w(BACK)
w()
w("| Category | Issues |")
w("|---|---:|")
for c, n in per_category_col.most_common(20):
    w(f"| {c} | {n} |")
w()

w("### 8.6 CHECK_CASES issue IDs")
w()
w(BACK)
w()
w(f"{len(check_cases_ids)} issues flagged for `check_case_avaliablity` (missing "
  f"XPU test case in repo):")
w()
w("> " + ", ".join(f"#{i}" for i in sorted(check_cases_ids)))
w()

# ---- write ----------------------------------------------------------------
OUT.write_text("\n".join(lines))
print(f"wrote {OUT} ({OUT.stat().st_size} bytes, {len(lines)} lines)")
