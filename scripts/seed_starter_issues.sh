#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — seed the launch labels + good-first-issue catalog
#  Author:  Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
#
#  SPDX-FileCopyrightText: 2026 Yash Sharma <https://www.linkedin.com/in/yashsharmaa/>
#  SPDX-License-Identifier: Apache-2.0
#
#  Doc H contributor_experience_plan / launch_checklist: "Milestones, starter
#  issues, and labels are live before announcement so outside interest has
#  somewhere to go." This script materialises the TRACKED catalog
#  (scripts/good_first_issues.yml) into real GitHub labels + issues using the
#  `gh` CLI, idempotently.
#
#  Safety: DRY-RUN by default — it only prints what it would create. Pass
#  `--apply` to actually create labels and issues. Requires an authenticated
#  `gh` (run `gh auth login` first) and `python3` to parse the catalog.
#
#  Usage:
#     scripts/seed_starter_issues.sh                 # dry run (default)
#     scripts/seed_starter_issues.sh --apply         # create labels + issues
#     scripts/seed_starter_issues.sh --apply --repo OWNER/NAME
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

APPLY=0
REPO=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CATALOG="${SCRIPT_DIR}/good_first_issues.yml"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --apply) APPLY=1 ;;
    --repo) shift; REPO="${1:-}" ;;
    --catalog) shift; CATALOG="${1:-}" ;;
    -h|--help)
      grep -E '^#( |$)' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

if [ ! -f "$CATALOG" ]; then
  echo "catalog not found: $CATALOG" >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required to parse the catalog" >&2
  exit 1
fi

REPO_ARGS=()
if [ -n "$REPO" ]; then
  REPO_ARGS=(--repo "$REPO")
fi

# Emit a TSV of records from the YAML catalog using only the stdlib. Each line is
# either:   LABEL<TAB>name<TAB>color<TAB>description
#       or  ISSUE<TAB>title<TAB>comma,labels<TAB>area<TAB>base64-body
parse_catalog() {
  python3 - "$CATALOG" <<'PY'
import base64, sys

# Minimal dependency-free YAML reader for THIS file's known shape (labels: / issues:
# lists of mappings, with a literal-block `body: |`). It is intentionally not a
# general YAML parser — it understands exactly the catalog this repo ships.
path = sys.argv[1]
lines = open(path, encoding="utf-8").read().splitlines()

def emit(*cols):
    print("\t".join(cols))

section = None
i = 0
n = len(lines)

def strip_comment(s):
    # keep '#' inside quotes; our values are simple, so split on ' #'
    return s.split("  #")[0].rstrip()

while i < n:
    raw = lines[i]
    line = raw.rstrip("\n")
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        i += 1
        continue
    if line.startswith("labels:"):
        section = "labels"; i += 1; continue
    if line.startswith("issues:"):
        section = "issues"; i += 1; continue

    if section == "labels" and stripped.startswith("- name:"):
        name = strip_comment(stripped.split("- name:", 1)[1].strip()).strip().strip('"')
        color = ""; desc = ""
        j = i + 1
        while j < n and not lines[j].lstrip().startswith("- ") and lines[j].strip():
            s = lines[j].strip()
            if s.startswith("color:"):
                color = strip_comment(s.split("color:",1)[1].strip()).strip().strip('"')
            elif s.startswith("description:"):
                desc = strip_comment(s.split("description:",1)[1].strip()).strip().strip('"')
            j += 1
        emit("LABEL", name, color, desc)
        i = j; continue

    if section == "issues" and stripped.startswith("- title:"):
        title = strip_comment(stripped.split("- title:",1)[1].strip()).strip().strip('"')
        area = ""; labels = ""; body_lines = []
        j = i + 1
        while j < n:
            s = lines[j]
            st = s.strip()
            if st.startswith("- title:"):
                break
            if st.startswith("area:"):
                area = strip_comment(st.split("area:",1)[1].strip()).strip().strip('"')
                j += 1; continue
            if st.startswith("labels:"):
                inside = st.split("labels:",1)[1].strip()
                inside = inside.strip("[]")
                labels = ",".join(x.strip().strip('"') for x in inside.split(",") if x.strip())
                j += 1; continue
            if st.startswith("body:"):
                # literal block: consume more-indented following lines
                j += 1
                base_indent = None
                while j < n:
                    bl = lines[j]
                    if bl.strip() and not bl.startswith(" "):
                        break
                    if bl.strip().startswith("- title:"):
                        break
                    if base_indent is None and bl.strip():
                        base_indent = len(bl) - len(bl.lstrip())
                    if bl.strip() == "" and base_indent is None:
                        j += 1; continue
                    if base_indent is not None:
                        body_lines.append(bl[base_indent:] if len(bl) >= base_indent else bl)
                    j += 1
                continue
            j += 1
        body = "\n".join(body_lines).rstrip() + "\n"
        b64 = base64.b64encode(body.encode()).decode()
        emit("ISSUE", title, labels, area, b64)
        i = j; continue

    i += 1
PY
}

echo "Seeding from catalog: $CATALOG"
if [ "$APPLY" -eq 0 ]; then
  echo "(DRY RUN — pass --apply to create labels + issues)"
fi
echo

n_labels=0
n_issues=0

while IFS=$'\t' read -r kind a b c d; do
  case "$kind" in
    LABEL)
      n_labels=$((n_labels + 1))
      echo "label: $a  (#$b)  — $c"
      if [ "$APPLY" -eq 1 ]; then
        # create-or-update: `label create --force` upserts.
        gh label create "$a" --color "$b" --description "$c" --force "${REPO_ARGS[@]}" \
          >/dev/null 2>&1 || echo "  ! could not upsert label $a" >&2
      fi
      ;;
    ISSUE)
      n_issues=$((n_issues + 1))
      body="$(printf '%s' "$d" | base64 --decode 2>/dev/null || printf '%s' "$d" | base64 -D)"
      echo "issue: [$c] $a  {labels: $b}"
      if [ "$APPLY" -eq 1 ]; then
        # Build repeated --label flags from the comma list.
        label_args=()
        IFS=',' read -ra L <<< "$b"
        for lab in "${L[@]}"; do
          [ -n "$lab" ] && label_args+=(--label "$lab")
        done
        gh issue create --title "$a" --body "$body" "${label_args[@]}" "${REPO_ARGS[@]}" \
          >/dev/null 2>&1 || echo "  ! could not create issue: $a" >&2
      fi
      ;;
  esac
done < <(parse_catalog)

echo
echo "Parsed $n_labels label(s) and $n_issues starter issue(s)."
if [ "$APPLY" -eq 0 ]; then
  echo "Re-run with --apply (and an authenticated gh) to create them."
fi
