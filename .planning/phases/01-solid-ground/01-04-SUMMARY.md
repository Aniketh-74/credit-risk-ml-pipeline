---
phase: 01-solid-ground
plan: "04"
subsystem: ui
tags: [streamlit, plotly, pandas, dashboard, monitoring, credit-risk]

# Dependency graph
requires:
  - phase: 01-01
    provides: project scaffold, src/ package structure
  - phase: 01-02
    provides: Docker infrastructure, Dockerfile.dashboard base

provides:
  - Streamlit monitoring dashboard with 4-section layout (KPI header, drift timeline, model registry, alert log)
  - Plotly drift score chart with threshold line and alert markers
  - 4 reusable component files with render(list[dict]) interface ready for Phase 6 wiring
  - Custom CSS with dark navy sidebar and red alert accents

affects: [06-live-data, 07-deployment]

# Tech tracking
tech-stack:
  added: [plotly==5.24.1, pandas==2.2.3]
  patterns:
    - "Component isolation: each dashboard section is a standalone render(data) function in its own file"
    - "Data agnosticism: all components accept list[dict] — Phase 1 mock data, Phase 6 DB rows, no UI change"
    - "No secrets in UI: all data passed as arguments; no DB connection strings in dashboard code"

key-files:
  created:
    - src/dashboard/app.py
    - src/dashboard/components/__init__.py
    - src/dashboard/components/drift_chart.py
    - src/dashboard/components/model_history.py
    - src/dashboard/components/alert_log.py
    - src/dashboard/components/metrics_header.py
    - src/dashboard/pages/__init__.py
  modified:
    - docker/Dockerfile.dashboard

key-decisions:
  - "Plotly chosen over Altair/matplotlib for drift chart — richer interactivity, threshold annotation, triangle marker support"
  - "Component render() functions accept plain list[dict] — Phase 6 passes real DB rows with zero UI code changes"
  - "Sidebar nav uses radio buttons with session state so all four sections share the KPI header row"
  - "Custom CSS injected via st.markdown(unsafe_allow_html=True) — dark navy #0D1B2A sidebar, #E63946 red alerts, #1D3557 navy text"
  - "Dockerfile.dashboard updated with plotly==5.24.1 and pandas==2.2.3 alongside existing streamlit==1.44.0"

patterns-established:
  - "Dashboard component pattern: standalone file with def render(...) -> None and Google-style docstring"
  - "Phase 1 mock data declared as module-level MOCK_* constants in app.py with inline Phase 6 swap comment"

requirements-completed: [INFRA-02]

# Metrics
duration: 15min
completed: 2026-03-28
---

# Phase 1 Plan 04: Streamlit Dashboard Summary

**Plotly drift timeline + four-section monitoring dashboard with data-agnostic render() components ready for Phase 6 PostgreSQL wiring**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-03-28T15:33:00Z
- **Completed:** 2026-03-28T15:48:47Z
- **Tasks:** 1 of 1
- **Files modified:** 8

## Accomplishments

- Built a styled Streamlit dashboard with wide layout, dark navy sidebar, and custom CSS that makes it visually distinct from default Streamlit gray
- Implemented Plotly drift chart with red dashed threshold line at PSI=1.0 and triangle alert markers at crossing days
- Created four components with clean `render()` interfaces — no database connection required to run in Phase 1
- Updated `Dockerfile.dashboard` to install plotly and pandas alongside streamlit

## Visual Design

The dashboard uses:
- **Sidebar:** dark navy `#0D1B2A` background with `#A8DADC` text — clearly distinguishes nav from content
- **KPI cards:** `st.metric()` tiles showing current drift score (red delta when above threshold), champion model version, 30-day alert count, and simulation day
- **Drift chart:** Plotly scatter+line chart with gridlines on white background; threshold annotation in red on right side; alert triangles in `#E63946` at days 22-30
- **Tables:** `st.dataframe()` with column config for typed rendering; champion model row highlighted in light blue; retired models grayed
- **Navigation:** four-section radio nav (Overview / Drift Analysis / Model Registry / Alerts); Overview shows drift chart + model history + alert log in one view

## Task Commits

1. **Task 1: Design and build the Streamlit dashboard** - `4256efd` (feat)

## Files Created/Modified

- `src/dashboard/app.py` — Main entrypoint with page config, custom CSS, sidebar nav, mock data, section routing
- `src/dashboard/components/__init__.py` — Package init
- `src/dashboard/components/drift_chart.py` — Plotly PSI timeline with threshold line and alert markers
- `src/dashboard/components/metrics_header.py` — Four KPI metric cards using st.metric()
- `src/dashboard/components/model_history.py` — Styled dataframe with champion highlight and AUC formatting
- `src/dashboard/components/alert_log.py` — Alert event table with bool-to-label conversion
- `src/dashboard/pages/__init__.py` — Reserved for Phase 6 multi-page expansion
- `docker/Dockerfile.dashboard` — Added plotly==5.24.1 and pandas==2.2.3

## Decisions Made

- **Plotly over Altair:** Plotly's `add_hline` with annotation and custom marker symbols (`triangle-up`) gave the cleanest threshold + alert visual without custom workarounds.
- **list[dict] interface:** All `render()` functions accept plain Python dicts matching the exact column names from the PostgreSQL schema defined in Phase 1. Phase 6 passes `cursor.fetchall()` rows directly — zero UI changes needed.
- **Custom CSS via st.markdown:** Streamlit doesn't support theming per-component; injecting CSS with `unsafe_allow_html=True` is the standard approach for professional-looking dashboards.
- **Mock data in app.py:** Kept as module-level constants rather than a separate mock module — simpler to find and replace in Phase 6, and the plan's mock data spec matches what was implemented exactly.

## Deviations from Plan

None — plan executed exactly as written.

The `frontend-design` skill referenced in the plan was not present in `.agents/skills/`. Design decisions were implemented directly following the plan's explicit design constraints (color scheme, layout spec, CSS approach, component interfaces, Plotly requirement). No additional permission was needed since the plan fully specified the design.

## Issues Encountered

None.

## Phase 6 Wiring Guide

To connect real PostgreSQL data in Phase 6, replace the three `MOCK_*` constants in `app.py`:

```python
# Phase 6: replace these with DB queries
MOCK_DRIFT_SCORES = db.query("SELECT day, score, threshold_crossed FROM drift_scores ORDER BY day")
MOCK_MODEL_HISTORY = db.query("SELECT version, auc, promoted_at, status FROM model_versions ORDER BY promoted_at")
MOCK_ALERTS = db.query("SELECT fired_at, drift_score, retrain, promoted FROM retraining_alerts ORDER BY fired_at")
```

Component `render()` signatures are unchanged — they accept `list[dict]` from either source.

## Next Phase Readiness

- Dashboard is visually complete and recruiters can run `streamlit run src/dashboard/app.py` with no setup
- Component interfaces locked — Phase 6 can wire DB data without touching component code
- No database connection or environment variables required for Phase 1 demo

---
*Phase: 01-solid-ground*
*Completed: 2026-03-28*
