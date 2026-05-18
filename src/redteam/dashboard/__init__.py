"""Streamlit dashboard package for the agentic red-team framework.

The user-facing entry point lives at ``dashboard/Home.py``; this package
holds the reusable bits (CSS injection, HTML component helpers, data
loaders, Plotly chart factories) so they can be tested in isolation and
reused by the Build-B Aggregate page.

Build A scope: see ``DASHBOARD_DESIGN_SYSTEM.md`` §9 and §13.
"""

from __future__ import annotations

__all__ = ["_css", "components", "data", "charts"]
