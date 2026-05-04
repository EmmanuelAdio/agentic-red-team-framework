from __future__ import annotations

import json
from pathlib import Path

from OLD.scripts.transform_structured_sources import build_structured_corpus


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_transform_structured_sources_expands_halls_and_courses(tmp_path: Path):
    halls = [
        {
            "name": "Alpha Hall",
            "short_description": "Friendly hall.",
            "catering_type": "self_catered",
            "services": ["Weekly cleaning"],
            "facilities": ["Laundry room"],
            "room_types": [
                {
                    "name": "Standard",
                    "occupancy": "single",
                    "bathroom_type": "shared",
                    "tenancy_weeks": 40,
                    "prices": [{"year": "2026/27", "per_week_gbp": 150, "total_contract_gbp": 6000}],
                }
            ],
            "contacts": {"general_enquiries_email": "alpha@example.edu"},
            "address": "Alpha Street",
        }
    ]
    courses = [
        {
            "course_title": "Computer Science",
            "degree_classification": "BSc",
            "typical_offer": "AAA",
            "fees": {
                "uk": {"Full-time course per annum": "£9000"},
                "international": {"Full-time course per annum": "£22000"},
            },
            "course_overview": "This course teaches systems, AI, and software engineering.",
            "course_content_by_year": [
                {
                    "year_label": "Year 1",
                    "year_index": 1,
                    "modules": [
                        {"title": "Programming", "description": "Learn programming fundamentals."},
                        {"title": "Maths", "description": "Maths for CS."},
                    ],
                }
            ],
        }
    ]

    halls_path = tmp_path / "halls.json"
    courses_path = tmp_path / "courses.json"
    _write_json(halls_path, halls)
    _write_json(courses_path, courses)

    first = build_structured_corpus(halls_path, courses_path)
    second = build_structured_corpus(halls_path, courses_path)

    assert first == second
    assert any(row["entity_type"] == "hall" and row["metadata"]["doc_section"] == "summary" for row in first)
    assert any(row["entity_type"] == "hall" and row["metadata"]["doc_section"] == "pricing_room" for row in first)
    assert any(row["entity_type"] == "course" and row["metadata"]["doc_section"] == "overview" for row in first)
    assert any(row["entity_type"] == "course" and row["metadata"]["doc_section"] == "entry_fees" for row in first)
    assert any(row["entity_type"] == "course" and row["metadata"]["doc_section"] == "year_modules" for row in first)

    for row in first:
        assert row["doc_type"] == "benign"
        assert "{" not in row["content"]


def test_transform_structured_sources_handles_missing_fields(tmp_path: Path):
    halls = [{"name": "Sparse Hall"}]
    courses = [{"course_title": "Sparse Course", "degree_classification": "BSc"}]
    halls_path = tmp_path / "halls.json"
    courses_path = tmp_path / "courses.json"
    _write_json(halls_path, halls)
    _write_json(courses_path, courses)

    rows = build_structured_corpus(halls_path, courses_path)
    assert len(rows) >= 2
    assert any(row["entity_id"].startswith("hall::") for row in rows)
    assert any(row["entity_id"].startswith("course::") for row in rows)
