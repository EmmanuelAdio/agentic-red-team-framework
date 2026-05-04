from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

try:
    from OLD.scripts.common import RetrievalDocument, make_slug, normalize_text, read_json, stable_id, write_jsonl
except ModuleNotFoundError:  # pragma: no cover - support direct script execution
    from OLD.scripts.common import RetrievalDocument, make_slug, normalize_text, read_json, stable_id, write_jsonl

# We transform nested arrays into focused JSONL documents so retrieval can match
# concise semantic units instead of searching one large nested object blob.


def _to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _hall_docs(hall: dict[str, Any], source_path: Path) -> list[RetrievalDocument]:
    name = str(hall.get("name") or "Unknown Hall")
    entity_id = f"hall::{make_slug(name)}"
    docs: list[RetrievalDocument] = []

    summary_parts: list[str] = [f"Hall: {name}."]
    if hall.get("short_description"):
        summary_parts.append(str(hall["short_description"]))
    if hall.get("catering_type"):
        summary_parts.append(f"Catering type: {hall['catering_type']}.")
    if hall.get("days_catered"):
        summary_parts.append(f"Days catered: {hall['days_catered']}.")
    if hall.get("meals_per_week") is not None:
        summary_parts.append(f"Meals per week: {hall['meals_per_week']}.")
    if hall.get("let_lengths"):
        lets = []
        for item in _to_list(hall.get("let_lengths")):
            label = item.get("label") or item.get("audience") or "tenancy"
            weeks = item.get("weeks")
            lets.append(f"{label}: {weeks} weeks")
        if lets:
            summary_parts.append("Tenancy options: " + "; ".join(lets) + ".")

    docs.append(
        RetrievalDocument(
            doc_id=stable_id("doc", entity_id, "summary"),
            entity_id=entity_id,
            entity_type="hall",
            title=f"{name} hall summary",
            source=str(source_path),
            source_type="structured_json",
            content=" ".join(summary_parts),
            doc_type="benign",
            attack_type=None,
            tags=["accommodation", "hall", "summary"],
            metadata={"doc_section": "summary", "hall_name": name},
        )
    )

    room_types = _to_list(hall.get("room_types"))
    room_lines: list[str] = []
    for room in room_types:
        room_name = room.get("name") or "Room option"
        line = [f"{room_name}"]
        if room.get("occupancy"):
            line.append(f"occupancy {room['occupancy']}")
        if room.get("bathroom_type"):
            line.append(f"bathroom {room['bathroom_type']}")
        if room.get("tenancy_weeks"):
            line.append(f"{room['tenancy_weeks']} weeks")
        prices = _to_list(room.get("prices"))
        if prices:
            price_bits: list[str] = []
            for price in prices:
                year = price.get("year") or "year"
                per_week = price.get("per_week_gbp")
                total = price.get("total_contract_gbp")
                bits = [str(year)]
                if per_week is not None:
                    bits.append(f"£{per_week}/week")
                if total is not None:
                    bits.append(f"£{total} total")
                price_bits.append(" ".join(bits))
            line.append("prices: " + "; ".join(price_bits))
        room_lines.append(", ".join(line) + ".")

    if room_lines:
        docs.append(
            RetrievalDocument(
                doc_id=stable_id("doc", entity_id, "pricing_room"),
                entity_id=entity_id,
                entity_type="hall",
                title=f"{name} room and pricing",
                source=str(source_path),
                source_type="structured_json",
                content=" ".join([f"Room and pricing options for {name}."] + room_lines),
                doc_type="benign",
                attack_type=None,
                tags=["accommodation", "hall", "pricing", "room"],
                metadata={"doc_section": "pricing_room", "hall_name": name},
            )
        )

    facilities = [normalize_text(item) for item in _to_list(hall.get("facilities")) if normalize_text(item)]
    services = [normalize_text(item) for item in _to_list(hall.get("services")) if normalize_text(item)]
    if facilities or services:
        parts = [f"Facilities and services for {name}."]
        if facilities:
            parts.append("Facilities: " + "; ".join(facilities) + ".")
        if services:
            parts.append("Services: " + "; ".join(services) + ".")
        docs.append(
            RetrievalDocument(
                doc_id=stable_id("doc", entity_id, "facilities_services"),
                entity_id=entity_id,
                entity_type="hall",
                title=f"{name} facilities and services",
                source=str(source_path),
                source_type="structured_json",
                content=" ".join(parts),
                doc_type="benign",
                attack_type=None,
                tags=["accommodation", "hall", "facilities", "services"],
                metadata={"doc_section": "facilities_services", "hall_name": name},
            )
        )

    contacts = hall.get("contacts") or {}
    contact_parts: list[str] = [f"Contact and location details for {name}."]
    if hall.get("address"):
        contact_parts.append(f"Address: {hall['address']}.")
    if hall.get("contact_email"):
        contact_parts.append(f"Accommodation email: {hall['contact_email']}.")
    if hall.get("contact_phone"):
        contact_parts.append(f"Accommodation phone: {hall['contact_phone']}.")
    if contacts.get("general_enquiries_email"):
        contact_parts.append(f"General enquiries email: {contacts['general_enquiries_email']}.")
    if contacts.get("general_enquiries_phone"):
        contact_parts.append(f"General enquiries phone: {contacts['general_enquiries_phone']}.")
    hall_manager = contacts.get("hall_manager") or {}
    if hall_manager.get("name"):
        contact_parts.append(f"Hall manager: {hall_manager['name']}.")
    if hall.get("official_url"):
        contact_parts.append(f"Official page: {hall['official_url']}.")

    if len(contact_parts) > 1:
        docs.append(
            RetrievalDocument(
                doc_id=stable_id("doc", entity_id, "contact_location"),
                entity_id=entity_id,
                entity_type="hall",
                title=f"{name} contact and location",
                source=str(source_path),
                source_type="structured_json",
                content=" ".join(contact_parts),
                doc_type="benign",
                attack_type=None,
                tags=["accommodation", "hall", "contact", "location"],
                metadata={"doc_section": "contact_location", "hall_name": name},
            )
        )

    return docs


def _course_year_chunks(course_title: str, year: dict[str, Any], max_modules_per_doc: int = 12) -> list[str]:
    year_label = str(year.get("year_label") or f"Year {year.get('year_index', '?')}")
    modules = _to_list(year.get("modules"))
    if not modules:
        return [f"{course_title} {year_label}: no module list provided."]

    chunks: list[str] = []
    for start in range(0, len(modules), max_modules_per_doc):
        subset = modules[start : start + max_modules_per_doc]
        lines = [f"{course_title} {year_label} modules."]
        for module in subset:
            title = normalize_text(module.get("title") or "Untitled module")
            desc = normalize_text(module.get("description") or "")
            if desc:
                lines.append(f"- {title}: {desc}")
            else:
                lines.append(f"- {title}")
        chunks.append(" ".join(lines))
    return chunks


def _course_docs(course: dict[str, Any], source_path: Path) -> list[RetrievalDocument]:
    title = str(course.get("course_title") or "Unknown Course")
    degree = str(course.get("degree_classification") or "").strip()
    canonical_name = f"{title} {degree}".strip()
    entity_id = f"course::{make_slug(canonical_name)}"
    docs: list[RetrievalDocument] = []

    overview_parts: list[str] = [f"Course: {canonical_name}."]
    if course.get("subject_area"):
        overview_parts.append(f"Subject area: {course['subject_area']}.")
    if course.get("start_date"):
        overview_parts.append(f"Start date: {course['start_date']}.")
    if course.get("course_overview"):
        overview_parts.append(str(course["course_overview"]))
    if course.get("professional_recognition"):
        overview_parts.append(f"Professional recognition: {course['professional_recognition']}.")

    docs.append(
        RetrievalDocument(
            doc_id=stable_id("doc", entity_id, "overview"),
            entity_id=entity_id,
            entity_type="course",
            title=f"{canonical_name} overview",
            source=str(source_path),
            source_type="structured_json",
            content=" ".join(overview_parts),
            doc_type="benign",
            attack_type=None,
            tags=["course", "overview"],
            metadata={"doc_section": "overview", "course_title": canonical_name},
        )
    )

    entry_parts: list[str] = [f"Entry requirements and fees for {canonical_name}."]
    if course.get("typical_offer"):
        entry_parts.append(f"Typical offer: {course['typical_offer']}.")
    if course.get("ucas_codes"):
        entry_parts.append(f"UCAS codes: {course['ucas_codes']}.")
    fees = course.get("fees") or {}
    uk_fees = fees.get("uk") or {}
    int_fees = fees.get("international") or {}
    if uk_fees:
        uk_parts = [f"{k}: {v}" for k, v in sorted(uk_fees.items()) if k != "heading" and v]
        if uk_parts:
            entry_parts.append("UK fees: " + "; ".join(uk_parts) + ".")
    if int_fees:
        int_parts = [f"{k}: {v}" for k, v in sorted(int_fees.items()) if k != "heading" and v]
        if int_parts:
            entry_parts.append("International fees: " + "; ".join(int_parts) + ".")
    if course.get("placement_year_available") is not None:
        entry_parts.append(f"Placement year available: {course['placement_year_available']}.")
    if course.get("year_abroad_available") is not None:
        entry_parts.append(f"Year abroad available: {course['year_abroad_available']}.")

    if len(entry_parts) > 1:
        docs.append(
            RetrievalDocument(
                doc_id=stable_id("doc", entity_id, "entry_fees"),
                entity_id=entity_id,
                entity_type="course",
                title=f"{canonical_name} entry requirements and fees",
                source=str(source_path),
                source_type="structured_json",
                content=" ".join(entry_parts),
                doc_type="benign",
                attack_type=None,
                tags=["course", "entry_requirements", "fees"],
                metadata={"doc_section": "entry_fees", "course_title": canonical_name},
            )
        )

    years = sorted(_to_list(course.get("course_content_by_year")), key=lambda row: (row.get("year_index") or 0, row.get("year_label") or ""))
    for year in years:
        year_label = str(year.get("year_label") or f"Year {year.get('year_index', '?')}")
        module_chunks = _course_year_chunks(canonical_name, year)
        for idx, chunk in enumerate(module_chunks):
            docs.append(
                RetrievalDocument(
                    doc_id=stable_id("doc", entity_id, "year_modules", year_label, str(idx)),
                    entity_id=entity_id,
                    entity_type="course",
                    title=f"{canonical_name} {year_label} modules {idx + 1}",
                    source=str(source_path),
                    source_type="structured_json",
                    content=chunk,
                    doc_type="benign",
                    attack_type=None,
                    tags=["course", "modules", make_slug(year_label)],
                    metadata={
                        "doc_section": "year_modules",
                        "course_title": canonical_name,
                        "year_label": year_label,
                        "year_index": year.get("year_index"),
                        "chunk_index": idx,
                    },
                )
            )

    return docs


def build_structured_corpus(halls_path: Path, courses_path: Path) -> list[dict[str, Any]]:
    halls = read_json(halls_path)
    courses = read_json(courses_path)

    hall_rows = sorted(halls if isinstance(halls, list) else [halls], key=lambda row: str(row.get("name") or ""))
    course_rows = sorted(
        courses if isinstance(courses, list) else [courses],
        key=lambda row: (str(row.get("course_title") or ""), str(row.get("degree_classification") or "")),
    )

    docs: list[RetrievalDocument] = []
    for hall in hall_rows:
        docs.extend(_hall_docs(hall, halls_path))
    for course in course_rows:
        docs.extend(_course_docs(course, courses_path))

    # Final deterministic ordering for stable experimental artifacts.
    ordered = sorted(docs, key=lambda doc: (doc.entity_type, doc.entity_id, doc.metadata.get("doc_section", ""), doc.doc_id))
    return [doc.to_dict() for doc in ordered]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transform structured university sources into retrieval-ready JSONL docs.")
    parser.add_argument("--halls", type=Path, default=Path("data/accommodation_halls.json"))
    parser.add_argument("--courses", type=Path, default=Path("data/all_ug_courses.json"))
    parser.add_argument("--output", type=Path, default=Path("data/corpus_structured.jsonl"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_structured_corpus(args.halls, args.courses)
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} structured retrieval documents to {args.output}")


if __name__ == "__main__":
    main()
