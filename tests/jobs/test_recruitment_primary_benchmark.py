import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from scripts.jobs.run_recruitment_primary_benchmark import (
    build_corpus, constraints_retained, evaluate_sql, load_fixture, populate_fixture,
)


class RecruitmentPrimaryBenchmarkTest(TestCase):
    def test_git_line_endings_do_not_break_annotation_integrity_but_content_changes_do(self):
        from scripts.jobs.run_recruitment_primary_benchmark import ROOT
        benchmark = (ROOT / "benchmark.json").read_bytes()
        annotations = (ROOT / "annotations.json").read_bytes().replace(b"\r\n", b"\n")
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "benchmark.json").write_bytes(benchmark)
            (root / "annotations.json").write_bytes(annotations.replace(b"\n", b"\r\n"))
            with patch("scripts.jobs.run_recruitment_primary_benchmark.ROOT", root):
                fixture, _ = load_fixture()
                self.assertEqual(134, len(fixture["retrieval_cases"]))
                (root / "annotations.json").write_bytes(annotations.replace(b"J99640", b"J99999", 1))
                with self.assertRaisesRegex(ValueError, "annotations changed"):
                    load_fixture()

    def test_frozen_source_units_are_parsed_without_query_leakage(self):
        fixture, file_hash = load_fixture()
        with TemporaryDirectory() as directory:
            chunks, units, pages, _ = build_corpus(fixture, Path(directory))
        self.assertEqual(43, len(pages))
        self.assertEqual(337, len(chunks))
        self.assertEqual(134, len(fixture["retrieval_cases"]))
        self.assertEqual(32, len(fixture["agent_cases"]))
        self.assertEqual(64, len(file_hash))
        self.assertFalse(fixture["human_reviewed"])
        corpus_text = "\n".join(chunk.text for chunk in chunks)
        for case in fixture["retrieval_cases"]:
            self.assertNotIn(case["query"], corpus_text)
            self.assertTrue(all(unit in units for unit in case["target_units"]))
        sources = {row["source_id"] for row in fixture["sources"]}
        self.assertTrue(all(document["source_id"] in sources for document in fixture["documents"]))

    def test_real_query_and_tool_match_all_frozen_filters_in_local_database(self):
        fixture, _ = load_fixture()
        engine = create_engine("sqlite+pysqlite:///:memory:", poolclass=StaticPool,
                               connect_args={"check_same_thread": False})
        try:
            with TemporaryDirectory() as directory:
                chunks, _, pages, document_pages = build_corpus(fixture, Path(directory))
                factory, aliases = populate_fixture(engine, fixture, chunks, pages, document_pages)
                report, _ = evaluate_sql(fixture, factory, aliases)
            self.assertEqual(32, report["cases"])
            self.assertEqual(32, report["passed"], json.dumps(report["results"], ensure_ascii=False))
            self.assertTrue(all(row["rejected"] for row in report["unsupported_argument_results"]))
        finally:
            engine.dispose()

    def test_missing_deadlines_are_not_derived_from_publication_dates(self):
        fixture, _ = load_fixture()
        jobs = [row for row in fixture["job_records"] if row["company"] == "百度"]
        self.assertEqual(40, len(jobs))
        self.assertTrue(all(row["application_deadline"] is None for row in jobs))
        self.assertTrue(all(row["published_date"] for row in jobs))
        self.assertTrue(any(row["education_requirement"] is None for row in jobs))
        announcement = next(row for row in fixture["job_records"] if row["id"] == "BS2026")
        self.assertEqual("2026-03-26", announcement["application_deadline"])
        self.assertIn("15:00", announcement["deadline_time_precision"])

    def test_equivalent_city_and_type_values_preserve_constraints_but_omission_fails(self):
        self.assertTrue(constraints_retained(
            {"location": "上海市", "recruitment_type": "internship"},
            {"location": "上海", "recruitment_type": "实习"}))
        self.assertFalse(constraints_retained({"recruitment_type": "internship"}, {"location": "上海"}))
        self.assertFalse(constraints_retained({"recruitment_type": "campus"}, {"recruitment_type": "实习"}))
