import unittest
from datetime import date

from db.client import upsert_article_queries
from ingestion.ingest_analytics import (
    _aggregate_gsc_query_rows,
    _extract_slug,
)
from ingestion.ingest_rss import get_existing_slugs


class FakeResult:
    def __init__(self, data):
        self.data = data


class FakeArticleQuery:
    def __init__(self, client):
        self.client = client
        self.candidates = []

    def select(self, columns):
        self.client.selected_columns.append(columns)
        return self

    def in_(self, column, candidates):
        self.client.filtered_columns.append(column)
        self.candidates = candidates
        return self

    def execute(self):
        self.client.candidate_batches.append(self.candidates)
        return FakeResult([
            {"url_slug": slug}
            for slug in self.candidates
            if slug in self.client.existing_slugs
        ])


class FakeArticleClient:
    def __init__(self, existing_slugs):
        self.existing_slugs = set(existing_slugs)
        self.selected_columns = []
        self.filtered_columns = []
        self.candidate_batches = []

    def table(self, name):
        self.asserted_table = name
        return FakeArticleQuery(self)


class FailingClient:
    def __init__(self):
        self.table_calls = 0

    def table(self, name):
        self.table_calls += 1
        raise AssertionError("database should not be called for duplicate rows")


class AnalyticsTransformTests(unittest.TestCase):
    def test_extract_slug_normalizes_url_variants(self):
        variants = [
            "/p/example",
            "/p/example/",
            "https://www.growth-memo.com/p/example?ref=home",
            "https://growth-memo.com/p/example/?utm_source=email",
        ]

        self.assertEqual(
            [_extract_slug(variant) for variant in variants],
            ["example"] * len(variants),
        )

    def test_query_rows_are_aggregated_after_url_normalization(self):
        rows = [
            {
                "keys": ["https://www.growth-memo.com/p/example", "seo"],
                "clicks": 2,
                "impressions": 10,
                "ctr": 0.2,
                "position": 4.0,
            },
            {
                "keys": [
                    "https://growth-memo.com/p/example/?utm_source=email",
                    "seo",
                ],
                "clicks": 1,
                "impressions": 30,
                "ctr": 0.0333,
                "position": 8.0,
            },
            {
                "keys": ["https://growth-memo.com/p/example", "content"],
                "clicks": 4,
                "impressions": 20,
                "ctr": 0.2,
                "position": 3.0,
            },
            {
                "keys": ["https://growth-memo.com/about", "seo"],
                "clicks": 10,
                "impressions": 100,
                "ctr": 0.1,
                "position": 1.0,
            },
        ]

        result = _aggregate_gsc_query_rows(rows, date(2026, 8, 24))

        self.assertEqual(len(result), 2)
        by_query = {row["query"]: row for row in result}
        self.assertEqual(by_query["seo"], {
            "url_slug": "example",
            "week_start": "2026-08-24",
            "query": "seo",
            "clicks": 3,
            "impressions": 40,
            "ctr": 0.075,
            "avg_position": 7.0,
        })
        self.assertEqual(by_query["content"]["clicks"], 4)


class RssLookupTests(unittest.TestCase):
    def test_lookup_queries_only_candidate_slugs_in_bounded_batches(self):
        candidates = [f"article-{i}" for i in range(205)]
        client = FakeArticleClient({"article-1", "article-204"})

        result = get_existing_slugs(client, candidates + ["article-1"])

        self.assertEqual(result, {"article-1", "article-204"})
        self.assertEqual(client.asserted_table, "articles")
        self.assertEqual(client.selected_columns, ["url_slug"] * 3)
        self.assertEqual(client.filtered_columns, ["url_slug"] * 3)
        self.assertEqual(
            [len(batch) for batch in client.candidate_batches],
            [100, 100, 5],
        )


class QueryUpsertTests(unittest.TestCase):
    def test_duplicate_keys_are_rejected_before_any_batch_is_written(self):
        client = FailingClient()
        row = {
            "url_slug": "example",
            "week_start": "2026-08-24",
            "query": "seo",
        }

        with self.assertRaisesRegex(ValueError, "must be unique"):
            upsert_article_queries(client, [row, row.copy()])

        self.assertEqual(client.table_calls, 0)


if __name__ == "__main__":
    unittest.main()
