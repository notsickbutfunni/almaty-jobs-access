"""Tests for POI category rules and the categorize function."""

from __future__ import annotations

import pandas as pd
import pytest

from almaty_access.config import Config
from almaty_access.prepare.pois import CategoryRule, categorize, load_category_rules


def test_load_rules_from_repo_yaml(repo_config: Config) -> None:
    rules = load_category_rules(repo_config.pois.categories_file)
    assert len(rules) >= 15
    categories = {rule.category for rule in rules}
    assert {"office", "grocery", "transit_stop", "industrial", "retail_other"} <= categories
    # Ordering matters: the specific grocery rules must precede the shop catch-all.
    first_grocery = next(i for i, r in enumerate(rules) if r.category == "grocery")
    first_retail = next(i for i, r in enumerate(rules) if r.category == "retail_other")
    assert first_grocery < first_retail


def test_load_rules_rejects_bad_yaml(tmp_path) -> None:
    bad = tmp_path / "cats.yaml"
    bad.write_text("categories:\n  - category: x\n    rules:\n      - values: [a]\n")
    with pytest.raises(ValueError, match="needs a 'key'"):
        load_category_rules(bad)


def test_categorize_first_match_wins() -> None:
    rules = [
        CategoryRule("grocery", "shop", frozenset({"supermarket"})),
        CategoryRule("retail_other", "shop", None),
    ]
    frame = pd.DataFrame({"shop": ["supermarket", "shoes", None]})
    result = categorize(frame, rules)
    assert result.iloc[0] == "grocery"
    assert result.iloc[1] == "retail_other"
    assert pd.isna(result.iloc[2])


def test_categorize_wildcard_and_missing_column() -> None:
    rules = [
        CategoryRule("craft", "craft", None),
        CategoryRule("office", "office", None),
    ]
    frame = pd.DataFrame({"office": ["company", None]})  # no 'craft' column at all
    result = categorize(frame, rules)
    assert result.iloc[0] == "office"
    assert pd.isna(result.iloc[1])


def test_categorize_with_repo_rules(repo_config: Config) -> None:
    rules = load_category_rules(repo_config.pois.categories_file)
    frame = pd.DataFrame(
        {
            "amenity": ["bank", "restaurant", None, None, "school"],
            "shop": [None, None, "supermarket", None, None],
            "office": [None, None, None, "it", None],
        }
    )
    result = categorize(frame, rules)
    assert result.tolist()[:2] == ["finance", "food_service"]
    assert result.iloc[2] == "grocery"
    assert result.iloc[3] == "office"
    assert result.iloc[4] == "education_school"
