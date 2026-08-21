from omegaconf import OmegaConf

from zotero_arxiv_daily.reranker.keyword import KeywordMatcher, KeywordReranker
from tests.canned_responses import make_sample_paper


def make_config():
    return OmegaConf.create({
        "min_score": 4,
        "min_group_matches": 1,
        "exclude": ["ferromagnet"],
        "groups": {
            "core": {
                "weight": 4,
                "all": [
                    ["solid-state battery"],
                    ["interface", "interphase"],
                    ["lithium metal"],
                ],
            }
        },
    })


def test_keyword_match_requires_all_clauses():
    matcher = KeywordMatcher(make_config())
    score, groups = matcher.match(
        "Lithium metal solid-state battery interface",
        "We study interphase stability.",
    )
    assert score == 4
    assert groups == ["core"]

    score, groups = matcher.match("Solid-state battery interface", "No lithium metal.")
    assert score == 0
    assert groups == []


def test_keyword_match_excludes_irrelevant_terms():
    matcher = KeywordMatcher(make_config())
    score, groups = matcher.match(
        "Solid-state battery interface with lithium metal",
        "A ferromagnet study.",
    )
    assert score == 0
    assert groups == []


def test_keyword_reranker_filters_and_sorts():
    config = OmegaConf.create({"keyword": make_config()})
    reranker = KeywordReranker(config)
    relevant = make_sample_paper(
        title="Solid-state battery interface for lithium metal",
        abstract="An interphase is investigated.",
    )
    irrelevant = make_sample_paper(title="Unrelated materials paper")
    ranked = reranker.rerank([irrelevant, relevant], [])
    assert [paper.title for paper in ranked] == [relevant.title]
    assert relevant.score == 4
