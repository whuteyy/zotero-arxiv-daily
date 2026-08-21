"""Keyword-based paper filtering and ranking.

The matcher intentionally works on title + abstract only.  This keeps the
subscription precise and avoids downloading full text for papers that are
obviously outside the configured research topics.
"""

from collections.abc import Mapping

from loguru import logger
from omegaconf import DictConfig, ListConfig

from .base import BaseReranker, register_reranker
from ..protocol import CorpusPaper, Paper


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, (list, ListConfig)):
        return list(value)
    return [value]


def _normalise(text: str) -> str:
    return " ".join(text.lower().replace("–", "-").replace("—", "-").split())


def _term_present(text: str, term: str) -> bool:
    term = _normalise(str(term))
    if not term:
        return False
    return term in text


class KeywordMatcher:
    def __init__(self, config: DictConfig | Mapping):
        self.config = config
        self.min_score = float(config.get("min_score", 4.0))
        self.min_group_matches = int(config.get("min_group_matches", 1))
        self.exclude = [str(term) for term in _as_list(config.get("exclude"))]
        self.groups = config.get("groups", {})

    def match(self, title: str, abstract: str) -> tuple[float, list[str]]:
        text = _normalise(f"{title} {abstract}")
        if any(_term_present(text, term) for term in self.exclude):
            return 0.0, []

        score = 0.0
        matched_groups: list[str] = []
        for name, group in self.groups.items():
            clauses = _as_list(group.get("all"))
            if not all(any(_term_present(text, term) for term in _as_list(clause)) for clause in clauses):
                continue
            matched_groups.append(str(name))
            score += float(group.get("weight", 1.0))
            for boost in _as_list(group.get("boost")):
                if isinstance(boost, (list, ListConfig)):
                    terms, boost_weight = boost[0], float(boost[1])
                else:
                    terms, boost_weight = boost, 0.5
                if any(_term_present(text, term) for term in _as_list(terms)):
                    score += boost_weight

        if len(matched_groups) < self.min_group_matches or score < self.min_score:
            return 0.0, []
        return min(score, 10.0), matched_groups


@register_reranker("keyword")
class KeywordReranker(BaseReranker):
    """Keep only papers satisfying configured keyword groups."""

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.matcher = KeywordMatcher(config.keyword)

    def rerank(self, candidates: list[Paper], corpus: list[CorpusPaper]) -> list[Paper]:
        selected: list[Paper] = []
        for paper in candidates:
            score, matched_groups = self.matcher.match(paper.title, paper.abstract)
            if score <= 0:
                continue
            paper.score = score
            # Kept on the object for optional email rendering and diagnostics.
            paper.matched_keywords = matched_groups
            selected.append(paper)
        selected.sort(key=lambda paper: paper.score or 0.0, reverse=True)
        logger.info("Keyword filter kept {} of {} papers", len(selected), len(candidates))
        return selected

    def get_similarity_score(self, s1: list[str], s2: list[str]):
        raise NotImplementedError("KeywordReranker does not use embedding similarity")
