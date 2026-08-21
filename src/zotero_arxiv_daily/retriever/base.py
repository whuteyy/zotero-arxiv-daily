from abc import ABC, abstractmethod
from omegaconf import DictConfig
from ..protocol import Paper, RawPaperItem
from tqdm import tqdm
from typing import Type
from time import sleep
from loguru import logger
from ..reranker.keyword import KeywordMatcher


class BaseRetriever(ABC):
    name: str
    def __init__(self, config:DictConfig):
        self.config = config
        self.retriever_config = getattr(config.source,self.name)
        self.keyword_matcher = (
            KeywordMatcher(config.keyword)
            if config.executor.reranker == "keyword"
            else None
        )

    @abstractmethod
    def _retrieve_raw_papers(self) -> list[RawPaperItem]:
        pass

    @abstractmethod
    def convert_to_paper(self, raw_paper:RawPaperItem) -> Paper | None:
        pass

    def retrieve_papers(self) -> list[Paper]:
        raw_papers = self._retrieve_raw_papers()
        if self.keyword_matcher is not None:
            before = len(raw_papers)
            filtered_raw_papers = []
            for raw_paper in raw_papers:
                title = raw_paper.get("title", "") if isinstance(raw_paper, dict) else getattr(raw_paper, "title", "")
                abstract = (
                    raw_paper.get("abstract", "")
                    if isinstance(raw_paper, dict)
                    else getattr(raw_paper, "summary", getattr(raw_paper, "abstract", ""))
                )
                score, _ = self.keyword_matcher.match(title, abstract)
                if score > 0:
                    filtered_raw_papers.append(raw_paper)
            raw_papers = filtered_raw_papers
            logger.info("Keyword pre-filter kept {} of {} raw papers", len(raw_papers), before)
        logger.info("Processing papers...")
        papers = []
        for raw_paper in tqdm(raw_papers, total=len(raw_papers), desc="Converting papers"):
            try:
                paper = self.convert_to_paper(raw_paper)
            except Exception as exc:
                logger.warning(f"Skipping paper {getattr(raw_paper, 'title', raw_paper)}: {exc}")
                continue
            if paper is not None:
                papers.append(paper)
            sleep(1)
        return papers

registered_retrievers = {}

def register_retriever(name:str):
    def decorator(cls):
        registered_retrievers[name] = cls
        cls.name = name
        return cls
    return decorator

def get_retriever_cls(name:str) -> Type[BaseRetriever]:
    if name not in registered_retrievers:
        raise ValueError(f"Retriever {name} not found")
    return registered_retrievers[name]
