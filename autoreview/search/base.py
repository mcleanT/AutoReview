from __future__ import annotations

from typing import Protocol, runtime_checkable

from autoreview.models.paper import CandidatePaper


@runtime_checkable
class SearchSource(Protocol):
    """Protocol for literature search sources."""

    @property
    def source_name(self) -> str: ...

    async def search(
        self,
        queries: list[str],
        max_results: int = 100,
    ) -> list[CandidatePaper]: ...

    async def get_paper_details(self, paper_id: str) -> CandidatePaper | None: ...
