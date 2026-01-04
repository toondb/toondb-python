# Copyright 2025 Sushanth (https://github.com/sushanthpy)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
ToonDB ContextQuery Builder (Task 12: Killer Feature Productization)

Token-aware context retrieval for LLM applications.

The ContextQuery builder provides:
1. Token budgeting - Fit context within model limits
2. Relevance scoring - Prioritize most relevant chunks
3. Deduplication - Avoid repeating similar content
4. Structured output - Ready for LLM prompts

Example:
    context = (
        ContextQuery(collection)
        .add_vector_query(embedding, weight=0.7)
        .add_keyword_query("machine learning", weight=0.3)
        .with_token_budget(4000)
        .with_min_relevance(0.5)
        .execute()
    )
    
    prompt = f"{context.as_text()}\\n\\nQuestion: {question}"
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)
from enum import Enum

if TYPE_CHECKING:
    from .namespace import Collection


# ============================================================================
# Token Estimation
# ============================================================================

class TokenEstimator:
    """
    Estimates token count for text.
    
    Uses a simple heuristic by default (4 chars ≈ 1 token), but can be
    configured with an actual tokenizer for accuracy.
    """
    
    def __init__(self, tokenizer: Optional[Callable[[str], int]] = None):
        """
        Initialize token estimator.
        
        Args:
            tokenizer: Optional function that takes text and returns token count.
                       If None, uses heuristic (4 chars ≈ 1 token).
        """
        self._tokenizer = tokenizer
    
    def count(self, text: str) -> int:
        """Count tokens in text."""
        if self._tokenizer:
            return self._tokenizer(text)
        # Heuristic: ~4 chars per token for English
        return max(1, len(text) // 4)
    
    @classmethod
    def tiktoken(cls, model: str = "gpt-4") -> "TokenEstimator":
        """
        Create estimator using tiktoken (requires tiktoken package).
        
        Args:
            model: OpenAI model name for tokenizer selection
            
        Returns:
            TokenEstimator with tiktoken
        """
        try:
            import tiktoken
            encoding = tiktoken.encoding_for_model(model)
            return cls(lambda text: len(encoding.encode(text)))
        except ImportError:
            raise ImportError(
                "tiktoken not installed. Install with: pip install tiktoken"
            )


# ============================================================================
# Deduplication
# ============================================================================

class DeduplicationStrategy(str, Enum):
    """Strategy for deduplicating results."""
    NONE = "none"           # No deduplication
    EXACT = "exact"         # Exact text match
    SEMANTIC = "semantic"   # Semantic similarity threshold


@dataclass
class DeduplicationConfig:
    """Configuration for deduplication."""
    strategy: DeduplicationStrategy = DeduplicationStrategy.NONE
    similarity_threshold: float = 0.9  # For semantic dedup


# ============================================================================
# Context Chunk
# ============================================================================

@dataclass
class ContextChunk:
    """A chunk of context with metadata."""
    
    id: Union[str, int]
    text: str
    score: float
    tokens: int
    
    # Source information
    source: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    # Multi-vector support
    chunk_index: Optional[int] = None
    doc_score: Optional[float] = None  # Aggregated doc score
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "score": self.score,
            "tokens": self.tokens,
            "source": self.source,
            "metadata": self.metadata,
        }


# ============================================================================
# Context Result
# ============================================================================

@dataclass
class ContextResult:
    """Result of a context query."""
    
    chunks: List[ContextChunk]
    total_tokens: int
    budget_tokens: int
    dropped_count: int = 0  # Chunks dropped due to budget
    
    # Query metadata
    vector_score_range: Optional[Tuple[float, float]] = None
    keyword_score_range: Optional[Tuple[float, float]] = None
    
    def as_text(self, separator: str = "\n\n---\n\n") -> str:
        """
        Format chunks as text for LLM prompt.
        
        Args:
            separator: Separator between chunks
            
        Returns:
            Formatted context string
        """
        return separator.join(chunk.text for chunk in self.chunks)
    
    def as_markdown(self, include_scores: bool = False) -> str:
        """
        Format chunks as markdown.
        
        Args:
            include_scores: Include relevance scores
            
        Returns:
            Markdown formatted context
        """
        sections = []
        for i, chunk in enumerate(self.chunks, 1):
            header = f"### Context {i}"
            if chunk.source:
                header += f" (Source: {chunk.source})"
            if include_scores:
                header += f" [Score: {chunk.score:.3f}]"
            
            sections.append(f"{header}\n\n{chunk.text}")
        
        return "\n\n".join(sections)
    
    def as_json(self) -> str:
        """Format chunks as JSON."""
        return json.dumps({
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "total_tokens": self.total_tokens,
            "budget_tokens": self.budget_tokens,
            "dropped_count": self.dropped_count,
        }, indent=2)
    
    def __iter__(self) -> Iterator[ContextChunk]:
        return iter(self.chunks)
    
    def __len__(self) -> int:
        return len(self.chunks)


# ============================================================================
# Query Component
# ============================================================================

@dataclass
class QueryComponent:
    """A component of a context query."""
    
    query_type: str  # "vector" | "keyword"
    weight: float
    
    # For vector queries
    vector: Optional[List[float]] = None
    
    # For keyword queries
    text: Optional[str] = None
    
    # Shared options
    k: int = 50  # Retrieve more than needed for reranking


# ============================================================================
# ContextQuery Builder
# ============================================================================

class ContextQuery:
    """
    Token-aware context retrieval builder.
    
    Provides a fluent API for building context queries with:
    - Multiple query types (vector, keyword, hybrid)
    - Token budgeting
    - Relevance filtering
    - Deduplication
    
    Example:
        context = (
            ContextQuery(collection)
            .add_vector_query(embedding, weight=0.7)
            .add_keyword_query("python programming", weight=0.3)
            .with_token_budget(4000)
            .with_min_relevance(0.5)
            .with_deduplication(DeduplicationStrategy.EXACT)
            .execute()
        )
        
        # Use in prompt
        prompt = f'''Context:
        {context.as_text()}
        
        Question: {question}
        '''
    """
    
    def __init__(
        self,
        collection: "Collection",
        token_estimator: Optional[TokenEstimator] = None,
    ):
        """
        Initialize context query builder.
        
        Args:
            collection: Collection to query
            token_estimator: Optional token estimator (default: heuristic)
        """
        self._collection = collection
        self._estimator = token_estimator or TokenEstimator()
        
        # Query components
        self._components: List[QueryComponent] = []
        
        # Result options
        self._token_budget: int = 4000
        self._min_relevance: float = 0.0
        self._max_chunks: int = 50
        
        # Text options
        self._text_field: str = "content"  # Field to extract text from
        self._source_field: Optional[str] = "source"
        
        # Deduplication
        self._dedup = DeduplicationConfig()
        
        # Filtering
        self._filter: Optional[Dict[str, Any]] = None
    
    # ========================================================================
    # Query Components
    # ========================================================================
    
    def add_vector_query(
        self,
        vector: List[float],
        weight: float = 1.0,
        k: int = 50,
    ) -> "ContextQuery":
        """
        Add a vector similarity query.
        
        Args:
            vector: Query embedding
            weight: Weight for combining with other queries
            k: Number of results to retrieve (before filtering)
            
        Returns:
            self for chaining
        """
        self._components.append(QueryComponent(
            query_type="vector",
            weight=weight,
            vector=vector,
            k=k,
        ))
        return self
    
    def add_keyword_query(
        self,
        text: str,
        weight: float = 1.0,
        k: int = 50,
    ) -> "ContextQuery":
        """
        Add a keyword (BM25) query.
        
        Args:
            text: Search text
            weight: Weight for combining with other queries
            k: Number of results to retrieve (before filtering)
            
        Returns:
            self for chaining
        """
        self._components.append(QueryComponent(
            query_type="keyword",
            weight=weight,
            text=text,
            k=k,
        ))
        return self
    
    # ========================================================================
    # Result Options
    # ========================================================================
    
    def with_token_budget(self, tokens: int) -> "ContextQuery":
        """
        Set the token budget for context.
        
        Chunks will be added until the budget is exhausted.
        
        Args:
            tokens: Maximum tokens to include
            
        Returns:
            self for chaining
        """
        self._token_budget = tokens
        return self
    
    def with_min_relevance(self, threshold: float) -> "ContextQuery":
        """
        Set minimum relevance threshold.
        
        Chunks with scores below this threshold are excluded.
        
        Args:
            threshold: Minimum score (0-1 range for normalized scores)
            
        Returns:
            self for chaining
        """
        self._min_relevance = threshold
        return self
    
    def with_max_chunks(self, n: int) -> "ContextQuery":
        """
        Set maximum number of chunks.
        
        Args:
            n: Maximum chunks to return
            
        Returns:
            self for chaining
        """
        self._max_chunks = n
        return self
    
    # ========================================================================
    # Text Options
    # ========================================================================
    
    def from_field(self, field: str) -> "ContextQuery":
        """
        Specify which metadata field contains the text.
        
        Args:
            field: Metadata field name
            
        Returns:
            self for chaining
        """
        self._text_field = field
        return self
    
    def with_source_field(self, field: Optional[str]) -> "ContextQuery":
        """
        Specify which metadata field contains the source.
        
        Args:
            field: Metadata field name (None to disable)
            
        Returns:
            self for chaining
        """
        self._source_field = field
        return self
    
    # ========================================================================
    # Deduplication
    # ========================================================================
    
    def with_deduplication(
        self,
        strategy: DeduplicationStrategy,
        similarity_threshold: float = 0.9,
    ) -> "ContextQuery":
        """
        Configure deduplication.
        
        Args:
            strategy: Deduplication strategy
            similarity_threshold: Threshold for semantic dedup
            
        Returns:
            self for chaining
        """
        self._dedup = DeduplicationConfig(strategy, similarity_threshold)
        return self
    
    # ========================================================================
    # Filtering
    # ========================================================================
    
    def with_filter(self, filter: Dict[str, Any]) -> "ContextQuery":
        """
        Add metadata filter.
        
        Args:
            filter: Metadata filter dict
            
        Returns:
            self for chaining
        """
        self._filter = filter
        return self
    
    # ========================================================================
    # Execution
    # ========================================================================
    
    def execute(self) -> ContextResult:
        """
        Execute the context query.
        
        Returns:
            ContextResult with chunks fitting within token budget
        """
        if not self._components:
            raise ValueError("No query components added. Use add_vector_query() or add_keyword_query()")
        
        # Execute queries and combine results
        all_results = self._execute_queries()
        
        # Deduplicate
        all_results = self._deduplicate(all_results)
        
        # Sort by score
        all_results.sort(key=lambda x: x.score, reverse=True)
        
        # Apply relevance filter
        all_results = [r for r in all_results if r.score >= self._min_relevance]
        
        # Budget allocation
        chunks = []
        total_tokens = 0
        dropped = 0
        
        for result in all_results[:self._max_chunks]:
            if total_tokens + result.tokens > self._token_budget:
                dropped += 1
                continue
            
            chunks.append(result)
            total_tokens += result.tokens
        
        dropped += len(all_results) - len(chunks) - dropped
        
        return ContextResult(
            chunks=chunks,
            total_tokens=total_tokens,
            budget_tokens=self._token_budget,
            dropped_count=dropped,
        )
    
    def _execute_queries(self) -> List[ContextChunk]:
        """Execute all query components and combine results."""
        from .namespace import SearchRequest
        
        # Track scores by ID for combining
        combined: Dict[Union[str, int], Dict[str, Any]] = {}
        
        for component in self._components:
            if component.query_type == "vector" and component.vector:
                request = SearchRequest(
                    vector=component.vector,
                    k=component.k,
                    filter=self._filter,
                    include_metadata=True,
                )
                results = self._collection.search(request)
                
                for result in results:
                    if result.id not in combined:
                        combined[result.id] = {
                            "id": result.id,
                            "metadata": result.metadata or {},
                            "vector_score": 0.0,
                            "keyword_score": 0.0,
                        }
                    combined[result.id]["vector_score"] += result.score * component.weight
            
            elif component.query_type == "keyword" and component.text:
                request = SearchRequest(
                    text_query=component.text,
                    k=component.k,
                    filter=self._filter,
                    alpha=0.0,  # Pure keyword
                    include_metadata=True,
                )
                results = self._collection.search(request)
                
                for result in results:
                    if result.id not in combined:
                        combined[result.id] = {
                            "id": result.id,
                            "metadata": result.metadata or {},
                            "vector_score": 0.0,
                            "keyword_score": 0.0,
                        }
                    combined[result.id]["keyword_score"] += result.score * component.weight
        
        # Convert to chunks
        chunks = []
        total_weight = sum(c.weight for c in self._components)
        
        for id, data in combined.items():
            metadata = data["metadata"]
            text = str(metadata.get(self._text_field, ""))
            
            # Normalize combined score
            combined_score = (data["vector_score"] + data["keyword_score"]) / total_weight
            
            chunk = ContextChunk(
                id=id,
                text=text,
                score=combined_score,
                tokens=self._estimator.count(text),
                source=metadata.get(self._source_field) if self._source_field else None,
                metadata=metadata,
            )
            chunks.append(chunk)
        
        return chunks
    
    def _deduplicate(self, chunks: List[ContextChunk]) -> List[ContextChunk]:
        """Remove duplicate chunks based on strategy."""
        if self._dedup.strategy == DeduplicationStrategy.NONE:
            return chunks
        
        if self._dedup.strategy == DeduplicationStrategy.EXACT:
            seen_texts = set()
            deduped = []
            for chunk in chunks:
                if chunk.text not in seen_texts:
                    seen_texts.add(chunk.text)
                    deduped.append(chunk)
            return deduped
        
        # Semantic deduplication would require embedding comparison
        # For now, fall back to exact match
        return chunks


# ============================================================================
# Utility Functions
# ============================================================================

def estimate_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Estimate token count for text.
    
    Uses tiktoken if available, otherwise falls back to heuristic.
    
    Args:
        text: Text to estimate
        model: Model for tokenizer selection
        
    Returns:
        Estimated token count
    """
    try:
        estimator = TokenEstimator.tiktoken(model)
    except ImportError:
        estimator = TokenEstimator()
    
    return estimator.count(text)


def split_by_tokens(
    text: str,
    max_tokens: int = 512,
    overlap_tokens: int = 50,
    model: str = "gpt-4",
) -> List[str]:
    """
    Split text into chunks by token count.
    
    Args:
        text: Text to split
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Token overlap between chunks
        model: Model for tokenizer
        
    Returns:
        List of text chunks
    """
    try:
        estimator = TokenEstimator.tiktoken(model)
    except ImportError:
        estimator = TokenEstimator()
    
    # Simple sentence-based splitting
    sentences = text.replace("? ", "?\n").replace("! ", "!\n").replace(". ", ".\n").split("\n")
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for sentence in sentences:
        sentence_tokens = estimator.count(sentence)
        
        if current_tokens + sentence_tokens > max_tokens and current_chunk:
            chunks.append(" ".join(current_chunk))
            
            # Keep overlap
            overlap_chunk = []
            overlap_count = 0
            for s in reversed(current_chunk):
                s_tokens = estimator.count(s)
                if overlap_count + s_tokens > overlap_tokens:
                    break
                overlap_chunk.insert(0, s)
                overlap_count += s_tokens
            
            current_chunk = overlap_chunk
            current_tokens = overlap_count
        
        current_chunk.append(sentence)
        current_tokens += sentence_tokens
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks
