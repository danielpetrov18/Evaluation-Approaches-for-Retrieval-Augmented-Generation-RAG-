# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0301
# pylint: disable=E1101
# pylint: disable=R0913
# pylint: disable=R0917
# pylint: disable=W0221

from typing import Optional, List, Any, Dict

import ollama
import numpy as np
from opik.evaluation.metrics.base_metric import BaseMetric
from opik.evaluation.metrics.score_result import ScoreResult

def compute_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """
    Compute the cosine similarity between two vector embeddings.
    
    Args:
        embedding1: First vector embedding
        embedding2: Second vector embedding
    
    Returns:
        float: Cosine similarity score between the two embeddings.
        The closer to 1, the more similar the embeddings are.
    
    Raises:
        ValueError: If the embeddings have different dimensions.
    """
    if len(embedding1) != len(embedding2):
        raise ValueError("Embeddings must have the same length!")

    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)

    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


class ReciprocalRank(BaseMetric):
    """
    Implementation of the Reciprocal Rank (RR) metric using embedding-based similarity.

    Reciprocal Rank measures how high the first relevant document appears in 
    the ranked list of `retrieved contexts` relative to the `expected_output`.
    The formula for RR is 1/rank of the first relevant document.

    **NOTE**: This metric computes `Reciprocal Rank for a single query`. The Mean Reciprocal Rank (MRR)
    will be calculated by the Opik evaluation framework by averaging this metric across all samples.
    
    This metric uses:
        - `expected_output` - The expected answer (ground truth)
        - `context` - The list of retrieved contexts in ranked order

    The metric uses semantic similarity (cosine similarity between embeddings) to determine
    which contexts are relevant to the expected output. A context is considered relevant
    if its similarity score exceeds the specified threshold.
    If neither of them do exceed the threshold, the RR score is 0.

    Args:
        embedding_model: The embedding model to use for semantic similarity scoring.
        name: The name of the metric.
        similarity_threshold: The minimum similarity score for a context to be considered relevant.
            Defaults to 0.75.
        track: Whether to track the metric. Defaults to True.
        project_name: Optional project name to track the metric in for the cases when
            there are no parent span/trace to inherit project name from.
    """

    def __init__(
        self,
        embedding_model: str = "mxbai-embed-large",
        name: str = "reciprocal_rank",
        similarity_threshold: float = 0.75,
        track: bool = True,
        project_name: Optional[str] = None,
    ):
        super().__init__(name=name, track=track, project_name=project_name)
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold

    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        response: ollama.EmbedResponse = ollama.embed(
            model=self.embedding_model,
            input=texts,
        )
        return np.array(response.embeddings)

    def _calculate_reciprocal_rank(
        self,
        expected_output: str,
        contexts: List[str]
    ) -> Dict[str, Any]:
        if not contexts:
            return {
                "score": 0.0,
                "reason": "No contexts provided for evaluation."
            }

        # Get embeddings for expected output and contexts
        all_texts: List[str] = [expected_output] + contexts
        all_embeddings: np.ndarray = self._get_embeddings(all_texts)

        expected_embedding: np.ndarray = all_embeddings[0]
        context_embeddings: np.ndarray = all_embeddings[1:] # List of lists

        # Calculate similarity scores
        similarities: List[float] = []
        for context_embedding in context_embeddings:
            # Pairwise similarity computation
            similarity: float = compute_similarity(
                expected_embedding.tolist(),
                context_embedding.tolist()
            )
            similarities.append(similarity)

        # Pairwise similarity values between `expected_output` and each `context`
        # len(similarities) == len(contexts)
        similarities: np.ndarray = np.array(similarities)

        # Find relevant contexts (above threshold)
        relevant_indices = np.where(similarities >= self.similarity_threshold)[0]

        # Calculate similarity details for explanation
        similarity_details = [
            f"Context {i}: {sim:.4f}" for i, sim in enumerate(similarities, 1)
        ]

        if len(relevant_indices) == 0:
            return {
                "score": 0.0,
                "reason": f"No relevant contexts found (similarity threshold: {self.similarity_threshold}).\n"
                          f"Similarity scores:\n" + "\n".join(similarity_details)
            }

        # Find first relevant context position (1-indexed)
        first_relevant_position: int = relevant_indices[0] + 1

        # Calculate RR score
        rr_score = 1.0 / first_relevant_position

        return {
            "score": rr_score,
            "reason": f"First relevant context found at position {first_relevant_position} (similarity: {similarities[relevant_indices[0]]:.4f}).\n"
                      f"Reciprocal Rank score: 1/{first_relevant_position} = {rr_score:.4f}\n"
                      f"Similarity scores:\n" + "\n".join(similarity_details),
            "context_similarities": similarities.tolist(),
            "first_relevant_position": int(first_relevant_position),
            "relevant_indices": relevant_indices.tolist()
        }

    def score(
        self,
        expected_output: str,
        context: List[str],
        **ignored_kwargs: Any,
    ) -> ScoreResult:
        """
        Calculate the Reciprocal Rank score for the given expected output and context list.
        
        Args:
            expected_output: The expected response (ground truth).
            context: A list of context strings in ranked order.
            **ignored_kwargs: Additional keyword arguments that are ignored (like input and output).
            
        Returns:
            ScoreResult: A ScoreResult object with the RR value and explanation.
        """
        result: Dict[str, Any] = self._calculate_reciprocal_rank(expected_output, context)

        return ScoreResult(
            name=self.name,
            value=result["score"],
            reason=result["reason"]
        )

    async def ascore(
        self,
        expected_output: str,
        context: List[str],
        **ignored_kwargs: Any,
    ) -> ScoreResult:
        """ 
        Args:
            expected_output: The expected response (ground truth).
            context: A list of context strings in ranked order.
            **ignored_kwargs: Additional keyword arguments that are ignored (like input and output).

        Returns:
            ScoreResult: A ScoreResult object with the RR value and explanation.
        """
        # For embedding-based similarity, async and sync methods can be the same
        return self.score(
            expected_output,
            context
        )
