"""
This module is copied from RAGAs, however I try to use not only summary, but themes and keyphrases
if available in the process of creating personas
"""

# pylint: disable=C0301
# pylint: disable=W0102
# pylint: disable=R0913

import typing as t

import numpy as np
from langchain_core.callbacks import Callbacks

from ragas.executor import run_async_batch
from ragas.llms.base import BaseRagasLLM
from ragas.prompt import (
    PydanticPrompt,
    StringIO
)
from ragas.testset.persona import Persona
from ragas.testset.graph import KnowledgeGraph, Node

def enhanced_filter(node: Node) -> bool:
    """Filter nodes either by summary, keyphrases, and themes."""
    return (
        node.type.name in ("DOCUMENT", "CHUNK")
            or node.properties.get("summary_embedding") is not None
            or node.properties.get("keyphrases") is not None
            or node.properties.get("themes") is not None
    )

class EnhancedPersonaGenerationPrompt(PydanticPrompt[StringIO, Persona]):
    """
    Custom prompt for enhanced persona generation using not only summary,
    but also keyphrases and themes.
    """
    instruction: str = (
        "Using the provided summary, keyphrases, and themes, generate a single persona "
        "who would likely interact with or benefit from the content. Include a unique name "
        "and a concise role description of who they are. The unique name should be more of a role "
        "that describes the persona's role and not the name of the persona itself. Avoid using "
        "actual names."
    )
    input_model: t.Type[StringIO] = StringIO
    output_model: t.Type[Persona] = Persona
    examples: t.List[t.Tuple[StringIO, Persona]] = [
        (
            StringIO(
                text = """Summary: Guide to Digital Marketing explains strategies for engaging audiences.
                Keyphrases: digital strategy, online marketing, audience engagement
                Themes: business, advertising"""
            ),
            Persona(
                name="Digital Marketing Specialist",
                role_description="Focuses on engaging audiences and growing the brand online.",
            ),
        )
    ]

def generate_personas_from_kg(
    kg: KnowledgeGraph,
    llm: BaseRagasLLM,
    persona_generation_prompt: EnhancedPersonaGenerationPrompt = EnhancedPersonaGenerationPrompt(),
    num_personas: int = 3,
    filter_fn: t.Callable[[Node], bool] = enhanced_filter,
    callbacks: Callbacks = [],
) -> t.List[Persona]:
    """Generate personas from a knowledge graph based on document properties (summary, keyphrases, themes)."""

    # Filter nodes
    nodes = [node for node in kg.nodes if filter_fn(node)]
    if not nodes:
        raise ValueError("No nodes matched the filter. Try modifying the filter criteria.")

    combined_texts = []
    for node in nodes:
        summary = node.properties.get("summary", "")
        keyphrases = ", ".join(node.properties.get("keyphrases", [])) if isinstance(node.properties.get("keyphrases"), list) else node.properties.get("keyphrases", "")
        themes = ", ".join(node.properties.get("themes", [])) if isinstance(node.properties.get("themes"), list) else node.properties.get("themes", "")

        combined_texts.append(f"Summary: {summary}\nKeyphrases: {keyphrases}\nThemes: {themes}")

    # Ensure we generate enough personas even if there are fewer nodes
    if len(combined_texts) < num_personas:
        combined_texts.extend(np.random.choice(combined_texts, num_personas - len(combined_texts)))

    kwargs_list = [
        {
            "llm": llm,
            "data": StringIO(text=data),
            "callbacks": callbacks,
            "temperature": 1.0, # High temperature for creative responses
        }
        for data in combined_texts[:num_personas]
    ]

    return run_async_batch(
        desc="Generating personas",
        func=persona_generation_prompt.generate,
        kwargs_list=kwargs_list,
    )
