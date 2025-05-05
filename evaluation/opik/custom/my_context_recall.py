# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0301
# pylint: disable=C0325
# pylint: disable=W0622
# pylint: disable=R0913
# pylint: disable=R0917

from typing import (
    Any,
    List,
    Optional,
    Union,
    Dict,
    TypedDict,
    Final
)

from opik.exceptions import MetricComputationError
from opik.evaluation.models import base_model
from opik.evaluation.metrics.score_result import ScoreResult
from opik.evaluation.metrics.llm_judges import parsing_helpers
from opik.evaluation.metrics.llm_judges.context_recall.metric import ContextRecall, ContextRecallResponseFormat

class MyFewShotExampleContextRecall(TypedDict):
    title: str
    input: str
    expected_output: str
    context: str
    context_recall_score: float
    reason: str

FEW_SHOT_EXAMPLES: Final[List[MyFewShotExampleContextRecall]] = [
    {
        "title": "Partial Context Support",
        "input": "What is quantum computing and how does it differ from classical computing?",
        "expected_output": "Quantum computing is a type of computing that uses quantum bits or qubits. Unlike classical computing which uses bits that can be either 0 or 1, qubits can exist in multiple states simultaneously due to superposition. Quantum computers leverage quantum entanglement where qubits can be correlated with each other. This allows quantum computers to solve certain problems much faster than classical computers. However, quantum computers face significant challenges including maintaining quantum coherence, error correction, and scaling up the number of qubits. Currently, quantum computers are mainly used for research and specific applications like cryptography, optimization problems, and simulating quantum systems.",
        "context": "1: Quantum computing is a rapidly-emerging technology that harnesses the laws of quantum mechanics to solve problems too complex for classical computers.\n\n2: In quantum computing, information is processed using quantum bits, or qubits. Unlike classical bits, which can be either 0 or 1, qubits can exist in multiple states at once due to a quantum phenomenon called superposition.\n\n3: Another key quantum property is entanglement, where qubits can become correlated with each other in ways that are impossible with classical bits. This property allows quantum computers to process a vast number of possibilities simultaneously.\n\n4: Quantum supremacy refers to the theoretical ability of quantum computing devices to solve problems that classical computers practically cannot.\n\n5: IBM, Google, and several other companies are currently developing quantum computers with different approaches to qubit technology, including superconducting circuits, trapped ions, and topological qubits.",
        "context_recall_score": 0.6,
        "reason": "3 out of 5 statements in the expected output are supported by the retrieved context: (1) quantum computing using qubits is supported by contexts 1 and 2, (2) the difference from classical computing with qubits existing in multiple states due to superposition is supported by context 2, and (3) quantum entanglement allowing correlated qubits is supported by context 3. However, the statements about quantum computers solving certain problems faster, challenges like maintaining quantum coherence and error correction, and current applications are not explicitly supported by the retrieved context."
    },
    {
        "title": "Mixed Context Support",
        "input": "What were the major causes and effects of the Great Depression?",
        "expected_output": "The Great Depression was triggered by the stock market crash of 1929 but had multiple underlying causes. These included overproduction in agriculture and manufacturing, weak consumer demand, growing wealth inequality, excessive speculation in the stock market, and banking system instability. The Federal Reserve's tight monetary policies worsened the situation. Effects were severe and global, with unemployment reaching 25% in the United States. Thousands of banks and businesses failed. Many people lost their homes and farms to foreclosure. Poverty and homelessness increased dramatically. The Depression led to major political changes, including Roosevelt's New Deal programs which expanded government's role in the economy. Internationally, economic hardship contributed to political extremism and the rise of fascist regimes.",
        "context": "1: The Great Depression was a severe worldwide economic depression that took place mostly during the 1930s, beginning in the United States. The timing of the Great Depression varied across the world; in most countries, it started in 1929 and lasted until the late 1930s.\n\n2: The Wall Street Crash of 1929, also known as the Great Crash, was a major American stock market crash that occurred in the autumn of 1929. It started in September and ended late in October, when share prices on the New York Stock Exchange collapsed. It was the most devastating stock market crash in the history of the United States.\n\n3: While the crash is often cited as the beginning of the Great Depression, several causes contributed to the economic downturn, including: bank failures, reduction in purchasing power, American economic policy with foreign countries, and drought conditions.\n\n4: In the United States, unemployment rose to 25% and in some countries rose as high as 33%. Personal income, tax revenue, profits and prices dropped, and international trade fell by more than 50%. Construction was virtually halted in many countries.\n\n5: President Franklin D. Roosevelt implemented his New Deal programs, including numerous regulations of the financial system, direct support for the unemployed, and massive public works projects. These measures are credited with helping to end the economic downturn, though the economy did not fully recover until the United States began mobilizing for World War II.\n\n6: In several nations, the Depression was a major factor in the rise of extremist political movements, such as Nazism in Germany.",
        "context_recall_score": 0.8,
        "reason": "4 out of 5 main statements in the expected output are supported by the retrieved context: (1) the Depression being triggered by the 1929 stock market crash is supported by contexts 2 and 3, (2) multiple underlying causes including bank failures and reduced purchasing power are mentioned in context 3, (3) effects like 25% unemployment in the US, business failures, and increased poverty are supported by context 4, and (4) political changes including Roosevelt's New Deal and the rise of fascist regimes are supported by contexts 5 and 6. However, specific details about overproduction, wealth inequality, Federal Reserve policies, and many people losing homes and farms to foreclosure are not explicitly mentioned in the retrieved context."
    }
]

class MyContextRecall(ContextRecall):

    def __init__(
        self,
        model: Optional[Union[str, base_model.OpikBaseModel]] = None,
        name: str = "context_recall_metric",
        few_shot_examples: Optional[List[MyFewShotExampleContextRecall]] = None,
        track: bool = True,
        project_name: Optional[str] = None,
    ):
        super().__init__(
            model=model,
            name=name,
            few_shot_examples=(few_shot_examples if few_shot_examples else FEW_SHOT_EXAMPLES),
            track=track,
            project_name=project_name,
        )

    def score(
        self,
        input: str,
        output: str,
        expected_output: str,
        context: List[str],
        **ignored_kwargs: Any,
    ) -> ScoreResult:
        """
        Calculate the context recall score for the given input-output pair.

        Args:
            input: The input text to be evaluated.
            output: The output text to be evaluated.
            expected_output: The expected output for the given input.
            context: A list of context strings relevant to the input.
            **ignored_kwargs: Additional keyword arguments that are ignored.

        Returns:
            score_result.ScoreResult: A ScoreResult object containing the context recall score
            (between 0.0 and 1.0) and a reason for the score.
        """
        llm_query = self._generate_query(
            input=input,
            expected_output=expected_output,
            context=context,
            few_shot_examples=self.few_shot_examples,
        )
        model_output = self._model.generate_string(
            input=llm_query, response_format=ContextRecallResponseFormat
        )

        return self._parse_model_output(content=model_output, name=self.name)

    async def ascore(
        self,
        input: str,
        output: str,
        expected_output: str,
        context: List[str],
        **ignored_kwargs: Any,
    ) -> ScoreResult:
        """
        Asynchronously calculate the context recall score for the given input-output pair.

        This method is the asynchronous version of :meth:`score`. For detailed documentation,
        please refer to the :meth:`score` method.

        Args:
            input: The input text to be evaluated.
            output: The output text to be evaluated.
            expected_output: The expected output for the given input.
            context: A list of context strings relevant to the input.
            **ignored_kwargs: Additional keyword arguments that are ignored.

        Returns:
            score_result.ScoreResult: A ScoreResult object with the context recall score and reason.
        """
        llm_query = self._generate_query(
            input=input,
            expected_output=expected_output,
            context=context,
            few_shot_examples=self.few_shot_examples,
        )
        model_output = await self._model.agenerate_string(
            input=llm_query, response_format=ContextRecallResponseFormat
        )

        return self._parse_model_output(content=model_output, name=self.name)

    def _generate_query(
        self,
        input: str,
        expected_output: str,
        context: List[str],
        few_shot_examples: Optional[List[MyFewShotExampleContextRecall]] = None,
    ) -> str:
        examples_str = "\n\n".join(
            [
                f"#### Example {i + 1}: {example['title']}\n\n"
                f'- **Input:** "{example["input"]}"\n'
                f'- **Expected Output:** "{example["expected_output"]}"\n'
                f'- **Context:** "{example["context"]}"\n'
                f"- **Result:**\n"
                f"  ```json\n"
                f"  {{\n"
                f'    "context_recall_score": {example["context_recall_score"]},\n'
                f'    "reason": "{example["reason"]}"\n'
                f"  }}\n"
                f"  ```"
                for i, example in enumerate(few_shot_examples)
            ]
        )

        # Format the context properly
        context: str = "\n\n".join([f"{i + 1}: {context_str}" for i, context_str in enumerate(context)])

        return f"""You are an evaluation expert for RAG (Retrieval-Augmented Generation) systems. Your task is to measure context recall - what fraction of the claims/statements in the expected output can be attributed to the retrieved context.
 
###INSTRUCTIONS###

1. Decompose the expected output into statements/claims
2. For each statement/claim, determine if it can be remotely attributed to the retrieved context
3. Calculate the context recall score as: (number of statements in expected output supported by context) / (total number of statements in expected output)
4. Provide a JSON output with two keys:
    - "context_recall_score": a float between 0.0 and 1.0, where higher values signify better context recall
    - "reason": a short explanation for the context recall score
5. DO NOT RETURN ANYTHING ELSE BESIDES THE JSON OBJECT
6. DO NOT PROVIDE ANY FURTHER EXPLANATIONS OR CLARIFICATIONS

###SCORING GUIDE###

- 0.0: None of the statements in the expected output can be attributed to the retrieved context
- 0.2: Only a small portion of statements in the expected output are supported by the context
- 0.4: Some statements in the expected output are supported by the context, but many are not
- 0.6: Most statements in the expected output are supported by the context, with some exceptions
- 0.8: Nearly all statements in the expected output are supported by the context
- 1.0: All statements in the expected output are fully supported by the retrieved context

###EXAMPLE OUTPUT FORMAT###
{{
    "context_recall_score": 0.75,
    "reason": "3 out of 4 statements in the expected output are supported by the retrieved context. The statement about X is not supported."
}}

###FEW-SHOT EXAMPLES###

{examples_str}

###EVALUATE:###

Input:
{input}

Expected Output:
{expected_output}

Context:
{context}

JSON:
"""

    def _parse_model_output(self, content: str, name: str) -> ScoreResult:
        try:
            dict_content: Dict = parsing_helpers.extract_json_content_or_raise(content)
            score: float = float(dict_content["context_recall_score"])

            if not (0.0 <= score <= 1.0):
                raise MetricComputationError(
                    f"Context recall score must be between 0.0 and 1.0, got {score}"
                )

            return ScoreResult(
                name=name, value=score, reason=dict_content["reason"]
            )
        except Exception as e:
            raise MetricComputationError from e
