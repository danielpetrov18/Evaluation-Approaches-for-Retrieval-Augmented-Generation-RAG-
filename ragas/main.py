import os
from dotenv import load_dotenv
from ragas.llms import LangchainLLMWrapper
from langchain_openai.chat_models import ChatOllama
from ragas.metrics import LLMContextRecall, FactualCorrectness, Faithfulness

CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.1")

evaluator_llm = LangchainLLMWrapper(ChatOllama(model=CHAT_MODEL))
metrics = [LLMContextRecall(), FactualCorrectness(), Faithfulness()]
#results = evaluate(dataset=eval_dataset, metrics=metrics, llm=evaluator_llm)