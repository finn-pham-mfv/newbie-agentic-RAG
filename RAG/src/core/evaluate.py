import deepeval
from tqdm import tqdm
from loguru import logger
from datasets import load_dataset
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
)
from deepeval.test_case import LLMTestCase
from deepeval.models.llms import GPTModel
from deepeval.evaluate.configs import AsyncConfig

from src.settings import settings
from src.deps.llms import LLMClient
from src.deps.vector_stores import QdrantVectorStore
from src.deps.embeddings import EmbeddingClient

llm_client = LLMClient(
    base_url=settings.llm_base_url,
    api_keys=settings.llm_api_key,
    model_id=settings.llm_model,
)

vector_store = QdrantVectorStore(
    uri=settings.qdrant_url,
    api_key=settings.qdrant_api_key,
)

embedding_client = EmbeddingClient(
    model_name=settings.embedding_model_name,
    batch_size=settings.embedding_batch_size,
    model_dim=settings.embedding_dimensions,
)


def generate_response(query: str, limit: int = 5) -> tuple[list[str], str]:
    embedding = embedding_client.embed([query])
    retrieved_documents = vector_store.query(
        collection_name=settings.qdrant_collection_name,
        query_vector=embedding[0],
        top_k=limit,
    )
    contexts = [
        "document: "
        + document.payload["content"]
        + ",source: "
        + document.payload["sources"]
        for document in retrieved_documents.points
    ]

    prompt_start = """
    # ROLE
    You are a precise Technical Support Assistant. Your goal is to answer questions based strictly on the provided documentation context.

    # RULES OF ENGAGEMENT
    1. **Greeting Logic:** If the user provides a general greeting (e.g., "Hi", "Hello"), respond with a friendly greeting and do not reference the documentation.
    2. **Contextual Fidelity:** Only answer using the provided Context. If the answer is not contained within the Context, respond exactly with: "I don't know."
    3. **Citation Requirement:** For every specific claim or instruction you provide, you MUST cite the source. Use the format: [Source Name, Page X].
    4. **Tone:** Maintain a professional, helpful, and concise tone. Avoid fluff or repetitive introductory phrases.

    # RESPONSE FORMAT
    - Use bullet points for steps or lists.
    - Bold key terms for readability.
    - Place citations at the end of the relevant sentence or paragraph.

    # CONTEXT:
    """

    # Joining contexts with clear markers
    context_block = "\n---\n".join(contexts)

    prompt_end = f"""
    ---
    # USER QUESTION: 
    {query}

    # ANSWER:
    """

    # prompt = prompt_start + context_block + prompt_end

    response = llm_client.chat_completion(
        messages=[
            {"role": "user", "content": prompt_start},
            {"role": "user", "content": context_block},
            {"role": "user", "content": prompt_end},
        ],
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )

    return contexts, response


def create_deepeval_dataset(
    evaluation_samples: int, retrieval_window_size: int
) -> list[LLMTestCase]:
    dataset = load_dataset("atitaarora/qdrant_doc_qna", split="train")
    logger.info(f"Loaded {len(dataset)} questions")

    test_cases = []
    for i in tqdm(range(evaluation_samples)):
        sample = dataset[i]
        input_question = sample["question"]
        expected_output = sample["answer"]
        context, actual_output = generate_response(
            input_question, retrieval_window_size
        )
        test_case = LLMTestCase(
            input=input_question,
            actual_output=actual_output,
            expected_output=expected_output,
            retrieval_context=context,
        )
        test_cases.append(test_case)

    logger.info(f"Created {len(test_cases)} test cases")

    return test_cases


test_cases = create_deepeval_dataset(
    evaluation_samples=10,
    retrieval_window_size=3,
)

evaluator = GPTModel(
    model=settings.llm_model,
    api_key=settings.llm_api_key,
    base_url=settings.llm_base_url,
)

deepeval.login(settings.confident_api_key)

deepeval.evaluate(
    test_cases=test_cases,
    metrics=[
        AnswerRelevancyMetric(
            model=evaluator, threshold=0.5, include_reason=True, async_mode=False
        ),
        FaithfulnessMetric(
            model=evaluator, threshold=0.5, include_reason=True, async_mode=False
        ),
        ContextualPrecisionMetric(
            model=evaluator, threshold=0.5, include_reason=True, async_mode=False
        ),
        ContextualRecallMetric(
            model=evaluator, threshold=0.5, include_reason=True, async_mode=False
        ),
        ContextualRelevancyMetric(
            model=evaluator, threshold=0.5, include_reason=True, async_mode=False
        ),
    ],
    async_config=AsyncConfig(run_async=False),
)
