from pathlib import Path
from deepeval.models.llms import GPTModel
from deepeval.models.embedding_models import LocalEmbeddingModel
from deepeval.synthesizer.config import (
    FiltrationConfig,
    EvolutionConfig,
    StylingConfig,
    ContextConstructionConfig,
)
from deepeval.synthesizer import Synthesizer, Evolution
from .utils import save_goldens_to_files

model = GPTModel(
    model="gemini-2.5-flash",
    api_key="AIzaSyA4UXPuO3t48C3EgxvrH3TrJ3S3-waTB34",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

embeder = LocalEmbeddingModel(
    model="text-embedding-all-minilm-l6-v2-embedding",
    base_url="http://127.0.0.1:1234/v1/",
    api_key="empty",
)

filtration_config = FiltrationConfig(
    # Relax threshold to avoid over-filtering all generated samples
    synthetic_input_quality_threshold=0.3,
    max_quality_retries=1,
    critic_model=model,
)

evolution_config = EvolutionConfig(
    evolutions={
        Evolution.MULTICONTEXT: 0.25,
        Evolution.CONCRETIZING: 0.25,
        Evolution.CONSTRAINED: 0.25,
        Evolution.COMPARATIVE: 0.25,
    },
    num_evolutions=2,
)

styling_config = StylingConfig(
    input_format="Natural language questions",
    expected_output_format="Detailed paragraph responses",
    task="Customer support knowledge retrieval",
    scenario="Users seeking product troubleshooting help",
)

context_construction_config = ContextConstructionConfig(
    embedder=embeder,
    critic_model=model,
    encoding="utf-8",
    chunk_size=512,
    chunk_overlap=50,
    max_contexts_per_document=3,
    min_contexts_per_document=1,
    max_context_length=3,
    min_context_length=1,
)

synthesizer = Synthesizer(
    model=model,
    async_mode=False,
    max_concurrent=100,
    filtration_config=filtration_config,
    evolution_config=evolution_config,
    styling_config=styling_config,
    cost_tracking=True,
)

file_path = "/Users/phung.pham/Documents/PHUNGPX/deepeval_exploration/RAG/data/theranos_legacy.txt"

goldens = synthesizer.generate_goldens_from_docs(
    document_paths=[file_path],
    include_expected_output=True,
    context_construction_config=context_construction_config,
    max_goldens_per_context=2,
)

output_dir = str(Path("RAG/data/goldens") / Path(file_path).stem)

save_goldens_to_files(goldens, output_dir)
