import json
import time
from loguru import logger
from pathlib import Path
from transformers import AutoTokenizer
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer


MAX_CHUNKED_TOKENS = 1024
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


def load_and_chunk_document(
    file_path: str,
    tokenizer_name: str = MODEL_ID,
    max_tokens: int = MAX_CHUNKED_TOKENS,
    output_dir: str | None = None,
) -> list[dict]:
    logger.info(f"[..] Loading document: {file_path}")
    logger.info(f"[..] Tokenizer name: {tokenizer_name}")
    logger.info(f"[..] Max tokens: {max_tokens}")
    logger.info(f"[..] Output directory: {output_dir}")

    tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(tokenizer_name),
    )
    chunker = HybridChunker(
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        merge_peers=True,  # Merges small adjacent items (like list items) into one chunk
        always_emit_headings=False,
    )

    t1 = time.time()
    loaded_document = DocumentConverter().convert(source=file_path).document
    t2 = time.time()
    logger.info(f"Time taken to load document: {t2 - t1} seconds")
    chunk_iter = chunker.chunk(loaded_document)
    t3 = time.time()
    logger.info(f"Time taken to chunk document: {t3 - t2} seconds")

    json_chunks = []
    for i, chunk in enumerate(chunk_iter):
        logger.info(f"Document {chunk.meta.origin.filename} - Chunk #{i + 1}")
        text_tokens = tokenizer.count_tokens(chunk.text)
        contextualized_text = chunker.contextualize(chunk=chunk)
        contextualized_tokens = tokenizer.count_tokens(contextualized_text)

        chunk_data = {
            "chunk_id": i,
            "text": chunk.text,
            "text_tokens": text_tokens,
            "contextualized_text": contextualized_text,
            "contextualized_tokens": contextualized_tokens,
            "metadata": {
                "filename": chunk.meta.origin.filename,
                "mimetype": chunk.meta.origin.mimetype,
            },
        }
        json_chunks.append(chunk_data)

    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)

        with open(
            output_dir / f"{Path(file_path).stem}.json", "w", encoding="utf-8"
        ) as f:
            json.dump(json_chunks, f, indent=4, ensure_ascii=False)
        logger.info(
            f"Saved {len(json_chunks)} chunks to '{output_dir / f'{Path(file_path).stem}.json'}"
        )

    return json_chunks


if __name__ == "__main__":
    file_paths = [
        "data/wikipedia/Albert_Einstein.md",
        "data/wikipedia/Isaac_Newton.md",
    ]
    for file_path in file_paths:
        load_and_chunk_document(file_path, output_dir="data/wikipedia/chunks")
