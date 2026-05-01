from pathlib import Path

from google.api_core.client_options import ClientOptions
from google.cloud import documentai_v1
from loguru import logger
from pypdf import PdfReader

from src.settings import settings

ONLINE_PROCESSING_MAX_BYTES = 20 * 1024 * 1024  # 20 MB


class DocumentAIOCRProcessor:
    """Pre-processes scanned PDFs via Google Document AI before the
    regular docling conversion pipeline."""

    def __init__(
        self,
        project_id: str | None = None,
        location: str | None = None,
        processor_id: str | None = None,
    ):
        doc_ai = settings.google_doc_ai
        self.project_id = project_id or doc_ai.google_doc_ai_project_id
        self.location = location or doc_ai.google_doc_ai_location
        self.processor_id = processor_id or doc_ai.google_doc_ai_processor_id

        if not all([self.project_id, self.processor_id]):
            raise ValueError(
                "Google Document AI requires project_id and processor_id. "
                "Set GOOGLE_DOC_AI_PROJECT_ID and GOOGLE_DOC_AI_PROCESSOR_ID in .env"
            )

        opts = ClientOptions(
            api_endpoint=f"{self.location}-documentai.googleapis.com"
        )
        self.client = documentai_v1.DocumentProcessorServiceClient(
            client_options=opts
        )
        self.processor_name = self.client.processor_path(
            self.project_id, self.location, self.processor_id
        )

    # ------------------------------------------------------------------
    # Scanned-PDF detection
    # ------------------------------------------------------------------

    @staticmethod
    def is_scanned_pdf(file_path: str, sample_pages: int = 5) -> bool:
        """Return True when most sampled pages contain no extractable text,
        which is a strong indicator the PDF is image-only / scanned."""
        path = Path(file_path)
        if path.suffix.lower() != ".pdf":
            return False

        try:
            reader = PdfReader(str(path))
        except Exception:
            return False

        total = len(reader.pages)
        if total == 0:
            return False

        pages_to_check = min(sample_pages, total)
        empty_pages = 0
        for i in range(pages_to_check):
            text = (reader.pages[i].extract_text() or "").strip()
            if len(text) < 30:
                empty_pages += 1

        return empty_pages / pages_to_check >= 0.6

    # ------------------------------------------------------------------
    # OCR via Document AI (online / synchronous processing)
    # ------------------------------------------------------------------

    def ocr(self, file_path: str) -> str:
        """Send the file to Document AI for OCR and return the full text."""
        raw = Path(file_path).read_bytes()

        if len(raw) > ONLINE_PROCESSING_MAX_BYTES:
            raise ValueError(
                f"File size ({len(raw) / 1024 / 1024:.1f} MB) exceeds the "
                f"Document AI online processing limit of 20 MB."
            )

        raw_document = documentai_v1.RawDocument(
            content=raw,
            mime_type="application/pdf",
        )
        request = documentai_v1.ProcessRequest(
            name=self.processor_name,
            raw_document=raw_document,
        )

        logger.info(f"Sending {file_path} to Document AI for OCR...")
        result = self.client.process_document(request=request)
        text = result.document.text
        logger.info(
            f"Document AI returned {len(text)} characters from {file_path}"
        )
        return text

    # ------------------------------------------------------------------
    # High-level orchestrator
    # ------------------------------------------------------------------

    def process(self, file_path: str, output_dir: str | None = None) -> str:
        """If *file_path* is a scanned PDF, OCR it and return the path to a
        markdown file containing the extracted text.  Otherwise return
        *file_path* unchanged so the caller can proceed with docling."""
        if not self.is_scanned_pdf(file_path):
            logger.info(f"{file_path} is not a scanned PDF, skipping OCR")
            return file_path

        text = self.ocr(file_path)

        if output_dir is None:
            out = Path(file_path).with_suffix(".md")
        else:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out = out_dir / f"{Path(file_path).stem}.md"

        out.write_text(text, encoding="utf-8")
        logger.info(f"OCR result saved to {out}")
        return str(out)
