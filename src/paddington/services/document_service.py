from uuid import UUID

from langchain_text_splitters import RecursiveCharacterTextSplitter

from paddington.llm.client import count_tokens
from paddington.llm.embedding_service import EmbeddingService
from paddington.logging_config import get_logger
from paddington.models.document import Document
from paddington.repositories.document_repository import DocumentRepository

logger = get_logger(__name__)

# Chunking config
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


class DocumentService:
    def __init__(
        self,
        repository: DocumentRepository,
        embedding_service: EmbeddingService,
    ) -> None:
        self._repository = repository
        self._embedding_service = embedding_service
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    async def ingest_document(
        self,
        title: str,
        content: str,
        user_id: UUID,
    ) -> Document:
        """Process a document: chunk it, embed chunks, and store everything."""
        # Step 1: Split into chunks
        chunk_texts = self._splitter.split_text(content)
        logger.info(
            "document_chunked",
            title=title,
            total_chars=len(content),
            chunk_count=len(chunk_texts),
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

        # Step 2: Embed all chunks in a single batch call
        embeddings = await self._embedding_service.embed_batch(chunk_texts)

        # Step 3: Create document record
        document = await self._repository.create_document(
            title=title,
            content=content,
            user_id=user_id,
            chunk_count=len(chunk_texts),
        )

        # Step 4: Create chunk records with embeddings
        chunks = []
        for i, (text, embedding) in enumerate(zip(chunk_texts, embeddings, strict=True)):
            chunk = await self._repository.create_chunk(
                document_id=document.id,
                chunk_index=i,
                content=text,
                token_count=count_tokens(text),
                embedding=embedding,
            )
            chunks.append(chunk)

        logger.info(
            "document_ingested",
            document_id=str(document.id),
            title=title,
            chunks_created=len(chunks),
            total_tokens=sum(c.token_count for c in chunks),
        )

        return document
