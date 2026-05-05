from dataclasses import dataclass
from typing import Any
from uuid import UUID

from openai.types.chat import ChatCompletionFunctionToolParam

from paddington.llm.embedding_service import EmbeddingService
from paddington.repositories.document_repository import DocumentRepository


@dataclass
class ToolDefinition:
    """Represents a tool the agent can use."""

    name: str
    description: str
    parameters: dict
    handler: Any  # The async function to call


class PaddingtonTools:
    """Real tools that operate on paddington's data."""

    def __init__(
        self,
        document_repository: DocumentRepository,
        embedding_service: EmbeddingService,
        user_id: UUID,
    ) -> None:
        self._doc_repo = document_repository
        self._embedding_service = embedding_service
        self._user_id = user_id

    def get_all_tools(self) -> list[ToolDefinition]:
        """Return all available tools with their schemas."""
        return [
            ToolDefinition(
                name="search_documents",
                description=(
                    "Search the user's uploaded documents for information relevant to a query. "
                    "Use this when the user asks about content in their documents."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to find relevant document chunks",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results to return (default: 5)",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
                handler=self.search_documents,
            ),
            ToolDefinition(
                name="list_documents",
                description=(
                    "List all documents the user has uploaded. "
                    "Use this when the user asks what documents are available."
                ),
                parameters={
                    "type": "object",
                    "properties": {},
                },
                handler=self.list_documents,
            ),
            ToolDefinition(
                name="get_document_content",
                description=(
                    "Get the full content of a specific document by its ID. "
                    "Use this when the user asks about a specific document."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "document_id": {
                            "type": "string",
                            "description": "The UUID of the document to retrieve",
                        },
                    },
                    "required": ["document_id"],
                },
                handler=self.get_document_content,
            ),
        ]

    def get_tool_schemas(self) -> list[dict]:
        """Return tool schemas in OpenAI format (LiteLLM standard)."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self.get_all_tools()
        ]

    def get_handler(self, tool_name: str):
        """Look up a tool handler by name."""
        for tool in self.get_all_tools():
            if tool.name == tool_name:
                return tool.handler
        return None

    # ─── Tool implementations ───

    async def search_documents(self, query: str, top_k: int = 5) -> str:
        """Search documents by semantic similarity."""
        query_embedding = await self._embedding_service.embed_text(query)
        chunks = await self._doc_repo.search_similar_chunks(
            query_embedding=query_embedding,
            user_id=self._user_id,
            top_k=top_k,
        )

        if not chunks:
            return "No relevant documents found."

        results = []
        for i, chunk in enumerate(chunks):
            doc = await self._doc_repo.get_document_by_id(chunk.document_id)
            results.append(f"[Result {i + 1} from '{doc.title}']\n{chunk.content}")

        return "\n\n---\n\n".join(results)

    async def list_documents(self) -> str:
        """List all user documents."""
        documents = await self._doc_repo.list_documents_by_user(
            user_id=self._user_id,
        )

        if not documents:
            return "No documents uploaded yet."

        lines = [f"- {doc.title} (ID: {doc.id}, {doc.chunk_count} chunks)" for doc in documents]
        return "Available documents:\n" + "\n".join(lines)

    async def get_document_content(self, document_id: str) -> str:
        """Get the full content of a document."""
        try:
            doc = await self._doc_repo.get_document_by_id(UUID(document_id))
        except Exception:
            return f"Document with ID '{document_id}' not found."

        # Truncate to avoid blowing up the context window
        max_chars = 5000
        content = doc.content
        if len(content) > max_chars:
            content = content[:max_chars] + f"\n\n[...truncated, {len(doc.content)} total chars]"

        return f"Title: {doc.title}\n\n{content}"
