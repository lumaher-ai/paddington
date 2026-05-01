from fastapi import APIRouter, Depends

from paddington.agent.agent_loop import AgentConfig, AgentLoop
from paddington.agent.tools import PaddingtonTools
from paddington.dependencies import (
    get_current_user,
    get_document_repository,
    get_embedding_service,
)
from paddington.llm.embedding_service import EmbeddingService
from paddington.models import User
from paddington.repositories.document_repository import DocumentRepository
from paddington.schemas.agent import AgentRunRequest, AgentRunResponse

router = APIRouter(prefix="/agent", tags=["agent"])


@router.post("/run", response_model=AgentRunResponse)
async def run_agent(
    data: AgentRunRequest,
    current_user: User = Depends(get_current_user),
    doc_repo: DocumentRepository = Depends(get_document_repository),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
) -> AgentRunResponse:
    tools = PaddingtonTools(
        document_repository=doc_repo,
        embedding_service=embedding_service,
        user_id=current_user.id,
    )

    config = AgentConfig(
        model=data.model,
        max_iterations=data.max_iterations,
    )

    agent = AgentLoop(tools=tools, config=config)
    result = await agent.run(user_message=data.message)

    return AgentRunResponse(
        answer=result.answer,
        iterations=result.iterations,
        tools_used=result.tools_used,
        total_input_tokens=result.total_input_tokens,
        total_output_tokens=result.total_output_tokens,
        total_cost_usd=result.total_cost_usd,
    )
