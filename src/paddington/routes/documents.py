from fastapi import APIRouter, Depends, status

from paddington.dependencies import get_current_user, get_document_service
from paddington.models import User
from paddington.schemas.document import DocumentResponse, DocumentUploadRequest
from paddington.services.document_service import DocumentService

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    data: DocumentUploadRequest,
    current_user: User = Depends(get_current_user),
    service: DocumentService = Depends(get_document_service),
) -> DocumentResponse:
    document = await service.ingest_document(
        title=data.title,
        content=data.content,
        user_id=current_user.id,
    )
    return DocumentResponse.model_validate(document)


@router.get("", response_model=list[DocumentResponse])
async def list_documents(
    current_user: User = Depends(get_current_user),
    service: DocumentService = Depends(get_document_service),
) -> list[DocumentResponse]:
    documents = await service.list_user_documents(user_id=current_user.id)
    return [DocumentResponse.model_validate(doc) for doc in documents]
