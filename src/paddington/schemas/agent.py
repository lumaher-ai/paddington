from pydantic import BaseModel, Field


class AgentRunRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000)
    model: str = Field(default="gpt-4o-mini")
    max_iterations: int = Field(default=10, ge=1, le=30)


class AgentRunResponse(BaseModel):
    answer: str
    iterations: int
    tools_used: list[str]
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float
