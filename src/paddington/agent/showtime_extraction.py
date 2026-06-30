"""Option B: turn Phase 1's prose answer into typed, structured showtimes.

Phase 1's inner ReAct agent returns a *string* (``AgentResult.answer``) — the LLM
"does not hold the pen" over ``BookingState``. That string jams three planes into one:
presentational data (the actual showtimes), agent chatter ("I'll skip the other
theater"), and the ``STATUS:`` control token. Phase 2 needs a clean, machine-checkable
list to present at the interrupt boundary.

Per the Phase 2 interrupt DDR, we resolve this with a dedicated ``with_structured_output``
call (Option B): the LLM is used as a *pure function* (prose -> typed object) invoked by
our code, never touching state. It fails loud and isolated — it returns a valid
``ShowtimeList`` or raises — so the Phase 1 node can map any failure to ``NEEDS_RETRY``
rather than silently presenting an empty list (the failure mode of brittle string parsing).

We deliberately structure only the *selection boundary* (a human label + a few hint
fields), NOT a full cinema schema: over-modeling fights the architecture's "let the LLM
read the live page" thesis. The ``id`` is a round-trip token *we* assign (``st_1`` …), not
a durable DOM id; Phase 3 re-grounds on the live page using the chosen option's label.
"""

from collections.abc import Awaitable, Callable
from typing import cast

from langchain_litellm import ChatLiteLLM
from pydantic import BaseModel, Field

from paddington.config import get_settings
from paddington.logging_config import get_logger

logger = get_logger(__name__)

# An extractor takes Phase 1's raw answer and returns the structured showtimes.
# Aliased so the Phase 1 node can accept a test double without importing ChatLiteLLM.
ShowtimeExtractor = Callable[[str], Awaitable["ShowtimeList"]]


class ExtractedShowtime(BaseModel):
    """One screening as transcribed from Phase 1's prose.

    No ``id`` here on purpose: the id is a round-trip token our code assigns
    (see ``offered_options``), not something the LLM should invent. The optional
    hint fields are best-effort — present them if the prose has them, omit otherwise.
    """

    time: str = Field(description="Start time exactly as shown, e.g. '7:20 P.M.'")
    hall: str = Field(description="Auditorium/hall, e.g. 'SALA 4'. Empty string if unknown.")
    format: str | None = Field(default=None, description="e.g. '2D', '3D', 'IMAX' if shown")
    language: str | None = Field(default=None, description="e.g. 'Subtitled' / 'Dubbed' if shown")


class ShowtimeList(BaseModel):
    """The structured result of transcribing Phase 1's answer."""

    selected_theater: str = Field(
        description="The theater whose showtimes these are, e.g. 'Andino' or 'Avenida Chile'"
    )
    showtimes: list[ExtractedShowtime] = Field(
        description="Every distinct screening found, in the order shown."
    )


_EXTRACTION_SYSTEM_PROMPT = (
    "You transcribe movie showtimes from an assistant's free-text report into a "
    "structured list. Extract ONLY the showtimes that were actually found and the name "
    "of the theater they belong to. Ignore the assistant's reasoning, any 'STATUS:' "
    "line, and theaters it decided to skip. Do not invent screenings that are not "
    "explicitly present in the text."
)


def _showtime_label(s: ExtractedShowtime) -> str:
    """Human-facing one-liner for an option, e.g. '7:20 P.M. · SALA 4 · 2D · Subtitled'.

    Built from whatever fields are present, joined with ' · '. This label — not a DOM
    id — is what Phase 3 uses to re-locate the screening on the live page.
    """
    parts = [s.time.strip(), s.hall.strip(), (s.format or "").strip(), (s.language or "").strip()]
    return " · ".join(p for p in parts if p)


def offered_options(extraction: ShowtimeList) -> list[dict]:
    """Assign round-trip ids and human labels to the extracted showtimes.

    Returns the option dicts persisted into ``BookingState.offered_showtimes`` so the
    Phase 2 interrupt node is a pure read on resume (no LLM re-run, no id drift). Each
    dict keeps the id, the presentational ``label``, and the raw ``time``/``hall`` so a
    later phase can use them without re-parsing the label.
    """
    return [
        {
            "id": f"st_{i}",
            "label": _showtime_label(s),
            "time": s.time,
            "hall": s.hall,
            "format": s.format,
            "language": s.language,
        }
        for i, s in enumerate(extraction.showtimes, 1)
    ]


def build_structuring_llm(model: str) -> ChatLiteLLM:
    """Build the small LLM used for structured extraction.

    Mirrors the ``ChatLiteLLM`` wiring in ``agent_loop.py`` (temperature 0, retries,
    configured fallback) so extraction inherits the same reliability posture as the
    inner agent without sharing its tool/middleware graph.
    """
    settings = get_settings()
    return ChatLiteLLM(
        model=model,
        temperature=0.0,
        max_retries=3,
        model_kwargs={"fallbacks": [settings.fallback_model]},
    )


def default_showtime_extractor(model: str) -> ShowtimeExtractor:
    """Return the production extractor: one ``with_structured_output`` call per answer.

    The LLM is invoked as a pure function — given Phase 1's prose, it must return a valid
    ``ShowtimeList`` or raise. The caller (the Phase 1 node) treats a raise as
    ``PHASE1_NEEDS_RETRY``; this function does not swallow errors.
    """
    structured = build_structuring_llm(model).with_structured_output(ShowtimeList)

    async def extract(answer: str) -> ShowtimeList:
        # ``with_structured_output(ShowtimeList)`` is statically typed as ``dict | BaseModel``
        # but returns a ``ShowtimeList`` at runtime because that schema was passed in.
        result = await structured.ainvoke(
            [
                ("system", _EXTRACTION_SYSTEM_PROMPT),
                ("human", answer),
            ]
        )
        return cast(ShowtimeList, result)

    return extract
