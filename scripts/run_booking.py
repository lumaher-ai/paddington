"""Interactive driver for the full booking graph (Phases 1-4).

Unlike ``test_booking_phase1.py`` (which invokes the graph once and stops at the first
interrupt), this runner loops: whenever the graph pauses at an interrupt it prints the
options, reads your answer from the terminal, resumes with the matching ``Command``, and
keeps going until the graph ends. That lets you drive a whole booking conversationally:

    Phase 1 finds showtimes -> you pick one -> Phase 3 reaches the seat map ->
    Phase 4 shows available seats -> you pick seats -> END.

Prerequisites (all verified present in this repo): Postgres running (DATABASE_URL),
Playwright chromium installed, OPENAI_API_KEY set in .env.

Run:   uv run python scripts/run_booking.py
Edit the MOVIE / DATE / THEATERS / SEAT_QUANTITY constants below to try other bookings.
"""

import asyncio
import uuid
from pathlib import Path

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import Command

from paddington.agent.agent_loop import AgentConfig, AgentLoop
from paddington.agent.booking_graph import build_booking_graph, initial_booking_state
from paddington.browser.browser_session import BrowserSessionManager
from paddington.browser.debug_recorder import DebugRecorder
from paddington.browser.tools import build_browser_tools
from paddington.config import get_settings

# --- What to book (edit these) ------------------------------------------------
MOVIE = "Minions & Monsters"
DATE = "Hoy"  # "Hoy" = today, or a date as the site shows it
THEATERS = ["Andino"]
SEAT_QUANTITY = 2


async def _ask(prompt: str) -> str:
    """Read a line from the terminal without blocking the event loop."""
    return (await asyncio.to_thread(input, prompt)).strip()


async def _resume_for_showtimes(payload: dict) -> dict:
    print(f"\n🎬 {payload['prompt']}")
    options = payload["options"]
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt['label']}  (id={opt['id']})")
    if payload.get("error"):
        print(f"  ⚠️  {payload['error']}")
    ans = await _ask("Pick a number (or 'reject'): ")
    if ans.lower() in {"reject", "r", "none"}:
        return {"action": "reject"}
    try:
        return {"action": "select", "showtime_id": options[int(ans) - 1]["id"]}
    except (ValueError, IndexError):
        # Fall back to treating the raw input as an id; the node re-prompts if invalid.
        return {"action": "select", "showtime_id": ans}


async def _resume_for_seats(payload: dict) -> dict:
    print(f"\n💺 {payload['prompt']}  (choose exactly {payload['seat_quantity']})")
    for row, seats in payload["rows"].items():
        print(f"  Row {row}: {', '.join(seats)}")
    if payload.get("error"):
        print(f"  ⚠️  {payload['error']}")
    ans = await _ask("Enter seat labels comma-separated (e.g. 'A1, A2') or 'reject': ")
    if ans.lower() in {"reject", "r", "none"}:
        return {"action": "reject"}
    seats = [s.strip().upper() for s in ans.split(",") if s.strip()]
    return {"action": "select", "seats": seats}


async def _resume_value(payload: dict) -> dict:
    kind = payload.get("kind")
    if kind == "present_showtimes":
        return await _resume_for_showtimes(payload)
    if kind == "present_seats":
        return await _resume_for_seats(payload)
    raise RuntimeError(f"Unknown interrupt kind: {kind!r}")


def _print_final(state: dict) -> None:
    print("\n" + "=" * 60)
    print(f"Outcome: {state.get('phase_outcome')}")
    if state.get("selected_theater"):
        print(f"Theater: {state['selected_theater']}")
    if state.get("chosen_showtime"):
        print(f"Showtime: {state['chosen_showtime']}")
    if state.get("chosen_seats"):
        print(f"Seats: {', '.join(state['chosen_seats'])}")
    if state.get("payment_link"):
        print(f"Payment link: {state['payment_link']}")
    msgs = state.get("messages") or []
    if msgs and hasattr(msgs[-1], "content"):
        print(f"\nAgent's last word:\n{str(msgs[-1].content)[:800]}")
    print("=" * 60)


async def main() -> None:
    settings = get_settings()
    # Fresh thread id per run so we never resume a stale/half-finished checkpoint.
    thread_id = f"run-booking-{uuid.uuid4().hex[:8]}"

    async with AsyncPostgresSaver.from_conn_string(settings.checkpointer_url) as checkpointer:
        await checkpointer.setup()

        async with BrowserSessionManager() as manager:
            session = await manager.get_or_create(thread_id)
            if settings.debug_screenshots:
                session.recorder = DebugRecorder(
                    thread_id, Path(settings.debug_dir), settings.max_runs_per_thread
                )

            agent_loop = AgentLoop(
                tools=build_browser_tools(session),
                config=AgentConfig(model="gpt-4o-mini", max_iterations=25, max_cost_usd=0.50),
                checkpointer=checkpointer,
            )
            graph = build_booking_graph(
                agent_loop=agent_loop, session=session, checkpointer=checkpointer
            )

            config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
            state = initial_booking_state(
                movie=MOVIE,
                date=DATE,
                preferred_multiplexes=THEATERS,
                seat_quantity=SEAT_QUANTITY,
            )

            print(f"Booking '{MOVIE}' on {DATE} at {THEATERS} — {SEAT_QUANTITY} seat(s).")
            print("Running Phase 1 (this drives a real browser; give it a moment)...")

            result = await graph.ainvoke(state, config)
            # Loop until the graph finishes with no pending interrupt.
            while "__interrupt__" in result:
                payload = result["__interrupt__"][0].value
                resume = await _resume_value(payload)
                result = await graph.ainvoke(Command(resume=resume), config)

            _print_final(result)


if __name__ == "__main__":
    asyncio.run(main())
