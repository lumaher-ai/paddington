import asyncio

# Import your checkpointer — adjust to match your actual setup
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from paddington.agent.agent_loop import AgentConfig, AgentLoop
from paddington.agent.booking_graph import build_booking_graph, initial_booking_state
from paddington.browser.browser_session import BrowserSessionManager
from paddington.browser.tools import build_browser_tools
from paddington.config import get_settings


async def main():
    settings = get_settings()

    async with AsyncPostgresSaver.from_conn_string(settings.checkpointer_url) as checkpointer:
        await checkpointer.setup()

        async with BrowserSessionManager() as manager:
            session = await manager.get_or_create("test-booking-001")

            # Browser-only Phase 1 test: the DB/embedding/user tools aren't
            # exercised, so use the browser tools directly.
            tools = build_browser_tools(session)

            agent_loop = AgentLoop(
                tools=tools,
                config=AgentConfig(model="gpt-4o-mini", max_iterations=25, max_cost_usd=0.50),
                checkpointer=checkpointer,
            )

            graph = build_booking_graph(agent_loop=agent_loop, checkpointer=checkpointer)

            # Pick a movie currently showing on cinecolombia.com
            state = initial_booking_state(
                movie="YOUR MOVIE HERE",
                date="sábado",
                preferred_multiplexes=["Titan Plaza", "Gran Estación"],
            )

            config: RunnableConfig = {"configurable": {"thread_id": "test-booking-001"}}

            print("Running Phase 1...")
            result = await graph.ainvoke(state, config)

            print(f"\nPhase outcome: {result['phase_outcome']}")
            print(f"Phase: {result['phase']}")

            last_message = result["messages"][-1]
            if hasattr(last_message, "content"):
                print(f"\nAgent answer:\n{last_message.content[:500]}")


if __name__ == "__main__":
    asyncio.run(main())
