import asyncio

from playwright.async_api import async_playwright

from paddington.browser.browser_session import BrowserSession


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)  # headless=False so you SEE it
        context = await browser.new_context()
        page = await context.new_page()
        session = BrowserSession(context, page)

        # Step 1: navigate
        result = await session.navigate("https://www.mercadolibre.com.co")
        print(f"Navigate: ok={result.ok}, status={result.status}, url={result.final_url}")

        # Step 2: snapshot
        snapshot = await session.get_snapshot(max_chars=5000)
        print(
            f"Snapshot: {len(snapshot.interactive_elements)} elements, {snapshot.total_chars} chars"
        )
        for el in snapshot.interactive_elements[:10]:
            print(f"  {el.ref} | {el.role} | {el.name[:60]}")

        # Step 3: find the search box and type
        search_box = next(
            (
                el
                for el in snapshot.interactive_elements
                if el.role in ("textbox", "combobox") and "buscar" in el.name.lower()
            ),
            None,
        )
        if search_box:
            print(f"\nTyping in {search_box.ref} ({search_box.name[:40]})")
            input_result = await session.input_text(
                ref=search_box.ref,
                text="MacBook Pro M4",
                press_enter=True,
            )
            print(f"Input: success={input_result.success}, value={input_result.value_set}")
        else:
            print("\nNo search box found! Available textbox/combobox elements:")
            for el in snapshot.interactive_elements:
                if el.role in ("textbox", "combobox"):
                    print(f"  {el.ref} | {el.role} | {el.name[:60]}")
        # Step 4: snapshot the results page
        results = await session.get_snapshot(max_chars=5000)
        print(f"\nResults page: {results.title}")
        print(f"URL: {results.url}")
        print(f"Elements: {len(results.interactive_elements)}")
        print(f"\nFirst 500 chars of markdown:\n{results.markdown[:500]}")

        # Step 5: click the first link that looks like a product
        product_link = next(
            (
                el
                for el in results.interactive_elements
                if el.role == "link" and "MacBook" in el.name
            ),
            None,
        )
        if product_link:
            print(f"\nClicking: {product_link.ref} | {product_link.name[:60]}")
            click_result = await session.click(ref=product_link.ref)
            print(f"Click: success={click_result.success}, navigated={click_result.navigated}")
            print(f"  {click_result.previous_url} → {click_result.current_url}")

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
