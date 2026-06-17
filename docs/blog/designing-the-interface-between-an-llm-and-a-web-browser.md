# Designing the interface between an LLM and a web browser

I asked my agent to find MacBook Pro prices on  Mercado Libre (the Amazon for Latinos).

It opened the homepage and read it. The raw page was tens of thousands of tokens of HTML — nav bars, tracking scripts, inline SVGs, ad slots, a thousand `<div>`s with hashed class names. The model dutifully ingested all of it. Then it hallucinated a "Search" button that wasn't in the DOM, emitted a click for a CSS selector it invented, the call threw a Playwright error, and the agent gave up with "I couldn't find that information."

I paid 50k tokens per page. At GPT-4 prices, 10 snapshots per task costs ~$0.50/task. Total cost: $0.50 for zero useful output.

The problem isn't the model. The problem is the interface. An LLM agent doesn't fail at browsing because it's not smart enough — it fails because we hand it the wrong representation of the page and then punish it for guessing. This post is about the four design decisions that turned that interface from "dump the HTML and pray" into something an agent can actually reason over, and cut per-task cost by roughly 15x along the way.

The whole thing comes down to one question: **how do you give an LLM eyes and hands on the web without going broke?**

## The setup

Paddington is an agent that drives a real Chromium browser. The agent loop is standard: the LLM has a set of tools, each tool is a function, and the model decides which one to call next based on what it sees. The interesting part isn't the loop — it's the four tools, and what they hand back to the model.

```
User ──▶ Agent (LLM) ──▶ Tools ──▶ BrowserSession ──▶ Chromium
                  ▲                        │
                  └────── results ─────────┘
```

There are exactly four tools, and they live on one class, `BrowserSession`:

- `navigate_to(url)` — go to a page
- `get_page_snapshot()` — read the current page
- `click(ref)` — click something
- `input_text(ref, text)` — type into something

Everything below is about the contract these four functions present to the model. Not the Playwright calls underneath — those are boring. The contract is where the engineering is.

## Decision 1 — Reducing token cost 15x by stripping what the LLM can't use

**The problem.** A Mercado Libre search results page contains 37,710 characters of HTML. The full HTML runs to tens of thousands of tokens once you count every script tag, style block, inline SVG, and tracking pixel. My agent calls `get_page_snapshot` 8–12 times in a single task — read the homepage, read the search results, read a product page, re-read after each click. At those volumes, feeding raw HTML to the model is the dominant cost line. Ten snapshots of a 50k-token page is half a million input tokens for one task. That's real money per task, and almost all of it is spent on bytes the model will never act on.

Here's the thing: the LLM cannot *do* anything with a `<script>` tag. It can't run the JavaScript. It can't render the SVG. The hashed class names mean nothing to it. Every one of those tokens is pure cost with zero decision value.

**The options I considered.**

1. Send the raw HTML and let the model figure it out. Simple, works on day one, bankrupts you by day two.
2. Take a screenshot and use a vision model. Genuinely viable, but pixels are expensive too, and you lose the structured element references I need for clicking.
3. Strip the HTML down to the text and structure the model actually reasons over, and convert it to Markdown.

**What I chose and why.** Option 3. The model reasons over *content* and *structure* — headings, links, prices, product names. It does not reason over presentation. So I throw away everything that's presentation-only and convert what's left to Markdown, which is the densest text format a model already understands natively.

```python
_NOISE_TAGS = ("script", "style", "noscript", "svg", "iframe", "link", "meta", "head")

html = await self.page.content()
soup = BeautifulSoup(html, "html.parser")
for tag in soup(list(_NOISE_TAGS)):
    tag.decompose()
# aria-hidden / hidden elements are invisible to a sighted user —
# so they're noise to the agent too. Drop them.
for hidden in soup.find_all(attrs={"aria-hidden": "true"}):
    hidden.decompose()
for hidden in soup.find_all(attrs={"hidden": True}):
    hidden.decompose()

body = soup.body or soup
markdown_raw = markdownify(str(body), heading_style="ATX").strip()
markdown_clean = re.sub(r"\n{3,}", "\n\n", markdown_raw)  # collapse runaway whitespace
```

What gets stripped, and why each one is useless to the model:

- `script`, `noscript` — code the model can't execute.
- `style`, `link`, `meta`, `head` — presentation and metadata, not content.
- `svg`, `iframe` — opaque blobs; an inline icon path is thousands of characters of nothing.
- `aria-hidden="true"` and `hidden` elements — invisible to a human, so invisible to the agent by design.

The `aria-hidden` rule is the one I'm proudest of. The accessibility tree already encodes "what a user can actually perceive." Reusing it as a token filter means I'm aligned with a standard instead of inventing my own heuristic for what counts as noise.

**The result.** A page that was tens of thousands of tokens of HTML collapses to roughly 3k tokens of clean Markdown — about a 15x reduction. I also cap the output with a `max_chars` limit and report it honestly to the model instead of silently cutting:

```python
truncated = total_chars > max_chars
markdown = markdown_clean[:max_chars] if truncated else markdown_clean
# PageSnapshot carries truncated=bool and total_chars=int,
# so the model knows there's more page below the fold.
```

Multiply that 15x across 8–12 snapshots per task and the unit economics of the whole agent change. That's the difference between an agent you can run in production and a demo you turn off when the bill arrives.

## Decision 2 — Designing an API contract for a non-human consumer

**The problem.** Once the page is readable, the agent needs to *act*. It has to tell the browser "click that one." How do I name the thing it wants to click?

The obvious answer is to hand the model a CSS selector — `div.ui-search-link > a.ui-search-item__group__element`. It's already in the DOM. It's what Playwright wants. Why not just pass it through?

Because the LLM is an API consumer, and a selector is an implementation detail. The moment you expose an implementation detail to a model, it tries to *help*. It "fixes" the selector. It generalizes `nth-child(3)` to `nth-child(4)` because the product it wants is one row down. It invents a selector for an element it never saw. A selector is a tempting, editable string, and the model will edit it — and then click something that doesn't exist.

This is the same anti-pattern as returning raw SQL column names in a REST response. You're leaking your internals into a contract, and the consumer will couple to them and break.

**The options I considered.**

1. Pass CSS selectors straight through. Zero translation work, maximum leakage.
2. Pass XPath. Same problem, uglier.
3. Hand out opaque, stable reference IDs and keep the real selectors private on the server.

**What I chose and why.** Option 3. The model only ever sees three fields per element — a `ref`, a `role`, and a `name`:

```python
class InteractiveElement(BaseModel):
    ref: str   # 'el_12' — opaque handle, the only thing the model passes back
    role: str  # 'link', 'button', 'textbox' — what kind of thing it is
    name: str  # accessible name — what it says to a human
```

That's the entire contract. `el_12`, "link", "MacBook Pro 14 M3". No selector. The `ref` is opaque — there's nothing in `el_12` for the model to "improve."

The real selector lives in a private map on the session, keyed by ref. The model never touches it:

```python
# server-side only; the model never sees this
self._ref_map[el["ref"]] = f'[data-paddington-ref="{el["ref"]}"]'
```

And the trick that makes the ref *stable*: when I collect interactive elements, I stamp a `data-paddington-ref` attribute directly onto each DOM node in the browser. So `el_12` isn't a fragile path through the tree — it's a marker I planted on the exact element I described to the model.

```javascript
const ref = `el_${i + 1}`;
el.setAttribute('data-paddington-ref', ref);  // plant a stable handle on the node itself
```

When the model later calls `click("el_12")`, I look up `[data-paddington-ref="el_12"]`, which resolves to exactly the node I tagged — no ambiguity, no guessing. The model picks from a menu of things I've confirmed exist. It cannot point at anything else, because the only valid inputs are refs I issued.

**The result.** Hallucinated clicks went to zero. Not "down" — to zero. The model literally cannot pass a target I didn't hand it, because the only thing it has is an opaque string from my last snapshot. I traded a small amount of server-side bookkeeping (one dict, one DOM attribute) for the entire class of "the model invented a selector" failures.

## Decision 3 — Observable facts over interpreted booleans

**The problem.** After a click, the agent needs to know what happened. Did the page change? Did I navigate? Did anything happen at all?

The lazy interface is to answer that question *for* the model: return `page_changed: bool`. I run some heuristic, decide whether the change was "significant," and hand back true or false.

The problem is that *I* don't know what the model considers significant. Maybe it clicked a filter and the URL gained a query param. Maybe it expected a full navigation. Maybe it's tracking the URL to detect a redirect loop. If I collapse all of that into one boolean, I've baked my judgment into the model's only source of truth — and my judgment is wrong sometimes.

**The options I considered.**

1. Return `page_changed: bool` from my own heuristic. Compact, opinionated, lossy.
2. Return nothing and make the model call `get_snapshot` again to diff. Correct but wasteful — that's another 3k tokens just to answer "did the URL change."
3. Return the raw observable facts and let the model interpret them.

**What I chose and why.** Option 3. The `ClickResult` reports what I can actually observe, not what I think it means:

```python
class ClickResult(BaseModel):
    success: bool       # did the click dispatch without throwing?
    previous_url: str   # where we were
    current_url: str    # where we are now
    navigated: bool     # previous_url != current_url — a fact, not a judgment
    elapsed_ms: int
    error: str | None
```

`previous_url` and `current_url` are facts. The model can compare them itself and decide what they mean *in its current context*. I even include `navigated`, but notice it's a pure mechanical comparison — `previous_url != current_url` — not an interpretation of significance:

```python
return ClickResult(
    success=success,
    previous_url=previous_url,
    current_url=current_url,
    navigated=previous_url != current_url,  # a comparison anyone can verify, not my opinion
    elapsed_ms=elapsed_ms,
    error=error,
)
```

The principle: **LLMs reason better over data than over your heuristics.** Give a model `previous_url` and `current_url` and it can answer questions I never anticipated. Give it `page_changed: true` and it can only know what I already decided. The first is a fact it can build on. The second is a conclusion it has to trust.

**The result.** The agent recovers from ambiguous clicks on its own. A click that lands on the same URL but expands a dropdown? `navigated=false`, `success=true` — the model reads that and knows to snapshot rather than assume a new page. I never had to enumerate those cases. I just stopped hiding the facts.

## Decision 4 — Error messages as recovery instructions

**The problem.** Refs go stale. The model takes a snapshot, reasons for a few steps, then clicks `el_12` — but in between, it navigated, and `el_12` was from the *previous* page. The ref map cleared on the new snapshot. Now what?

Most browser automation answers this with a Python traceback, or `KeyError: 'el_12'`, or a generic "click failed." All three are written for a developer reading a log. None of them are written for the consumer that's actually going to read them — the model, mid-loop, deciding its next move.

This is the insight that took me longest to internalize: **in an agent, the error message is not a log line. It's part of the control flow.** The model reads the error and acts on it. So the error should tell it what to *do*, not what went wrong inside my code.

**What I chose.** Every error is phrased as a recovery instruction. When a ref isn't in the map, I don't raise — I return a result whose error field tells the model its exact next move:

```python
selector = self._ref_map.get(ref)
if selector is None:
    return ClickResult(
        success=False,
        previous_url=previous_url,
        current_url=previous_url,
        navigated=False,
        elapsed_ms=int((time.perf_counter() - start) * 1000),
        # not "KeyError" — a sentence telling the model what to do next
        error=f"unknown ref {ref!r}. Call get_snapshot() to refresh available refs.",
    )
```

The same pattern is in `input_text`:

```python
error=f"unknown ref {ref!r}. Call get_snapshot() to refresh available refs."
```

Watch the control flow this creates:

1. Model calls `click("el_12")` with a stale ref.
2. Instead of a crash, it gets back: *"unknown ref 'el_12'. Call get_snapshot() to refresh available refs."*
3. The model reads that, calls `get_snapshot()`, gets a fresh set of refs for the current page.
4. It finds the element it wanted — now `el_7` — and clicks again. Success.

That's a self-healing loop, and I built it entirely out of one well-phrased string. No retry decorator, no orchestration logic. The recovery lives in the error message because the error message is the only thing the model reads when the call fails.

Compare it to the alternative. A traceback gives the model a stack of file paths and a `KeyError`. The best it can do is guess that something went wrong. It doesn't know that the fix is to re-snapshot — unless I tell it, in the one place it's looking.

I also never let real failures crash the tool. Timeouts and Playwright errors get caught and surfaced as plain text in the same `error` field:

```python
try:
    await self.page.locator(selector).click(timeout=timeout_ms)
except PlaywrightTimeoutError:
    error = f"click timed out after {timeout_ms}ms"
except PlaywrightError as e:
    error = str(e)
```

`success=False` plus a human-readable reason, every time. The tool never throws into the agent loop. The model always gets something it can read and act on.

## Results

With all four decisions in place, the MacBook task that opened this post now runs cleanly:

- `navigate` to Mercado Libre.
- `get_snapshot` — ~3k tokens of Markdown, plus a list of interactive elements with refs.
- `input_text` into the search box ref, with `press_enter=True` to submit.
- `get_snapshot` on the results — product names and prices in clean Markdown.
- Read off the prices. Done.

The per-snapshot token cost dropped about 15x, which dropped per-task cost by roughly the same factor, since snapshots dominate the bill. Hallucinated clicks went to zero because the model can only pass refs I issued. And stale-ref failures became a one-step self-correction instead of a crash, because the error told the model exactly what to do.

The agent didn't get smarter. The interface got honest — dense input, an opaque contract, raw facts, and errors written for the thing that reads them.

## What I'd do differently

The ref system has a real weak spot: single-page apps that re-render the DOM between snapshot and click. I stamp `data-paddington-ref` onto live nodes, but if a React re-render replaces those nodes after the snapshot, the attribute vanishes with them, and `[data-paddington-ref="el_12"]` resolves to nothing. Right now that surfaces as a timeout, which is honest but slow. The better fix is explicit stale-ref detection — a cheap check that the tagged node still exists before I attempt the action — so I can return the "call get_snapshot()" recovery message immediately instead of making the model wait 30 seconds for a timeout. It's a known limitation, and it's next on the list.

The second thing is truncation. I cap the Markdown at `max_chars` and report `truncated` and `total_chars` honestly, but the model can't ask for "the next page." If the thing it needs is below the cap, it's stuck re-snapshotting blind. A paginated or scroll-aware snapshot would let the model request more on demand instead of guessing. The current design is the right default — most decisions live above the fold — but it's a default, not a solution, and I'd rather the model could choose.

Neither of these makes the current design wrong. They're the cracks I already know about, which is the only kind worth writing down.
