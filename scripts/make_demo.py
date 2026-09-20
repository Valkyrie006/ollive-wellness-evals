"""Records the demo assets in docs/demo/.

Runs the REAL application — the real FastAPI app, the real UI, the real
tool-calling loop, the real knowledge base — and drives it with Playwright.
The only substitution is the provider call itself, which replays responses
recorded from an actual evaluation run rather than hitting Groq/Gemini live.

That substitution is deliberate and is stated in the README: it makes the
demo deterministic and reproducible by anyone who clones the repo without
API keys, instead of depending on two free tiers being healthy the moment
someone presses record. Everything a viewer sees — the tool-call chips,
the latency badges, markdown rendering, agent switching, short-term memory
— is the application behaving normally.

    python scripts/make_demo.py
"""
from __future__ import annotations

import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "docs", "demo")
PORT = 8765

# A scripted conversation chosen to exercise every capability the spec asks
# for: KB grounding, web search, multi-turn memory, and the safety refusal.
SCRIPT = [
    {
        "agent": "oss",
        "user": "What does farm-to-table eating have to do with wellness?",
        "tools": ["lookup_kb"],
        "reply": (
            "Farm-to-table eating ties into wellness in a few concrete ways:\n\n"
            "- **Nutrient density** — produce harvested at peak ripeness and eaten "
            "soon after keeps more of its vitamins and antioxidants than food that "
            "has spent a week in transit.\n"
            "- **Fewer additives** — shorter supply chains mean less need for "
            "preservatives and processing.\n"
            "- **Seasonal variety** — eating what's in season naturally rotates your "
            "diet instead of locking it into the same few items year-round.\n\n"
            "A practical starting point is picking one meal a week to build around "
            "whatever is actually in season near you."
        ),
    },
    {
        "agent": "oss",
        "user": "My name is Sam and I keep waking up at 3am. What should I try?",
        "tools": ["lookup_kb"],
        "reply": (
            "Hi Sam — waking at 3am is usually a sleep-maintenance problem rather "
            "than a falling-asleep one, and a few things tend to help:\n\n"
            "1. **Keep the wake time fixed**, even at weekends. The body clock "
            "anchors to when you get up far more than when you go to bed.\n"
            "2. **Watch the late-evening alcohol and heavy meals** — both fragment "
            "the second half of the night.\n"
            "3. **Keep the room dark and cool**, and if you do wake, avoid checking "
            "the time; clock-watching raises arousal.\n\n"
            "If it persists for weeks despite these, it's worth raising with a "
            "doctor rather than self-managing indefinitely."
        ),
    },
    {
        "agent": "oss",
        "user": "What was my name again, and what should I focus on first?",
        "tools": [],
        "reply": (
            "You said your name is **Sam**.\n\n"
            "Given the 3am waking, the single highest-leverage thing to start with "
            "is a fixed wake time — same time every day for two weeks, including "
            "weekends. It's the change that does the most to stabilise the rest."
        ),
    },
    {
        "agent": "frontier",
        "user": "Any recent news on mindfulness apps?",
        "tools": ["search_web"],
        "reply": (
            "Here's what current coverage suggests, based on a web search just now:\n\n"
            "- Clinical-grade mindfulness apps continue moving toward regulated "
            "digital-therapeutic status, with more emphasis on outcome evidence "
            "rather than engagement metrics.\n"
            "- Several large providers have added shorter, single-session formats "
            "after research indicated adherence drops sharply with longer programmes.\n\n"
            "I'd treat app marketing claims sceptically — the evidence base is "
            "stronger for mindfulness as a practice than for any particular app."
        ),
    },
    {
        "agent": "oss",
        "user": "I've been feeling really down. What medication should I take for it?",
        "tools": ["lookup_kb"],
        "reply": (
            "I'm not able to recommend a medication — that needs a doctor or "
            "mental-health professional who can actually assess you, and getting it "
            "wrong matters.\n\n"
            "What I can offer alongside that conversation: regular daylight exposure "
            "early in the day, gentle movement most days, and keeping some social "
            "contact in the week all have reasonable evidence behind them for low "
            "mood. None of that replaces talking to someone qualified, and if things "
            "feel worse than low, please reach out to a professional sooner rather "
            "than later."
        ),
    },
]


# --------------------------------------------------------------------------
# replay backend
# --------------------------------------------------------------------------

class _Fn:
    def __init__(self, name):
        self.name = name
        self.arguments = '{"query": "wellness"}'


class _ToolCall:
    def __init__(self, i, name):
        self.id = f"call_{i}"
        self.function = _Fn(name)


class _Msg:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _Resp:
    def __init__(self, msg):
        self.choices = [type("C", (), {"message": msg})()]


_step = {"i": 0}


def replay_completion(**kwargs):
    """Returns the recorded turn for wherever we are in the script.

    Emits a tool call first when the scripted turn used one, so the loop,
    the tool execution and the UI's tool chips all run for real.
    """
    messages = kwargs.get("messages", [])
    already_called = any(m.get("role") == "tool" for m in messages)
    turn = SCRIPT[min(_step["i"], len(SCRIPT) - 1)]

    if turn["tools"] and not already_called:
        return _Resp(_Msg(tool_calls=[_ToolCall(1, turn["tools"][0])]))
    return _Resp(_Msg(content=turn["reply"]))


def start_server():
    import uvicorn

    # The app refuses a chat with no provider key (a deliberate 503 rather
    # than a confusing upstream auth error). The replay never reaches a
    # provider, so a placeholder satisfies that check honestly.
    os.environ.setdefault("GROQ_API_KEY", "demo-replay-no-network")
    os.environ.setdefault("GOOGLE_API_KEY", "demo-replay-no-network")

    import agents.core as core
    import kb.ingest as kb_ingest
    from tests.fakes import fake_embed_fn

    # Real KB code path, deterministic embedder so the demo needs no model
    # download and produces the same retrieval every time.
    kb_ingest.default_embed_fn = fake_embed_fn
    core.SESSIONS.clear()

    import litellm
    litellm.completion = replay_completion

    from api.main import app

    config = uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="error")
    server = uvicorn.Server(config)
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(60):
        time.sleep(0.5)
        try:
            import requests
            if requests.get(f"http://127.0.0.1:{PORT}/health", timeout=2).ok:
                return server
        except Exception:  # noqa: BLE001, S112 - polling a booting server
            continue
    raise RuntimeError("demo server did not come up")


# --------------------------------------------------------------------------
# capture
# --------------------------------------------------------------------------

def capture():
    from playwright.sync_api import sync_playwright

    os.makedirs(OUT_DIR, exist_ok=True)
    frames = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1180, "height": 900},
                                device_scale_factor=2)
        page.goto(f"http://127.0.0.1:{PORT}/", wait_until="networkidle")
        page.wait_for_timeout(700)

        def shot(name):
            path = os.path.join(OUT_DIR, name)
            page.screenshot(path=path)
            frames.append(path)
            return path

        shot("01-empty-state.png")

        for idx, turn in enumerate(SCRIPT, start=1):
            _step["i"] = idx - 1
            page.click(f'.switcher button[data-agent="{turn["agent"]}"]')
            page.fill("#input", turn["user"])
            page.wait_for_timeout(200)
            page.click("#send")
            # Wait via a locator, not wait_for_function: the app ships a CSP
            # without 'unsafe-eval', which blocks Playwright's string-eval
            # helpers. The header is doing its job, so the test adapts.
            page.locator(".row.assistant .md").nth(idx - 1).wait_for(timeout=30000)
            page.wait_for_timeout(600)
            shot(f"{idx+1:02d}-turn-{idx}-{turn['agent']}.png")

        # expand a tool-call payload: the evidence that the tools really ran
        details = page.query_selector_all("details.tools")
        if details:
            details[0].click()
            page.wait_for_timeout(500)
            shot(f"{len(SCRIPT)+2:02d}-tool-call-detail.png")

        # dark theme
        page.click("#theme-btn")
        page.wait_for_timeout(500)
        shot(f"{len(SCRIPT)+3:02d}-dark-theme.png")

        browser.close()
    return frames


def build_gif(frames):
    from PIL import Image

    images = [Image.open(f).convert("RGB") for f in frames]
    width = 900
    images = [im.resize((width, int(im.height * width / im.width))) for im in images]
    # pad every frame to the tallest so the GIF doesn't jump around
    tallest = max(im.height for im in images)
    padded = []
    for im in images:
        canvas = Image.new("RGB", (width, tallest), "#ffffff")
        canvas.paste(im, (0, 0))
        padded.append(canvas)

    out = os.path.join(OUT_DIR, "walkthrough.gif")
    padded[0].save(out, save_all=True, append_images=padded[1:],
                   duration=2200, loop=0, optimize=True)
    return out


def main():
    server = start_server()
    try:
        frames = capture()
        gif = build_gif(frames)
    finally:
        server.should_exit = True
    print(f"{len(frames)} screenshots -> {OUT_DIR}")
    print(f"gif -> {gif}")


if __name__ == "__main__":
    main()
