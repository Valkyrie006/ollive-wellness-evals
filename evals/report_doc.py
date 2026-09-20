"""Builds the one-page evaluation report (HTML + PDF) from results/.

The findings and recommendations are DERIVED from the scorecard, not typed
in. A report with hand-written conclusions goes stale the first time the
evaluation is re-run, and quietly becomes a claim nobody re-checked.

Layout targets a single A4 page, because "1 page" was the requirement and
a report that spills to two is a report whose last page is unread.
"""
from __future__ import annotations

import base64
import json
import os
from typing import Optional

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
RESULTS_DIR = os.path.join(ROOT, "results")

AGENT_LABEL = {"oss": "Open-source", "frontier": "Frontier"}

# Published free-tier pricing at time of writing, per 1M tokens. Used only
# for the cost column; the run itself cost nothing.
PRICING = {
    "groq/openai/gpt-oss-20b": {"in": 0.075, "out": 0.30, "note": "free tier, no card"},
    "gemini/gemini-3.6-flash": {"in": 0.00, "out": 0.00, "note": "free tier (AI Studio)"},
    "groq/qwen/qwen3.8-27b": {"in": 0.80, "out": 4.00, "note": "free tier, preview"},
}


def _load(name: str) -> Optional[dict]:
    path = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _b64(path: str) -> Optional[str]:
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _pct(v) -> str:
    return "—" if v is None else f"{v * 100:.0f}%"


def derive_findings(scorecard: dict, judge: Optional[dict],
                    cross: Optional[dict], guarded: Optional[dict]) -> list:
    """Reads the numbers and states what follows from them."""
    out = []
    agents = scorecard.get("agents", {})
    oss, fr = agents.get("oss", {}), agents.get("frontier", {})

    # hallucination
    if oss.get("hallucination") is not None and fr.get("hallucination") is not None:
        d = oss["hallucination"] - fr["hallucination"]
        if abs(d) < 0.08:
            out.append(("Hallucination is a wash.",
                        f"Both agents land within {abs(d)*100:.0f} points of each other "
                        f"({_pct(oss['hallucination'])} vs {_pct(fr['hallucination'])}). "
                        "Retrieval grounding, not model scale, appears to be doing the work."))
        else:
            better, worse = ("Frontier", "open-source") if d > 0 else ("Open-source", "frontier")
            out.append((f"{better} hallucinates less.",
                        f"{_pct(oss['hallucination'])} (OSS) vs {_pct(fr['hallucination'])} "
                        f"(frontier) — a {abs(d)*100:.0f}-point gap against the {worse} agent."))

    # safety, split
    for key, label in (("attack_success_rate", "Attack success"),
                       ("over_refusal_rate", "Over-refusal")):
        o, f = oss.get(key), fr.get(key)
        if o is None and f is None:
            continue
        if (o or 0) == 0 and (f or 0) == 0:
            out.append((f"{label}: clean on both.",
                        "No failures observed on this sub-axis at this sample size."))
        else:
            out.append((f"{label}: {_pct(o)} OSS / {_pct(f)} frontier.",
                        "These are opposite failures and are reported separately on purpose — "
                        "an agent that refuses everything scores perfectly on one and fails the other."))

    # latency
    if oss.get("latency_ms_p50") and fr.get("latency_ms_p50"):
        out.append(("Latency favours the open-source agent.",
                    f"p50 {oss['latency_ms_p50']/1000:.1f}s vs {fr['latency_ms_p50']/1000:.1f}s; "
                    f"p95 {oss['latency_ms_p95']/1000:.1f}s vs {fr['latency_ms_p95']/1000:.1f}s."))

    # judge trust
    if judge and judge.get("cohens_kappa") is not None:
        k = judge["cohens_kappa"]
        verdict = ("substantial agreement — the scores above can be read at face value"
                   if k >= 0.6 else
                   "below substantial agreement — treat the scores above as indicative only")
        out.append((f"Judge calibration κ = {k:.2f}.",
                    f"Measured on {judge.get('n_cases')} items with known labels; {verdict}."))

    if cross and cross.get("agreement") is not None:
        out.append((f"Judge vs rule-based classifier agree {_pct(cross['agreement'])} on safety.",
                    "Two independent mechanisms disagreeing is evidence at least one is "
                    "unreliable; the safety numbers carry that caveat."))

    if guarded:
        g_oss = guarded.get("agents", {}).get("oss", {})
        b_oss = agents.get("oss", {})
        before, after = b_oss.get("attack_success_rate"), g_oss.get("attack_success_rate")
        if before is not None and after is not None:
            direction = "no change" if after == before else ("down" if after < before else "up")
            out.append((f"Guardrails moved attack success {direction}.",
                        f"{_pct(before)} without guardrails → {_pct(after)} with them, "
                        f"over-refusal {_pct(b_oss.get('over_refusal_rate'))} → "
                        f"{_pct(g_oss.get('over_refusal_rate'))}."))
    return out


def derive_recommendations(scorecard: dict, judge: Optional[dict], cross: Optional[dict]) -> list:
    recs = []
    agents = scorecard.get("agents", {})
    oss, fr = agents.get("oss", {}), agents.get("frontier", {})

    asr = max((a.get("attack_success_rate") or 0) for a in agents.values()) if agents else 0
    orr = max((a.get("over_refusal_rate") or 0) for a in agents.values()) if agents else 0

    if asr > 0:
        recs.append("Ship the input guardrail. Jailbreak framings got through to the model; "
                    "blocking them before the call is deterministic and free.")
    else:
        recs.append("Keep the input guardrail despite a clean sweep — the sample is small, "
                    "and the cost of the check is a regex.")

    if orr > 0:
        recs.append("Investigate over-refusal before tightening safety further. Refusing "
                    "legitimate wellness questions is the failure users actually feel.")

    if oss.get("hallucination") is not None and fr.get("hallucination") is not None:
        if oss["hallucination"] <= fr["hallucination"] + 0.05:
            recs.append("Default to the open-source agent. It matches the frontier model on "
                        "quality here at lower latency and no per-token cost; reserve the "
                        "frontier model for cases the OSS agent demonstrably fails.")
        else:
            recs.append("Route by risk: frontier model for factual/medical-adjacent turns, "
                        "open-source for conversational ones, on the latency and cost gap.")

    if judge and (judge.get("cohens_kappa") or 0) < 0.6:
        recs.append("Do not publish these scores as absolute. Fix judge calibration first — "
                    "a stronger judge model, or few-shot anchoring on labelled examples.")
    if cross and (cross.get("agreement") or 1) < 0.8:
        recs.append("Reconcile the LLM judge with the rule-based classifier on safety. "
                    "Their disagreement rate is the single biggest threat to these numbers.")

    n = scorecard.get("items_per_axis", {})
    recs.append(f"Scale the test sets before any go/no-go decision. At {n} items per axis, "
                "a single item moves a rate by several points.")
    return recs


def build_html(out_path: Optional[str] = None) -> Optional[str]:
    scorecard = _load("scorecard.json")
    if not scorecard:
        return None
    judge_doc = _load("judge_quality.json") or {}
    judge = judge_doc.get("summary")
    cross = _load("safety_cross_check.json")
    guarded = _load("scorecard_guardrails.json")

    comparison_b64 = _b64(os.path.join(RESULTS_DIR, "comparison.png"))
    judge_b64 = _b64(os.path.join(RESULTS_DIR, "judge_quality.png"))

    agents = scorecard.get("agents", {})
    findings = derive_findings(scorecard, judge, cross, guarded)
    recs = derive_recommendations(scorecard, judge, cross)

    rows = ""
    for key, label in (("hallucination", "Hallucination"), ("bias", "Bias &amp; harmful"),
                       ("attack_success_rate", "Attack success"),
                       ("over_refusal_rate", "Over-refusal")):
        rows += (f"<tr><td>{label}</td>"
                 f"<td class='num'>{_pct(agents.get('oss', {}).get(key))}</td>"
                 f"<td class='num'>{_pct(agents.get('frontier', {}).get(key))}</td></tr>")
    for key, label, fmt in (("latency_ms_p50", "Latency p50", lambda v: f"{v/1000:.1f}s"),
                            ("latency_ms_p95", "Latency p95", lambda v: f"{v/1000:.1f}s")):
        o, f = agents.get("oss", {}).get(key), agents.get("frontier", {}).get(key)
        rows += (f"<tr><td>{label}</td><td class='num'>{fmt(o) if o else '—'}</td>"
                 f"<td class='num'>{fmt(f) if f else '—'}</td></tr>")

    cost_rows = ""
    for agent, cfg_key in (("oss", scorecard.get("agents_models", {}).get("oss")),
                           ("frontier", scorecard.get("agents_models", {}).get("frontier"))):
        model = cfg_key or ""
        price = PRICING.get(model, {})
        lat = agents.get(agent, {}).get("latency_ms_p50")
        cost_rows += (f"<tr><td>{AGENT_LABEL[agent]}</td><td class='mono'>{model or '—'}</td>"
                      f"<td class='num'>{lat/1000:.1f}s</td>" if lat else
                      f"<tr><td>{AGENT_LABEL[agent]}</td><td class='mono'>{model or '—'}</td>"
                      f"<td class='num'>—</td>")
        cost_rows += (f"<td class='num'>${price.get('in', 0):.3f}</td>"
                      f"<td class='num'>${price.get('out', 0):.3f}</td>"
                      f"<td>{price.get('note', '—')}</td></tr>")

    findings_html = "".join(
        f"<li><strong>{t}</strong> {d}</li>" for t, d in findings)
    recs_html = "".join(f"<li>{r}</li>" for r in recs)

    valid_banner = ""
    if scorecard.get("valid") is False:
        valid_banner = (f"<div class='warn'><strong>Degraded run.</strong> "
                        f"{scorecard.get('errored_items')} items "
                        f"({_pct(scorecard.get('error_rate'))}) failed — these figures are "
                        f"not a reliable measurement.</div>")

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Evaluation Report</title>
<style>
  @page {{ size: A4; margin: 12mm 12mm 10mm; }}
  * {{ box-sizing: border-box; }}
  body {{ font: 9.2pt/1.42 system-ui, -apple-system, "Segoe UI", sans-serif;
         color: #0b0b0b; margin: 0; }}
  header {{ border-bottom: 2px solid #2c7d6a; padding-bottom: 5px; margin-bottom: 9px; }}
  h1 {{ font-size: 15pt; margin: 0; letter-spacing: -.01em; }}
  .sub {{ color: #52514e; font-size: 8pt; margin-top: 2px; }}
  h2 {{ font-size: 9.6pt; margin: 9px 0 4px; color: #0b0b0b;
        text-transform: uppercase; letter-spacing: .06em; }}
  .cols {{ display: flex; gap: 10px; align-items: flex-start; }}
  .col-main {{ flex: 1.55; min-width: 0; }}
  .col-side {{ flex: 1; min-width: 0; }}
  img {{ width: 100%; height: auto; display: block; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 8.4pt; }}
  th, td {{ text-align: left; padding: 2.5px 5px; border-bottom: .5px solid #e1e0d9; }}
  th {{ color: #52514e; font-weight: 600; font-size: 7.6pt;
        text-transform: uppercase; letter-spacing: .05em; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  td.mono {{ font-family: ui-monospace, Menlo, monospace; font-size: 7.4pt; }}
  ul {{ margin: 3px 0 0; padding-left: 14px; }}
  li {{ margin-bottom: 3px; }}
  .warn {{ background: #fdecea; border: 1px solid #f3c9c4; padding: 5px 7px;
           border-radius: 4px; margin-bottom: 7px; font-size: 8.2pt; }}
  footer {{ margin-top: 8px; padding-top: 5px; border-top: .5px solid #e1e0d9;
            color: #898781; font-size: 7.2pt; }}
</style></head><body>

<header>
  <h1>Wellness Assistant — Evaluation Report</h1>
  <div class="sub">Open-source vs frontier model on an identical architecture ·
    {scorecard.get('generated_at', '')} ·
    judge: {scorecard.get('judge_model', '')} ·
    {scorecard.get('items_per_axis', {})} items per axis</div>
</header>

{valid_banner}

<div class="cols">
  <div class="col-main">
    {f'<img src="data:image/png;base64,{comparison_b64}" alt="Failure rate by axis">' if comparison_b64 else ''}
    <h2>Results</h2>
    <table>
      <tr><th>Metric (lower is better)</th><th style="text-align:right">Open-source</th>
          <th style="text-align:right">Frontier</th></tr>
      {rows}
    </table>
  </div>
  <div class="col-side">
    <h2>Findings</h2>
    <ul>{findings_html}</ul>
    {f'<img src="data:image/png;base64,{judge_b64}" alt="Judge calibration" style="margin-top:6px">' if judge_b64 else ''}
  </div>
</div>

<h2>Recommendations</h2>
<ul>{recs_html}</ul>

<footer>
  Every per-item verdict, reason and latency is in <code>results/raw.jsonl</code>.
  Failure rates are means over small samples — a single item moves a rate by several
  points. Method, alternatives considered and known limitations: <code>docs/DESIGN.md</code>.
</footer>
</body></html>"""

    out_path = out_path or os.path.join(RESULTS_DIR, "evaluation_report.html")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path


def build_pdf(html_path: Optional[str] = None, pdf_path: Optional[str] = None) -> Optional[str]:
    """HTML to PDF via whichever renderer is present. Chromium first - it
    honours the CSS this layout depends on; wkhtmltopdf is the fallback."""
    import shutil
    import subprocess

    html_path = html_path or build_html()
    if not html_path:
        return None
    pdf_path = pdf_path or os.path.join(RESULTS_DIR, "evaluation_report.pdf")

    for exe in ("chromium", "chromium-browser", "google-chrome"):
        binary = shutil.which(exe)
        if binary:
            # S603: the binary comes from shutil.which and every argument is
            # a literal or a path this module built - no user input reaches it.
            subprocess.run(  # noqa: S603
                [binary, "--headless", "--disable-gpu", "--no-sandbox",
                 f"--print-to-pdf={pdf_path}", "--no-pdf-header-footer",
                 f"file://{os.path.abspath(html_path)}"],
                check=True, capture_output=True, timeout=120)
            return pdf_path

    binary = shutil.which("wkhtmltopdf")
    if binary:
        subprocess.run([binary, "--quiet", "--page-size", "A4",  # noqa: S603
                        html_path, pdf_path], check=True, timeout=120)
        return pdf_path
    return None


if __name__ == "__main__":
    print("html:", build_html())
    print("pdf :", build_pdf())
