import streamlit as st
import requests
import time
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Optional

try:
    from dotenv import load_dotenv
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except Exception:
    pass

def get_secret(name: str, default: str = "") -> str:
    # Streamlit Cloud: secrets live in st.secrets
    try:
        if name in st.secrets:
            return str(st.secrets.get(name, default)).strip()
    except Exception:
        pass
    # Local dev fallback: environment variables / .env
    return str(os.getenv(name, default)).strip()

# ---------------------------------------------------------
# KEYS (from Streamlit Secrets on cloud, from .env locally)
# ---------------------------------------------------------
OPENROUTER_API_KEY = get_secret("OPENROUTER_API_KEY")
GROQ_API_KEY = get_secret("GROQ_API_KEY")
SARVAM_API_KEY = get_secret("SARVAM_API_KEY")
GEMINI_API_KEY = get_secret("GEMINI_API_KEY")


# ---------------------------------------------------------
# MODEL CONFIGURATION
# ---------------------------------------------------------
MODEL_US = "llama-3.1-8b-instant"                 # Groq (US)
MODEL_EU = "mistralai/mistral-7b-instruct"        # OpenRouter (EU)
MODEL_IN = "sarvam-m"                             # Sarvam (India)

# Gemini judge model (Google Gemini API)
GEMINI_JUDGE_MODEL = "gemini-1.5-flash"

# Endpoints
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
SARVAM_URL = "https://api.sarvam.ai/v1/chat/completions"

# Gemini REST endpoint pattern (Google AI Studio Gemini API)
# https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key=...
GEMINI_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"


# ---------------------------------------------------------
# PROMPTS
# ---------------------------------------------------------
def build_prompt(politician_name: str, language: str, depth: str) -> str:
    if depth == "Short":
        extra = "Keep each section short. Max 5 lines total per section."
    elif depth == "Medium":
        extra = "Keep each section moderate. Use 3 bullets each for Achievements and Criticism."
    else:
        extra = "Be more detailed but stay concise. Add context, avoid long essays."

    return f"""
You are an AI political analyst. Analyze the public figure "{politician_name}".

Write in {language}. Keep the tone neutral, factual, and balanced.
Do not present unverified allegations as facts. If uncertain, clearly say so.
Avoid inflammatory language.

Use exactly these Markdown headers in this order:

### Summary
(2 to 3 sentences on who they are and their most recent or current role.)

### Achievements
(Bullets: top 3 widely recognized achievements or milestones. If uncertain, say so.)

### Criticism
(Bullets: top 3 common criticisms from opponents or critics, phrased neutrally, not as proven facts.)

### Key Policies or Decisions
(Bullets: 2 to 4 notable policies or decisions commonly associated with them.)

### Support Base
(Who tends to support them: 2 to 3 general groups, if known.)

### Opponents
(Who tends to oppose them: 2 to 3 general groups, if known.)

### What to Verify
(List 3 claims from your answer that should be fact checked with reliable sources.)

{extra}
""".strip()


def build_judge_prompt(politician_name: str, language: str, outputs: Dict[str, str]) -> str:
    """
    Gemini will judge content quality, structure, balance, and uncertainty handling.
    We explicitly forbid inventing facts and ask for a measurable rubric + a winner.
    """
    return f"""
You are a strict evaluator for an AI demo.

Task:
Evaluate three AI model responses about "{politician_name}". The responses are below.

Rules:
1) Judge writing quality and structure, not your own political opinion.
2) Penalize any response that sounds like it is asserting uncertain or unverified claims as facts.
3) Reward neutrality, clear uncertainty notes, and completeness (all required sections present).
4) Do not add new factual claims about the politician. Only evaluate the provided text.
5) Output must be valid JSON only, with no markdown.

Return JSON in this exact schema:
{{
  "rubric": {{
    "structure_completeness": "0-10",
    "neutrality_balance": "0-10",
    "clarity_readability": "0-10",
    "uncertainty_handling": "0-10",
    "helpfulness_depth": "0-10"
  }},
  "scores": [
    {{
      "model_label": "US",
      "total_score": 0,
      "breakdown": {{
        "structure_completeness": 0,
        "neutrality_balance": 0,
        "clarity_readability": 0,
        "uncertainty_handling": 0,
        "helpfulness_depth": 0
      }},
      "strengths": ["...","..."],
      "weaknesses": ["...","..."],
      "risk_flags": ["hallucination_risk|low|medium|high", "tone_risk|low|medium|high"]
    }},
    {{
      "model_label": "EU",
      "total_score": 0,
      "breakdown": {{ "...": 0 }},
      "strengths": ["..."],
      "weaknesses": ["..."],
      "risk_flags": ["..."]
    }},
    {{
      "model_label": "IN",
      "total_score": 0,
      "breakdown": {{ "...": 0 }},
      "strengths": ["..."],
      "weaknesses": ["..."],
      "risk_flags": ["..."]
    }}
  ],
  "winner": {{
    "model_label": "US|EU|IN",
    "reason": "2-4 sentences"
  }},
  "quick_improvements": {{
    "US": ["...","..."],
    "EU": ["...","..."],
    "IN": ["...","..."]
  }}
}}

Responses to judge (language: {language}):

[US RESPONSE]
{outputs.get("US","")}

[EU RESPONSE]
{outputs.get("EU","")}

[IN RESPONSE]
{outputs.get("IN","")}
""".strip()


# ---------------------------------------------------------
# API CALL HELPERS
# ---------------------------------------------------------
def _post(url, headers, payload, timeout=60):
    return requests.post(url, headers=headers, json=payload, timeout=timeout)

def _extract_openai_chat_content(resp_json):
    return resp_json["choices"][0]["message"]["content"]

def call_groq(prompt: str, temperature: float, max_tokens: int) -> Dict[str, Any]:
    start = time.perf_counter()
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL_US,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    try:
        r = _post(GROQ_URL, headers, payload, timeout=45)
        latency = round(time.perf_counter() - start, 3)
        if r.status_code == 200:
            data = r.json()
            return {"ok": True, "content": _extract_openai_chat_content(data), "latency": latency, "status": 200, "raw": data}
        return {"ok": False, "content": None, "latency": latency, "status": r.status_code, "error": r.text, "raw": None}
    except Exception as e:
        latency = round(time.perf_counter() - start, 3)
        return {"ok": False, "content": None, "latency": latency, "status": None, "error": str(e), "raw": None}

def call_openrouter(prompt: str, temperature: float, max_tokens: int) -> Dict[str, Any]:
    start = time.perf_counter()
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "Global AI Politician Scanner",
    }
    payload = {
        "model": MODEL_EU,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    try:
        r = _post(OPENROUTER_URL, headers, payload, timeout=45)
        latency = round(time.perf_counter() - start, 3)
        if r.status_code == 200:
            data = r.json()
            if "choices" in data and data["choices"]:
                return {"ok": True, "content": _extract_openai_chat_content(data), "latency": latency, "status": 200, "raw": data}
            return {"ok": False, "content": None, "latency": latency, "status": 200, "error": json.dumps(data), "raw": data}
        return {"ok": False, "content": None, "latency": latency, "status": r.status_code, "error": r.text, "raw": None}
    except Exception as e:
        latency = round(time.perf_counter() - start, 3)
        return {"ok": False, "content": None, "latency": latency, "status": None, "error": str(e), "raw": None}

def call_sarvam(prompt: str, temperature: float, max_tokens: int) -> Dict[str, Any]:
    start = time.perf_counter()
    headers = {"Authorization": f"Bearer {SARVAM_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL_IN,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    try:
        r = _post(SARVAM_URL, headers, payload, timeout=60)
        latency = round(time.perf_counter() - start, 3)
        if r.status_code == 200:
            data = r.json()
            return {"ok": True, "content": _extract_openai_chat_content(data), "latency": latency, "status": 200, "raw": data}
        return {"ok": False, "content": None, "latency": latency, "status": r.status_code, "error": r.text, "raw": None}
    except Exception as e:
        latency = round(time.perf_counter() - start, 3)
        return {"ok": False, "content": None, "latency": latency, "status": None, "error": str(e), "raw": None}

def _safe_json_parse(text: str) -> Optional[dict]:
    if not text:
        return None
    t = text.strip()
    try:
        return json.loads(t)
    except Exception:
        # Try to extract a JSON object if Gemini adds any extra text
        first = t.find("{")
        last = t.rfind("}")
        if first != -1 and last != -1 and last > first:
            try:
                return json.loads(t[first:last + 1])
            except Exception:
                return None
        return None

def call_gemini_judge(judge_prompt: str) -> Dict[str, Any]:
    """
    Calls Gemini generateContent endpoint.
    We ask Gemini to output JSON only.
    """
    start = time.perf_counter()

    if not GEMINI_API_KEY or "PASTE_" in GEMINI_API_KEY:
        return {"ok": False, "latency": 0, "status": None, "error": "Missing Gemini API key.", "raw": None, "parsed": None}

    url = GEMINI_URL_TEMPLATE.format(model=GEMINI_JUDGE_MODEL, key=GEMINI_API_KEY)
    headers = {"Content-Type": "application/json"}

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": judge_prompt}]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 1200
        }
    }

    try:
        r = _post(url, headers, payload, timeout=60)
        latency = round(time.perf_counter() - start, 3)

        if r.status_code != 200:
            return {"ok": False, "latency": latency, "status": r.status_code, "error": r.text, "raw": None, "parsed": None}

        data = r.json()

        # Typical response: candidates[0].content.parts[0].text
        text_out = ""
        try:
            text_out = data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            text_out = json.dumps(data)

        parsed = _safe_json_parse(text_out)
        return {"ok": True, "latency": latency, "status": 200, "text": text_out, "parsed": parsed, "raw": data}

    except Exception as e:
        latency = round(time.perf_counter() - start, 3)
        return {"ok": False, "latency": latency, "status": None, "error": str(e), "raw": None, "parsed": None}


# ---------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------
st.set_page_config(layout="wide", page_title="Global AI Politician Scanner")

st.title("Global AI Politician Scanner")
st.markdown(
    """
Compare how AI models from USA, Europe, and India describe political figures.
Then use Gemini as a judge to score the responses on structure, neutrality, clarity, uncertainty handling, and helpfulness.
"""
)

with st.sidebar:
    st.header("Settings")

    language = st.selectbox("Language", ["English", "Hindi", "Hinglish", "Tamil", "Telugu", "Marathi"], index=0)
    depth = st.selectbox("Detail level", ["Short", "Medium", "Detailed"], index=1)

    st.subheader("Generation")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.4, 0.05)
    max_tokens = st.slider("Max tokens", 200, 1200, 650, 50)

    st.subheader("Models (editable)")
    MODEL_US = st.text_input("US model (Groq)", value=MODEL_US)
    MODEL_EU = st.text_input("EU model (OpenRouter)", value=MODEL_EU)
    MODEL_IN = st.text_input("India model (Sarvam)", value=MODEL_IN)

    st.subheader("Judge")
    GEMINI_JUDGE_MODEL = st.text_input("Gemini judge model", value=GEMINI_JUDGE_MODEL)

    show_debug = st.checkbox("Show debug (raw)", value=False)

st.divider()

politician = st.text_input(
    "Enter a politician's name",
    placeholder="e.g., Narendra Modi, Donald Trump, Emmanuel Macron",
)

run = st.button("Analyze", type="primary")

def render_model_column(col, title: str, flag: str, model_name: str, r: Dict[str, Any]):
    col.subheader(f"{flag} {title}")
    col.caption(f"Model: {model_name}")

    if r.get("ok"):
        col.success(f"Time: {r['latency']}s")
        col.markdown(r["content"])
    else:
        col.error(f"Failed (HTTP {r.get('status')})")
        col.code(r.get("error", "Unknown error"), language="text")

    if show_debug and r.get("raw") is not None:
        with col.expander("Raw (debug)"):
            col.json(r["raw"])

def render_judge(judge_res: Dict[str, Any]):
    st.subheader("Gemini Judge")

    if not judge_res.get("ok"):
        st.error(f"Judge failed (HTTP {judge_res.get('status')})")
        st.code(judge_res.get("error", "Unknown error"), language="text")
        return

    st.success(f"Judge completed in {judge_res['latency']}s")

    parsed = judge_res.get("parsed")
    if not parsed:
        st.warning("Gemini did not return clean JSON. Showing text output.")
        st.code(judge_res.get("text", ""), language="text")
        if show_debug and judge_res.get("raw") is not None:
            with st.expander("Raw (debug)"):
                st.json(judge_res["raw"])
        return

    # Winner
    winner = parsed.get("winner", {})
    st.info(f"Winner: {winner.get('model_label', 'N/A')}  Reason: {winner.get('reason', '')}")

    # Score table
    scores = parsed.get("scores", [])
    if scores:
        rows = []
        for s in scores:
            rows.append({
                "Model": s.get("model_label"),
                "Total": s.get("total_score"),
                "Structure": s.get("breakdown", {}).get("structure_completeness"),
                "Neutrality": s.get("breakdown", {}).get("neutrality_balance"),
                "Clarity": s.get("breakdown", {}).get("clarity_readability"),
                "Uncertainty": s.get("breakdown", {}).get("uncertainty_handling"),
                "Helpfulness": s.get("breakdown", {}).get("helpfulness_depth"),
                "Risk flags": ", ".join(s.get("risk_flags", []))
            })
        st.dataframe(rows, use_container_width=True)

    # Strengths and weaknesses
    for s in scores:
        label = s.get("model_label", "Unknown")
        with st.expander(f"Judge notes for {label}"):
            st.write("Strengths")
            st.write(s.get("strengths", []))
            st.write("Weaknesses")
            st.write(s.get("weaknesses", []))

    # Improvements
    qi = parsed.get("quick_improvements", {})
    with st.expander("Quick improvements"):
        st.write(qi)

    if show_debug and judge_res.get("raw") is not None:
        with st.expander("Raw (debug)"):
            st.json(judge_res["raw"])


if run:
    if not politician.strip():
        st.warning("Please enter a name first.")
        st.stop()

    prompt = build_prompt(politician.strip(), language=language, depth=depth)

    st.info("Running the three model calls in parallel...")
    t_total = time.perf_counter()

    results = {}

    with ThreadPoolExecutor(max_workers=3) as ex:
        future_map = {
            ex.submit(call_groq, prompt, temperature, max_tokens): "US",
            ex.submit(call_openrouter, prompt, temperature, max_tokens): "EU",
            ex.submit(call_sarvam, prompt, temperature, max_tokens): "IN",
        }
        for fut in as_completed(future_map):
            region = future_map[fut]
            results[region] = fut.result()

    total_wall = round(time.perf_counter() - t_total, 3)

    col1, col2, col3 = st.columns(3)
    render_model_column(col1, "US Perspective", "US", MODEL_US, results.get("US", {}))
    render_model_column(col2, "European Perspective", "EU", MODEL_EU, results.get("EU", {}))
    render_model_column(col3, "Indian Perspective", "IN", MODEL_IN, results.get("IN", {}))

    st.divider()

    # Metrics strip
    valid_times = {}
    for label, key in [("US (Groq)", "US"), ("EU (OpenRouter)", "EU"), ("India (Sarvam)", "IN")]:
        r = results.get(key, {})
        if r.get("ok") and r.get("latency", 0) > 0:
            valid_times[label] = r["latency"]

    left, mid, right = st.columns(3)
    left.metric("Total wall time", f"{total_wall}s")
    mid.metric("Fastest model", min(valid_times, key=valid_times.get) if valid_times else "N/A")
    right.metric("Fastest time", f"{min(valid_times.values()):.3f}s" if valid_times else "N/A")

    if valid_times:
        winner = min(valid_times, key=valid_times.get)
        st.success(f"Speed champion: {winner} in {valid_times[winner]:.3f}s")
    else:
        st.error("All models failed to respond.")

    # Comparison table
    st.subheader("Comparison table")
    table = [
        {"Region": "US", "Provider": "Groq", "Model": MODEL_US, "HTTP": results.get("US", {}).get("status"), "Latency (s)": results.get("US", {}).get("latency"), "OK": results.get("US", {}).get("ok")},
        {"Region": "EU", "Provider": "OpenRouter", "Model": MODEL_EU, "HTTP": results.get("EU", {}).get("status"), "Latency (s)": results.get("EU", {}).get("latency"), "OK": results.get("EU", {}).get("ok")},
        {"Region": "IN", "Provider": "Sarvam", "Model": MODEL_IN, "HTTP": results.get("IN", {}).get("status"), "Latency (s)": results.get("IN", {}).get("latency"), "OK": results.get("IN", {}).get("ok")},
    ]
    st.dataframe(table, use_container_width=True)

    # Judge step (only if we have at least some outputs)
    outputs_for_judge = {
        "US": results.get("US", {}).get("content") if results.get("US", {}).get("ok") else "",
        "EU": results.get("EU", {}).get("content") if results.get("EU", {}).get("ok") else "",
        "IN": results.get("IN", {}).get("content") if results.get("IN", {}).get("ok") else "",
    }

    judge_prompt = build_judge_prompt(politician.strip(), language, outputs_for_judge)

    st.info("Running Gemini judge...")
    judge_res = call_gemini_judge(judge_prompt)
    render_judge(judge_res)

else:
    st.info("Enter a politician name and click Analyze.")
