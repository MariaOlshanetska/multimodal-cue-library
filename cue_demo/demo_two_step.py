#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from llama_cpp import Llama

MODALITIES = ("body_gesture", "facial_expression", "prosody", "vocal_burst")

ALLOWED_INTENTS = [
    # --- Interaction management ---
    "backchannel_ack", "turn_hold", "elicit_response", "address_other",
    # --- Epistemic / trouble / repair ---
    "uncertainty", "repair_initiation", "cognitive_load", "cognitive_load_resolved",
    # --- Discourse structuring ---
    "emphasis", "introduce_topic", "topic_continuation", "explain_structure",
    "explain_sequence", "explain_repetition", "summarize", "topic_closure",
    # --- Stance / social meaning ---
    "affiliation", "mitigation",
]

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text)

def detokenize(tokens: List[str]) -> str:
    out = []
    for i, t in enumerate(tokens):
        if i == 0:
            out.append(t)
            continue
        if re.match(r"^[\.\,\?\!\:\;\)\]\}]+$", t):
            out.append(t)
        else:
            out.append(" " + t)
    return "".join(out)

def extract_json(text: str) -> Optional[Dict[str, Any]]:
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def load_library(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["cues"]

def _matches(value: str, allowed: Any) -> bool:
    if allowed is None:
        return True
    if isinstance(allowed, list):
        return ("ANY" in allowed) or (value in allowed)
    return True

def cue_applicable(cue: Dict[str, Any], ctx: Dict[str, Any]) -> bool:
    c = cue.get("constraints", {}) or {}
    return (
        _matches(ctx["language"], c.get("languages", ["ANY"])) and
        _matches(ctx["culture"], c.get("cultures", ["ANY"])) and
        _matches(ctx["formality"], c.get("formality", ["ANY"])) and
        _matches(ctx["roles"], c.get("roles", ["ANY"]))
    )

def shortlist_candidates(lib: List[Dict[str, Any]], intents: List[str], ctx: Dict[str, Any], per_modality: int = 10) -> Dict[str, List[Dict[str, Any]]]:
    target = set(intents)
    filtered = [c for c in lib if c.get("modality") in MODALITIES and cue_applicable(c, ctx)]

    def overlap_score(cue: Dict[str, Any]) -> float:
        ci = set(cue.get("intents", []))
        inter = len(ci & target)
        if inter == 0:
            return 0.0
        return inter / max(1, (len(ci) * max(1, len(target))) ** 0.5)

    def score(cue: Dict[str, Any]) -> float:
        return (
            0.60 * overlap_score(cue) +
            0.20 * float(cue.get("universality", 0.5)) +
            0.20 * float(cue.get("salience", 0.5))
        )

    out: Dict[str, List[Dict[str, Any]]] = {m: [] for m in MODALITIES}
    for m in MODALITIES:
        cands = [c for c in filtered if c["modality"] == m]
        cands = sorted(cands, key=score, reverse=True)[:per_modality]
        out[m] = [{"id": c["id"], "intents": c.get("intents", []), "salience": c.get("salience", 0.5), "universality": c.get("universality", 0.5)} for c in cands]
    return out

def validate_intents(intents: Any) -> List[str]:
    if not isinstance(intents, list):
        return []
    allowed = set(ALLOWED_INTENTS)
    cleaned = []
    for x in intents:
        if isinstance(x, str) and x in allowed and x not in cleaned:
            cleaned.append(x)
    return cleaned[:6]  # keep it small & stable

def validate_plan(plan: Dict[str, Any], token_count: int, allowed_ids: set) -> Tuple[bool, str]:
    if not isinstance(plan, dict) or "insertions" not in plan:
        return False, "missing insertions"
    ins = plan["insertions"]
    if not isinstance(ins, list):
        return False, "insertions not list"
    # budget: max 2 per modality, 8 total
    if len(ins) > 8:
        return False, "too many insertions"
    for it in ins:
        if not isinstance(it, dict):
            return False, "insertion not dict"
        if it.get("position") not in ("before", "after"):
            return False, "bad position"
        if "token_index" not in it or "cue_id" not in it:
            return False, "missing fields"
        idx = int(it["token_index"])
        if idx < 0 or idx >= token_count:
            return False, f"token_index out of range: {idx}"
        if it["cue_id"] not in allowed_ids:
            return False, f"unknown cue_id: {it['cue_id']}"
    return True, "ok"

def apply_insertions(tokens: List[str], insertions: List[Dict[str, Any]]) -> str:
    by_idx: Dict[int, Dict[str, List[str]]] = {}
    for ins in insertions:
        idx = int(ins["token_index"])
        pos = ins["position"]
        cid = ins["cue_id"]
        by_idx.setdefault(idx, {"before": [], "after": []})
        by_idx[idx][pos].append(f"[{cid}]")

    out_tokens: List[str] = []
    for i, tok in enumerate(tokens):
        if i in by_idx:
            out_tokens.extend(by_idx[i]["before"])
        out_tokens.append(tok)
        if i in by_idx:
            out_tokens.extend(by_idx[i]["after"])
    return detokenize(out_tokens)

def step1_infer_intents(llm: Llama, text: str, ctx: Dict[str, Any]) -> List[str]:
    system = (
        "You label communicative intents for a single utterance. "
        "Return ONLY valid JSON: {\"intents\": [..]}. "
        "Choose from this allowed set only:\n"
        + json.dumps(ALLOWED_INTENTS)
        + "\nRules: output 2-6 intents max; prefer the most salient ones; no extra keys."
    )
    user = json.dumps({"text": text, "context": ctx}, ensure_ascii=False)
    resp = llm.create_chat_completion(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.0, top_p=1.0
    )
    raw = resp["choices"][0]["message"]["content"]
    parsed = extract_json(raw) or {}
    return validate_intents(parsed.get("intents"))

def step2_plan_tags(llm: Llama, text: str, tokens: List[str], intents: List[str], candidates: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    system = (
        "You plan where to insert multimodal cue tags into tokenized text. "
        "Return ONLY JSON with:\n"
        "{\n"
        "  \"insertions\": [ {\"cue_id\":\"...\",\"token_index\":0,\"position\":\"before\"} ... ]\n"
        "}\n"
        "Constraints:\n"
        "- cue_id MUST be from the provided candidates.\n"
        "- token_index MUST be within range.\n"
        "- Max 2 insertions per modality; 8 total.\n"
        "Placement heuristics:\n"
        "- uncertainty/repair cues near hesitations, ellipses, commas.\n"
        "- elicit_response near question mark / question clause.\n"
        "- emphasis near content word focus or clause start.\n"
        "- mitigation/affiliation near softeners or clause boundaries.\n"
        "Do NOT invent new text."
    )
    payload = {
        "text": text,
        "intents": intents,
        "tokens": [{"i": i, "t": t} for i, t in enumerate(tokens)],
        "candidates": candidates
    }
    resp = llm.create_chat_completion(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
        temperature=0.0, top_p=1.0
    )
    raw = resp["choices"][0]["message"]["content"]
    plan = extract_json(raw)
    if plan is None:
        raise ValueError(f"Step2 not JSON:\n{raw}")

    allowed_ids = {c["id"] for m in MODALITIES for c in candidates[m]}
    ok, reason = validate_plan(plan, token_count=len(tokens), allowed_ids=allowed_ids)
    if not ok:
        raise ValueError(f"Invalid plan ({reason}). Raw:\n{raw}")
    return plan

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to local instruct .gguf")
    ap.add_argument("--library", required=True, help="Path to cues JSON")
    ap.add_argument("--text", default="", help="Sentence to annotate (if empty, interactive)")
    ap.add_argument("--language", default="en")
    ap.add_argument("--culture", default="ANY")
    ap.add_argument("--formality", default="neutral")
    ap.add_argument("--roles", default="symmetric")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--print_debug", action="store_true")
    args = ap.parse_args()

    lib = load_library(Path(args.library))
    llm = Llama(model_path=args.model, n_ctx=4096, n_threads=8, seed=args.seed, verbose=False)

    ctx = {"language": args.language, "culture": args.culture, "formality": args.formality, "roles": args.roles}

    text = args.text.strip()
    if not text:
        text = input("Enter a sentence: ").strip()

    tokens = tokenize(text)

    intents = step1_infer_intents(llm, text, ctx)
    candidates = shortlist_candidates(lib, intents, ctx, per_modality=10)
    plan = step2_plan_tags(llm, text, tokens, intents, candidates)
    tagged = apply_insertions(tokens, plan["insertions"])

    print(tagged)

    if args.print_debug:
        debug = {"context": ctx, "intents": intents, "plan": plan, "candidates": candidates, "tokens": [{"i": i, "t": t} for i, t in enumerate(tokens)]}
        print("\n--- DEBUG JSON ---")
        print(json.dumps(debug, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
