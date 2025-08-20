import os, json
import boto3
from typing import Any, Dict, List, Optional
from botocore.config import Config
from app.core.config import settings

AWS_REGION = settings.AWS_REGION
BEDROCK_KB_ID = settings.BEDROCK_KB_ID
BEDROCK_KB_MODEL_ID = settings.BEDROCK_KB_MODEL_ID

# 연결/재시도 설정(과도한 재시도는 지연을 유발할 수 있으니 상황에 맞게 조절)
_boto_cfg = Config(
    retries={"max_attempts": 3, "mode": "standard"},
    read_timeout=30,
    connect_timeout=10,
    region_name=AWS_REGION,
)

# Boto3 clients
agent_rt = boto3.client("bedrock-agent-runtime", config=_boto_cfg)
bedrock_rt = boto3.client("bedrock-runtime", config=_boto_cfg)

# -----------------------------------------------------------------------------
# KB Retrieve
# -----------------------------------------------------------------------------
def kb_retrieve(
    query_text: str,
    kb_id: Optional[str] = None,
    k: int = 5,
    metadata_filter: Optional[Dict[str, Any]] = None,
    max_pages: int = 1,
) -> List[Dict[str, Any]]:
    """
    Bedrock Knowledge Base에서 유사 문서 검색.

    metadata_filter 예시(AND/OR 조합 가능):
    {
      "and": [
        {"equals": {"key": "label", "value": "pneumonia"}},
        {"equals": {"key": "view", "value": "PA"}}
      ]
    }
    또는
    {"equals": {"key": "label", "value": "pneumonia"}}

    반환: [{"id": "...", "report_text_snippet": "...", "sim": 0.87, "metadata": {...}}, ...]
    """
    kb_id = kb_id or BEDROCK_KB_ID
    if not kb_id:
        raise RuntimeError("BEDROCK_KB_ID is not set")

    req = {
        "knowledgeBaseId": kb_id,
        "retrievalConfiguration": {
            "vectorSearchConfiguration": {
                "numberOfResults": k
            }
        },
        "input": {"text": query_text}
    }
    if metadata_filter:
        req["retrievalConfiguration"]["vectorSearchConfiguration"]["filter"] = metadata_filter

    items: List[Dict[str, Any]] = []
    next_token: Optional[str] = None
    pages = 0

    while True:
        if next_token:
            req["nextToken"] = next_token
        try:
            resp = agent_rt.retrieve(**req)
        except Exception as e:
            # 네트워크/권한/파서 오류 방어
            raise RuntimeError(f"Bedrock KB retrieve failed: {e}") from e

        for r in resp.get("retrievalResults", []) or []:
            content = (r.get("content") or {}).get("text", "") or ""
            # 소스 식별: s3 uri 또는 KB doc id
            source = (
                (r.get("location") or {}).get("s3Location", {}) or {}
            ).get("uri") or (r.get("metadata") or {}).get("x-amz-bedrock-kb-doc-id")
            score = r.get("score")
            meta = r.get("metadata") or {}

            items.append({
                "id": source,
                "report_text_snippet": content[:400].replace("\n", " "),
                "sim": score,
                "metadata": meta
            })

        next_token = resp.get("nextToken")
        pages += 1

        if not next_token or pages >= max_pages or len(items) >= k:
            break

    # 상위 k만 반환
    return items[:k]

# -----------------------------------------------------------------------------
# Sonnet 3.7로 보고서 생성
# -----------------------------------------------------------------------------
def generate_report_with_sonnet(
    label_pred: str,
    quadrant_pred: Optional[str],
    qconf: Optional[float],
    query_text: str,
    neighbors: List[Dict[str, Any]],
    model_id: Optional[str] = None,
    max_tokens: int = 700,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """
    neighbors: kb_retrieve() 결과
    반환: {"sections": {"findings": [...], "impression": [...]},
           "evidence": {"sources": [...], "rationale": "..."},
           "meta": {...}}
    """
    model_id = model_id or BEDROCK_KB_MODEL_ID
    if not model_id:
        raise RuntimeError("BEDROCK_KB_MODEL_ID is not set")

    system_prompt = """You are a radiology assistant generating concise, clinically sound chest X-ray reports.
Rules:
- Findings: 2–4 short sentences. Neutral, professional tone.
- Impression: 1–2 sentences summarizing the key diagnosis or lack thereof.
- Use laterality and lobe only if reasonably supported; if location confidence (qconf) is low, soften (e.g., "left lung lower zone").
- Do not invent measurements. Prefer majority from retrieved context.
- Return strict JSON only, no extra text.
"""

    lines, src = [], []
    for r in neighbors:
        sid = r.get("id", "")
        src.append(sid)
        snippet = (r.get("report_text_snippet") or "").replace("\n", " ")
        sim = r.get("sim", 0)
        lines.append(f"- id={sid} | sim={sim}: {snippet}")
    snippets = "\n".join(lines)

    user_content = f"""Study:
- label_pred: {label_pred}
- quadrant_pred: {quadrant_pred}
- qconf: {qconf}
- query_text: {query_text}

Top-{len(neighbors)} retrieved snippets:
{snippets}

Return JSON with keys: sections(findings[], impression[]), evidence(sources[], rationale), meta(label_pred, quadrant_pred, qconf).
"""

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "system": system_prompt,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": user_content}]
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        # 필요 시 멈춤 시퀀스와 포맷 힌트 추가 가능
        # "stop_sequences": ["</json>"]
    }

    try:
        resp = bedrock_rt.invoke_model(
            modelId=model_id,
            accept="application/json",
            contentType="application/json",
            body=json.dumps(body),
        )
        payload = json.loads(resp["body"].read())
        text = "".join(
            blk.get("text", "")
            for blk in payload.get("content", []) or []
        ).strip()
    except Exception as e:
        raise RuntimeError(f"Bedrock invoke_model failed: {e}") from e

    # Sonnet 출력이 반드시 JSON이라는 보장은 없으므로 방어
    try:
        obj = json.loads(text)
    except Exception:
        obj = {
            "sections": {
                "findings": [text[:400]],
                "impression": []
            },
            "evidence": {
                "sources": src[:3],
                "rationale": "Generated with fallback parsing."
            },
            "meta": {
                "label_pred": label_pred,
                "quadrant_pred": quadrant_pred,
                "qconf": qconf
            }
        }

    # 필드 보정
    obj.setdefault("sections", {})
    obj["sections"].setdefault("findings", [])
    obj["sections"].setdefault("impression", [])
    obj.setdefault("evidence", {})
    obj["evidence"].setdefault("sources", src[:3])
    obj.setdefault("meta", {})
    obj["meta"].update({
        "label_pred": label_pred,
        "quadrant_pred": quadrant_pred,
        "qconf": qconf
    })
    return obj