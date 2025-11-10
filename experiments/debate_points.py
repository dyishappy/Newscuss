#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from typing import List, Dict, Any, Callable

# -------------------------
# 1. gpt (OpenAI Responses API)
# -------------------------
def generate_gpt(summary: str, keywords: List[str], topic: str, opts: Dict[str, Any]) -> str:
    from openai import OpenAI
    client = OpenAI()
    model = opts.get("model", "gpt-5-nano-2025-08-07")

    system_prompt = (
        "너는 토론 어시스턴트다. 주어진 요약문과 키워드를 근거로 "
        "주제에 대한 '찬성하는 주장'과 '반대하는 주장'을 한국어로 명확히 작성하라. "
        "각 주장은 3~5문장, 논리 전개(핵심 근거→추론→시사점)를 갖추고, 과장/허위 없이 근거 중심으로 쓴다. "
        "출력 형식(딱 이 구조만, 불필요한 문구 금지):\n"
        "[찬성]\n"
        "<찬성하는 주장 본문>\n\n"
        "[반대]\n"
        "<반대하는 주장 본문>"
    )
    user_text = (
        f"- 토론 주제: {topic}\n"
        f"- 요약문: {summary}\n"
        f"- 키워드: {', '.join(keywords)}"
    )

    # temperature 제거 (nano 모델은 미지원)
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "developer", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "input_text", "text": user_text}]},
        ],
    )

    text = getattr(resp, "output_text", None)
    if not text:
        text = resp.output[0].content[0].text
    return text.strip()

# -------------------------
# 2. llama-3.1-8B (open source by meta)
# -------------------------

def generate_hf_llama(summary: List[str] | str, keywords: List[str], topic: str, opts: Dict[str, Any]) -> str:
    """
    meta-llama/Llama-3.1-8B (Hugging Face Transformers) 로컬 추론
    반환 형식(그대로 출력):
    [찬성]
    ...

    [반대]
    ...
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

    model_id = opts.get("model_id", "meta-llama/Llama-3.1-8B")
    max_new_tokens = int(opts.get("max_new_tokens", 320))
    do_sample = bool(opts.get("do_sample", False))  # 재현성 원하면 False
    temperature = float(opts.get("temperature", 0.7))  # do_sample=True일 때만 의미
    top_p = float(opts.get("top_p", 0.9))

    if isinstance(summary, list):
        summary = " ".join([str(s) for s in summary])
    kw_str = ", ".join([k.strip() for k in keywords])

    system_prompt = (
        "너는 토론 어시스턴트다. 주어진 요약문과 키워드를 근거로 "
        "주제에 대한 '찬성하는 주장'과 '반대하는 주장'을 한국어로 명확히 작성하라. "
        "각 주장은 3~5문장, 논리 전개(핵심 근거→추론→시사점)를 갖추고, 과장/허위 없이 근거 중심으로 쓴다. "
        "출력 형식(정확히 아래 구조만, 불필요한 문구 금지):\n"
        "[찬성]\n"
        "<찬성하는 주장 본문>\n\n"
        "[반대]\n"
        "<반대하는 주장 본문>"
    )
    user_prompt = (
        f"- 토론 주제: {topic}\n"
        f"- 요약문: {summary}\n"
        f"- 키워드: {kw_str}"
    )

    # 토크나이저/모델 로드
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Llama 3.1는 chat 템플릿을 제공 → 템플릿으로 프롬프트 생성
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # 경고 억제 패딩 설정(일부 모델에서 필요)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.eos_token_id

    gen_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    out = gen_pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        return_full_text=False,  # 프롬프트를 제외한 생성만
    )

    # return_full_text=False 이므로 바로 본문만 옴
    text = out[0]["generated_text"].strip()
    return text


# -------------------------
# 미구현(확장용) - 나중에 채우세요
# -------------------------
def generate_ollama(summary: str, keywords: List[str], topic: str, opts: Dict[str, Any]) -> str:
    raise NotImplementedError("ollama 모델 생성 함수는 아직 미구현입니다.")

def generate_hf_tgi(summary: str, keywords: List[str], topic: str, opts: Dict[str, Any]) -> str:
    raise NotImplementedError("hf_tgi 모델 생성 함수는 아직 미구현입니다.")

# 디스패처 테이블 (각 함수는 str 반환)
DISPATCH: Dict[str, Callable[[str, List[str], str, Dict[str, Any]], str]] = {
    "gpt": generate_gpt,
    "hf_llama": generate_hf_llama,
    # "ollama": generate_ollama,
    # "hf_tgi": generate_hf_tgi,
}

def main() -> int:
    # 사용법: python debate_points.py MODEL TOPIC SUMMARY "kw1,kw2,kw3"
    if len(sys.argv) < 5:
        print("사용법: debate_points.py MODEL TOPIC SUMMARY \"kw1,kw2,kw3\"", file=sys.stderr)
        return 1

    model_key = sys.argv[1].strip().lower()
    topic     = sys.argv[2].strip()
    summary   = sys.argv[3].strip()
    keywords  = [k.strip() for k in sys.argv[4].split(",") if k.strip()]

    if model_key not in DISPATCH:
        print(f"[에러] 지원하지 않는 모델: {model_key}", file=sys.stderr)
        return 2

    # 엔진별 옵션(필요 시 여기서 분기해서 넣으세요)
    # gpt api 사용
    opts: Dict[str, Any] = {}
    if model_key == "gpt":
        opts["model"] = "gpt-5-nano-2025-08-07"
        opts["temperature"] = 0.5
    # llama 사용
    if model_key == "hf_llama":
        opts["model_id"] = "meta-llama/Llama-3.1-8B"
        opts["max_new_tokens"] = 320
        opts["do_sample"] = False      # 재현성 우선. 창의성 원하면 True로 하고 temperature/top_p 조절
        # opts["temperature"] = 0.8
        # opts["top_p"] = 0.9


    try:
        text = DISPATCH[model_key](summary, keywords, topic, opts)
    except NotImplementedError as nie:
        print(f"[미구현] {nie}", file=sys.stderr)
        return 3
    except Exception as e:
        print(f"[실패] 모델 호출 에러: {e}", file=sys.stderr)
        return 4

    # 표준출력으로 그대로 내보냄 (사용자가 직접 파싱)
    print(text)
    return 0

if __name__ == "__main__":
    sys.exit(main())
