#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=“/nas/datahub/HFHOME/hub”
export HF_HOME=~/.cache/huggingface
export OPENAI_API_KEY="여기 키 입력"

############################################
MODEL="gpt"                                # gpt | clovax | hf_llama | deepseek | oss (현재는 gpt, llama만 구현)
TOPIC="고교 무상급식 확대"                 # 토론 주제
SUMMARY=$'독일의 과학 유튜브 채널 ‘쿠르츠게작트’는 ‘한국은 끝났다’라는 영상에서 한국의 극심한 저출산 문제와 인구 감 소, 경제와 사회의 붕괴 가능성을 경고했다.\n
          영상은 현재 출산율이 매우 낮아 2060년에는 인구의 절반 이상이 사 라지고 노인 인구가 대부분이 될 것이라고 분석하며, 국민연금 고갈과 경기침체도 전망했다.\n
          분석은 한국의 근로 문화와 경쟁 사회가 출산률 저하를 초래한 원인으로 지목하며, 출산율 회복을 위한 급진적 변화의 필요성을 제언 하고 있다.'  # 줄바꿈 필요하면 $''로
KEYWORDS="예산,영양,형평성"                # 쉼표로 구분: "a,b,c"
############################################

# gpt일 때 OpenAI 키 확인
if [ "$MODEL" = "gpt" ]; then
  : "${OPENAI_API_KEY:?환경변수 OPENAI_API_KEY가 필요합니다. export OPENAI_API_KEY=...}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "${SCRIPT_DIR}/debate_points.py" "$MODEL" "$TOPIC" "$SUMMARY" "$KEYWORDS"
