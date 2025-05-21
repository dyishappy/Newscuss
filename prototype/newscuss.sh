#!/usr/bin/env bash

# 환경변수 설정
# OpenAI API 키를 환경변수로 설정 -> 따로 .py파일 수정할 필요 x
export OPENAI_API_KEY="여기에_본인의_API_KEY를_입력하세요"
# 캐시 디렉토리 지정
export CACHE_DIR="${CACHE_DIR:-./cache}"

# 3) JSON 파일 경로 지정 (기본값: source.json)
SOURCE_JSON="${1:-source.json}"

# 4) 실행
python3 main.py --source "$SOURCE_JSON" --cache_dir "$CACHE_DIR"
