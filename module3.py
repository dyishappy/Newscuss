from transformers import pipeline, set_seed

# 핵심 모듈3
# 키워드 기반 논쟁 거리 자동 추출

# 4. 토론 주제 생성 함수 (텍스트 생성 모델 활용)
def generate_discussion_topic(sumsmary, keywords, max_length=50):
    try:
        generator = pipeline("text-generation", model="gpt2")
        set_seed(42)  # 재현성을 위한 시드 설정
        prompt = (f"기사 요약: {summary}\n"
                  f"키워드: {', '.join(keywords)}\n"
                  f"위 정보를 바탕으로 흥미로운 토론 주제를 한 문장으로 제안해 주세요: ")
        generated = generator(prompt, max_length=max_length, num_return_sequences=1)
        # 생성된 텍스트에서 prompt 부분을 제거할 수 있음
        topic = generated[0]['generated_text'][len(prompt):].strip()
        return topic
    except Exception as e:
        print(f"토론 주제 생성 중 오류 발생: {e}")
        return None