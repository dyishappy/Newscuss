from rake_nltk import Rake
from transformers import set_seed

# 핵심 모듈2
# 요약문에서 키워드 추출 구현

# 3. 요약문에서 키워드 추출 함수 (RAKE 알고리즘 활용)
def extract_keywords(text, num_keywords=5):
    try:
        rake = Rake()  # 기본 영어 불용어 사용
        rake.extract_keywords_from_text(text)
        # 추출된 키워드들 중 상위 num_keywords개 선택
        keywords = rake.get_ranked_phrases()[:num_keywords]
        return keywords
    except Exception as e:
        print(f"키워드 추출 중 오류 발생: {e}")
        return []