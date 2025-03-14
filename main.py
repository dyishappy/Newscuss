import nltk
from module1 import extract_article_text, summarize_text
from module2 import extract_keywords
from module3 import generate_discussion_topic

# nltk에서 필요한 데이터 다운로드 (최초 실행 시 필요)
nltk.download('punkt')

# 전체 파이프라인 실행 함수
def process_article(url):
    print("기사 추출 중...")
    article_text = extract_article_text(url)
    if not article_text:
        return
    
    print("기사 요약 중...")
    summary = summarize_text(article_text)
    if not summary:
        return
    
    print("키워드 추출 중...")
    keywords = extract_keywords(summary)
    
    print("토론 주제 생성 중...")
    discussion_topic = generate_discussion_topic(summary, keywords)
    
    # 결과 출력
    print("\n--- 결과 ---")
    print("요약문:")
    print(summary)
    print("\n키워드:")
    print(keywords)
    print("\n토론 주제:")
    print(discussion_topic)

# 예제 실행 (원하는 뉴스 기사 URL로 변경)
if __name__ == "__main__":
    url = input("뉴스 기사 URL을 입력하세요: ")
    process_article(url)