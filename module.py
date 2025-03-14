import nltk
from newspaper import Article
from transformers import pipeline, set_seed
from rake_nltk import Rake

# nltk에서 필요한 데이터 다운로드 (최초 실행 시 필요)
nltk.download('punkt')

# 1. 뉴스 기사 내용 추출 함수
def extract_article_text(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        print(f"기사 추출 중 오류 발생: {e}")
        return None

# 2. 기사 요약 함수 (Hugging Face Summarization Pipeline 활용)
def summarize_text(text, max_length=150, min_length=40):
    try:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        print(f"요약 생성 중 오류 발생: {e}")
        return None

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

# 4. 토론 주제 생성 함수 (텍스트 생성 모델 활용)
def generate_discussion_topic(summary, keywords, max_length=50):
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
