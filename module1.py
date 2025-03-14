from newspaper import Article
from transformers import pipeline

# 핵심 모듈1
# 뉴스 기사 크롤링 및 요약 기능 구현

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
