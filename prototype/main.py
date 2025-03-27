import os
from openai import OpenAI
from newspaper import Article

# 전역 변수로 클라이언트 초기화
# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# client = OpenAI(api_key="여기 gpt api 키 넣어줘야함! 이 라인 아에 지우고 공유사항 폴더에 있는 거 복붙ㄱㄱ")

# 주어진 URL로부터 뉴스 기사의 텍스트 추출
def extract_article_text(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        print(f"기사 추출 중 오류 발생: {e}")
        return None

# 추출된 기사를 GPT API를 이용해 간결하게 요약
def summarize_text_with_gpt(article_text, model="gpt-4o"):
    try:
        instructions = "다음 뉴스 기사를 간결하게 요약해 주세요."
        response = client.responses.create(
            model=model,
            instructions=instructions,
            input=article_text,
        )
        summary = response.output_text.strip()
        return summary
    except Exception as e:
        print(f"요약 생성 중 오류 발생: {e}")
        return None

# 요약문을 입력받아, 사용자가 지정한 개수만큼 핵심 키워드를 추출합니다.(default=2) 지시문에서는 키워드만 쉼표로 구분하여 출력하도록 요청
def extract_keywords_from_summary(summary_text, num_keywords=2, model="gpt-4o"):
    try:
        instructions = f"다음 요약문에서 핵심 키워드를 {num_keywords}개만 추출해 주세요. 단순하게 빈도가 높은 단어가 아니라, 전체 내용에서 중요한 키워드를 추출해주고 키워드만 쉼표로 구분하여 출력해 주세요."
        response = client.responses.create(
            model=model,
            instructions=instructions,
            input=summary_text,
        )
        keywords = response.output_text.strip()
        return keywords
    except Exception as e:
        print(f"핵심 키워드 추출 중 오류 발생: {e}")
        return None
# Discussion! prompt 잘 줘봐도 요약문으로 키워드 추출하는게 뭔가 성능이 않좋아보임. 차라리 요약문 만들 때 그냥 키워드도 원문에서 뽑는게 나을것 같음
    
# 요약문과 핵심 키워드를 바탕으로 토론 주제를 생성 
def generate_discussion_topic(summary_text, keywords, model="gpt-4o"):
    try:
        instructions = (
            "다음 뉴스 기사 요약문과 핵심 키워드를 바탕으로 "
            "흥미로운 토론 주제를 하나 생성해 주세요.\n"
            f"요약문: {summary_text}\n"
            f"핵심 키워드: {keywords}\n"
            "토론 주제:"
        )
        # input 파라미터에 공백 하나(" ")를 전달하여 non-empty 값을 제공합니다.
        response = client.responses.create(
            model=model,
            instructions=instructions,
            input=" ",
        )
        topic = response.output_text.strip()
        return topic
    except Exception as e:
        print(f"토론 주제 생성 중 오류 발생: {e}")
        return None

if __name__ == '__main__':
    # 기사 URL 입력
    url = input("기사 URL을 입력해 주세요: ")
    article_text = extract_article_text(url)
    if article_text:
        summary = summarize_text_with_gpt(article_text)
        print("\n=== 기사 요약 ===")
        print(summary)
        
        keywords = extract_keywords_from_summary(summary)
        print("\n=== 핵심 키워드 ===")
        print(keywords)
        
        topic = generate_discussion_topic(summary, keywords)
        print("\n=== 토론 주제 ===")
        print(topic)
    else:
        print("기사 텍스트를 추출하지 못했습니다.")