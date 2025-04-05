import json
import os
from openai import OpenAI
import openai
from newspaper import Article
from gpt_module import extract_article_text, summarize_text_with_gpt, extract_keywords_from_summary, generate_discussion_topic, interactive_debate, summarize_and_print_debate_results

def main():

    client = OpenAI(api_key="이 부분 맞는 키로 바꿔줘야함.")

    # JSON 파일에서 데이터 읽기
    try:
        with open('source.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print("JSON 파일을 읽는 중 오류 발생:", e)
        return

    url = data.get("url", "")
    if not url:
        print("기사 URL이 JSON에 없습니다.")
        return

    article_text = extract_article_text(url)
    if article_text:
        summary = summarize_text_with_gpt(client, article_text)
        print("\n=== 기사 요약 ===")
        print(summary)
        
        keywords = extract_keywords_from_summary(client, summary)
        print("\n=== 핵심 키워드 ===")
        print(keywords)
        
        topic = generate_discussion_topic(client, summary, keywords)
        print("\n=== 토론 주제 ===")
        print(topic)
        
        # JSON 파일에서 사용자의 토론 입장과 수준을 가져옵니다.
        user_stance = data.get("user_stance", "찬성")
        debate_level = data.get("debate_level", "초급")
        
        messages = interactive_debate(client, topic, user_stance, debate_level)
        
        if messages:
            summarize_and_print_debate_results(client, messages)
    else:
        print("기사 텍스트를 추출하지 못했습니다.")

if __name__ == '__main__':
    main()
