import os
import json
from openai import OpenAI
import openai
from newspaper import Article
#import torch
#from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration

def main():
    # JSON 파일에서 데이터 읽기
    # try:
    #     with open('source.json', 'r', encoding='utf-8') as f:
    #         data = json.load(f)
    # except Exception as e:
    #     print("JSON 파일을 읽는 중 오류 발생:", e)
    #     return

    # url = data.get("url", "")
    # if not url:
    #     print("기사 URL이 JSON에 없습니다.")
    #     return

    # article = Article(url)
    #article = Article('https://www.mk.co.kr/news/economy/11321998')
    article = Article('https://www.mk.co.kr/news/society/11322448')
    article.download()
    article.parse()
    cleaned_text = article.text.replace('사진 확대', '').strip()
    print(cleaned_text)
    

if __name__ == '__main__':
    main()