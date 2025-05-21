import json
import os
from openai import OpenAI
import openai
from newspaper import Article
from gpt_module import extract_article_text, summarize_text_with_gpt, summarize_text_with_clova, summarize_text_with_kogpt, summarize_text_with_blc, summarize_text_with_kobart, extract_keywords_from_summary, generate_discussion_topic, interactive_debate, summarize_and_print_debate_results

parser = argparse.ArgumentParser()
parser.add_argument('--modle',type=str,default=gpt)
parser.add_argument('--cache_dir',type=str,default='/hub')
parser.add_argument('--client',type=str,default='gpt api key')
args = parser.parse_args()

client = OpenAI(api_key=args.client) if args.client else None

models_map = {
    'gpt':    summarize_text_with_gpt,
    'clova':  summarize_text_with_clova,
    'kogpt':  summarize_text_with_kogpt,
    'blc':    summarize_text_with_blc,
    'kobart': summarize_text_with_kobart,
}

sumarize_function = models_map[args.model]

# JSON 파일에서 데이터 읽기
with open('source.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

url = data.get("url", "")
if not url:
    print("기사 URL이 JSON에 없습니다.")
    exit(1)

article_text = extract_article_text(url)
if not article_text:
    print("기사 텍스트를 추출하지 못했습니다.")
    exit(1)

if args.model == 'gpt':
    # GPT 모델은 (client, text)
    summary = models_map['gpt'](client, article_text)
else:
    # 나머지 모델은 (text, cache_dir)
    func = models_map[args.model]
    summary = func(article_text, args.cache_dir)

print("\n=== 기사 요약 ===")
print(summary)

keywords = extract_keywords_from_summary(client, summary)
print("\n=== 핵심 키워드 ===")
print(keywords)

topic = generate_discussion_topic(client, summary, keywords)
print("\n=== 토론 주제 ===")
print(topic)

user_stance = data.get("user_stance", "찬성")
debate_level = data.get("debate_level", "초급")
messages = interactive_debate(client, topic, user_stance, debate_level)

if messages:
    summarize_and_print_debate_results(client, messages)