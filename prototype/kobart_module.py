import os
from openai import OpenAI
import openai
from newspaper import Article
import torch
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration


# 1. 주어진 URL로부터 뉴스 기사의 텍스트 추출
def extract_article_text(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        print(f"기사 추출 중 오류 발생: {e}")
        return None

# 2. 추출된 기사를 GPT API를 이용해 간결하게 요약
# def summarize_text_with_gpt(client, article_text, model="gpt-4o"):
#     try:
#         instructions = "다음 뉴스 기사를 간결하게 요약해 주세요."
#         response = client.responses.create(
#             model=model,
#             instructions=instructions,
#             input=article_text,
#         )
#         summary = response.output_text.strip()
#         return summary
#     except Exception as e:
#         print(f"요약 생성 중 오류 발생: {e}")
#         return None

# 2. 추출된 기사를 KoBART로 요약
def summarize_text_with_kobart(
    article_text: str,
    cache_dir: str = None,
    max_input_length: int = 1024,
    max_summary_length: int = 256,
    num_beams: int = 4,
) -> str:
    """
    article_text: 요약 대상 텍스트
    model_name: Hugging Face 모델 식별자
    cache_dir: 모델/토크나이저 캐시 경로 (None 이면 기본 경로 사용)
    max_input_length: 입력 토큰 최대 길이
    max_summary_length: 생성 요약 토큰 최대 길이
    num_beams: 빔 서치 빔 개수
    """
    try:
        # 1) 디바이스 결정
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 2) 토크나이저·모델 로드
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            pretrained_model_name_or_path="gogamza/kobart-base-v1",
            cache_dir=cache_dir,
        )
        model = BartForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path="gogamza/kobart-base-v1",
            cache_dir=cache_dir,
        )
        model.to(device)
        model.eval()  # inference 모드로 전환

        # 3) 토큰화 (token_type_ids 미생성) - 프롬프트 추가
        prompt = "다음 뉴스 기사를 간결하게 요약해 주세요."
        input_text = prompt + article_text

        raw = tokenizer(
            input_text,
            return_tensors="pt",
            max_length=max_input_length,
            truncation=True,
            padding="longest",
            return_token_type_ids=False,
        )
        inputs = {k: v.to(device) for k, v in raw.items()}

        # 4) 생성
        with torch.no_grad():
            summary_ids = model.generate(
                **inputs,
                max_length=max_summary_length,
                num_beams=num_beams,
                early_stopping=True,
            )

        # 5) 디코딩
        summary = tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        return summary.strip()

    except Exception as e:
        print(f"요약 생성 중 오류 발생: {e}")
        return None

# 3. 요약문을 입력받아, 사용자가 지정한 개수만큼 핵심 키워드를 추출합니다.(default=2) 지시문에서는 키워드만 쉼표로 구분하여 출력하도록 요청
def extract_keywords_from_summary(client, summary_text, num_keywords=2, model="gpt-4o"):
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
# Discussion! prompt 잘 줘봐도 요약문으로 키워드 추출하는게 뭔가 성능이 않좋아보임. 차라리 요약문 만들 때 그냥 키워드도 원문에서 뽑는게 나을것 같음.
    
# 4. 요약문과 핵심 키워드를 바탕으로 토론 주제를 생성 
def generate_discussion_topic(client, summary_text, keywords, model="gpt-4o"):
    try:
        instructions = (
            "다음 뉴스 기사 요약문과 핵심 키워드를 바탕으로 "
            "찬성과 반대로 나뉠 수 있는 토론 주제를 하나만 생성해 주세요.\n"
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

# 5. GPT와 찬/반, 수준을 선택하여 토론하고 결과를 반환해줌
def interactive_debate(client, topic, user_stance, debate_level, model="gpt-4o"):
    # 사용자 입장에 따라 GPT의 반대 입장을 결정합니다.
    if user_stance == "찬성":
        opponent_stance = "반대"
    elif user_stance == "반대":
        opponent_stance = "찬성"
    else:
        print("입력한 토론 입장이 올바르지 않습니다. '찬성' 또는 '반대'로 입력해 주세요.")
        return None

    # 초기 시스템 메시지를 포함하여 대화 내역을 생성합니다.
    messages = [
        {
            "role": "system",
            "content": (
                f"당신은 토론에 참여하는 인공지능 토론자입니다. "
                f"토론 주제는 '{topic}'입니다. "
                f"사용자(나)의 입장은 '{user_stance}'이고, 당신은 자동으로 반대 입장인 '{opponent_stance}'를 취해야 합니다. "
                f"토론 수준은 '{debate_level}'로 진행합니다. "
                "대화는 계속 이어지며, 이전 대화 내용을 모두 반영하여 답변해 주세요."
            )
        }
    ]

    print("\n토론을 시작합니다. 종료하려면 'exit'를 입력하세요.")
    while True:
        user_input = input("\n당신: ")
        if user_input.lower() in ['exit', 'quit']:
            print("토론을 종료합니다.")
            break
        # 사용자의 메시지를 추가합니다.
        messages.append({"role": "user", "content": user_input})
        # Chat Completions API를 이용하여 응답을 생성합니다.
        response = client.chat.completions.create(
            model=model,
            messages=messages
        )
        assistant_reply = response.choices[0].message.content.strip()
        print("\nGPT (반대 입장):", assistant_reply)
        # 응답을 대화 내역에 추가합니다.
        messages.append({"role": "assistant", "content": assistant_reply})
    return messages

# 6. 토론 내용을 요약하고 토론 결과 출력
def summarize_and_print_debate_results(client, messages, model="gpt-4o"):
    """
    전체 토론 대화 내역을 요약하고 토론 결과를 출력합니다.
    """
    # 대화 내역을 텍스트로 변환 (시스템 메시지를 제외하거나 포함할 수 있음)
    conversation_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
    instructions = (
        "다음 대화 내용을 바탕으로 전체 토론을 간결하게 요약하고 토론 결과를 출력해 주세요. "
        "각 참여자의 주장, 토론의 흐름, 그리고 최종 결론(있는 경우)을 포함해 주세요.\n\n"
        f"{conversation_text}"
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": instructions}
        ]
    )
    summary = response.choices[0].message.content.strip()
    print("\n=== 토론 요약 및 결과 ===")
    print(summary)

if __name__ == '__main__':

    client = OpenAI(api_key="키")
    cache_dir = "모델을 저장해두고 이후에 사용할 때 이 캐시에서 바로 로드함"

    url = input("기사 URL을 입력해 주세요: ")
    article_text = extract_article_text(url)
    if article_text:
        summary = summarize_text_with_kobart(article_text, cache_dir)
        print("\n=== 기사 요약 ===")
        print(summary)
        
        keywords = extract_keywords_from_summary(client, summary)
        print("\n=== 핵심 키워드 ===")
        print(keywords)
        
        topic = generate_discussion_topic(client, summary, keywords)
        print("\n=== 토론 주제 ===")
        print(topic)
        
        user_stance = input("\n토론에서 당신의 입장을 선택해 주세요 (찬성/반대): ").strip()
        debate_level = input("토론 수준을 선택해 주세요 (초급/중급/상급): ").strip()
        
        # interactive_debate 함수 내에서 대화가 진행되고, 최종 대화 내역을 반환받습니다.
        messages = interactive_debate(client, topic, user_stance, debate_level)
        
        # 토론이 끝난 후 전체 대화 내용을 요약하고 결과를 출력합니다.
        if messages:
            summarize_and_print_debate_results(client, messages)
    else:
        print("기사 텍스트를 추출하지 못했습니다.")