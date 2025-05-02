from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import os
import time
import re
from dotenv import load_dotenv

# 모듈 추가
from newspaper import Article
from openai import OpenAI, RateLimitError
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.chat_completion_system_message_param import ChatCompletionSystemMessageParam
from openai.types.chat.chat_completion_user_message_param import ChatCompletionUserMessageParam
from openai.types.chat.chat_completion_assistant_message_param import ChatCompletionAssistantMessageParam

# 환경 변수 로드
load_dotenv()

# Flask 앱 생성
app = Flask(__name__)
CORS(app)  # Cross-Origin Resource Sharing 허용

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# OpenAI API 클라이언트 설정
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 재시도 관련 설정
MAX_RETRIES = 5  # 최대 재시도 횟수
INITIAL_BACKOFF = 1  # 초기 대기 시간(초)


# 1. 주어진 URL로부터 뉴스 기사의 텍스트 추출
def extract_article_text(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        logger.error(f"기사 추출 중 오류 발생: {e}")
        return None


# 2. 추출된 기사를 GPT API를 이용해 간결하게 요약
def summarize_text_with_gpt(article_text, model="gpt-4o"):
    try:
        instructions = "다음 뉴스 기사를 간결하게 요약해 주세요."
        messages: list[ChatCompletionMessageParam] = [
            ChatCompletionSystemMessageParam(
                role="system",
                content=instructions
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content=article_text
            )
        ]

        response = client.chat.completions.create(
            model=model,
            messages=messages
        )
        summary = response.choices[0].message.content.strip()
        return summary
    except Exception as e:
        logger.error(f"요약 생성 중 오류 발생: {e}")
        return None


# 3. 요약문에서 핵심 키워드 추출
def extract_keywords_from_summary(summary_text, num_keywords=3, model="gpt-4o"):
    try:
        instructions = f"다음 요약문에서 핵심 키워드를 {num_keywords}개만 추출해 주세요. 단순하게 빈도가 높은 단어가 아니라, 전체 내용에서 중요한 키워드를 추출해주고 키워드만 쉼표로 구분하여 출력해 주세요."
        messages: list[ChatCompletionMessageParam] = [
            ChatCompletionSystemMessageParam(
                role="system",
                content=instructions
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content=summary_text
            )
        ]

        response = client.chat.completions.create(
            model=model,
            messages=messages
        )
        keywords = response.choices[0].message.content.strip()
        return keywords.split(',')
    except Exception as e:
        logger.error(f"핵심 키워드 추출 중 오류 발생: {e}")
        return []


# 4. 요약문과 핵심 키워드를 바탕으로 토론 주제를 생성
def generate_discussion_topic(summary_text, keywords, model="gpt-4o"):
    try:
        keywords_str = ", ".join(keywords)
        instructions = (
            "다음 뉴스 기사 요약문과 핵심 키워드를 바탕으로 "
            "찬성과 반대로 나뉠 수 있는 토론 주제를 하나만 생성해 주세요.\n"
            f"요약문: {summary_text}\n"
            f"핵심 키워드: {keywords_str}\n"
            "토론 주제:"
        )

        messages: list[ChatCompletionMessageParam] = [
            ChatCompletionSystemMessageParam(
                role="system",
                content=instructions
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content=" "
            )
        ]

        response = client.chat.completions.create(
            model=model,
            messages=messages
        )
        topic = response.choices[0].message.content.strip()

        # 주제에 대한 설명 생성
        description_prompt = f"다음 토론 주제에 대한 간략한 설명을 100자 이내로 작성해 주세요.\n토론 주제: {topic}"
        description_messages: list[ChatCompletionMessageParam] = [
            ChatCompletionSystemMessageParam(
                role="system",
                content="당신은 토론 주제에 대한 간략한 설명을 제공하는 도우미입니다."
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content=description_prompt
            )
        ]

        description_response = client.chat.completions.create(
            model=model,
            messages=description_messages
        )
        description = description_response.choices[0].message.content.strip()

        return {
            "topic": topic,
            "description": description
        }
    except Exception as e:
        logger.error(f"토론 주제 생성 중 오류 발생: {e}")
        return {"topic": "생성 실패", "description": "토론 주제 생성에 실패했습니다."}


# 5. 토론 시작 및 AI의 첫 번째 메시지 생성
def generate_first_ai_message(topic, user_position, ai_position, difficulty, model="gpt-4o"):
    try:
        system_message = (
            f"당신은 토론에 참여하는 인공지능 토론자입니다. "
            f"토론 주제는 '{topic}'입니다. "
            f"사용자의 입장은 '{user_position}'이고, 당신은 반대 입장인 '{ai_position}'을 취해야 합니다. "
            f"토론 수준은 '{difficulty}'로 진행합니다. "
            f"사용자와의 토론을 시작하는 첫 번째 메시지를 작성해주세요. "
            f"상대방의 입장을 존중하되, 논리적이고 설득력 있는 주장을 펼쳐야 합니다."
        )

        messages: list[ChatCompletionMessageParam] = [
            ChatCompletionSystemMessageParam(
                role="system",
                content=system_message
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content="토론을 시작합니다."
            )
        ]

        response = client.chat.completions.create(
            model=model,
            messages=messages
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"AI 첫 메시지 생성 중 오류 발생: {e}")
        return "죄송합니다, 토론을 시작하는데 문제가 발생했습니다. 다시 시도해 주세요."


# 6. 사용자 메시지에 대한 AI 응답 생성
def generate_ai_response(topic, user_position, ai_position, difficulty, messages, model="gpt-4o"):
    try:
        system_message = (
            f"당신은 토론에 참여하는 인공지능 토론자입니다. "
            f"토론 주제는 '{topic}'입니다. "
            f"사용자의 입장은 '{user_position}'이고, 당신은 반대 입장인 '{ai_position}'을 취해야 합니다. "
            f"토론 수준은 '{difficulty}'로 진행합니다. "
            f"사용자의 메시지에 논리적이고 설득력 있게 응답해야 합니다."
        )

        # 메시지 히스토리 구성
        gpt_messages: list[ChatCompletionMessageParam] = [
            ChatCompletionSystemMessageParam(
                role="system",
                content=system_message
            )
        ]

        for msg in messages:
            if msg["role"] == "user":
                gpt_messages.append(ChatCompletionUserMessageParam(
                    role="user",
                    content=msg["content"]
                ))
            else:
                gpt_messages.append(ChatCompletionAssistantMessageParam(
                    role="assistant",
                    content=msg["content"]
                ))

        response = client.chat.completions.create(
            model=model,
            messages=gpt_messages
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"AI 응답 생성 중 오류 발생: {e}")
        return "죄송합니다, 응답을 생성하는데 문제가 발생했습니다. 다시 시도해 주세요."


# OpenAI API 호출 함수 (재시도 로직 포함)
def call_openai_with_retry(func, *args, max_retries=MAX_RETRIES, initial_backoff=INITIAL_BACKOFF, **kwargs):
    """
    OpenAI API 호출 함수를 래핑하여 속도 제한 오류 시 재시도 로직을 구현합니다.

    Args:
        func: 호출할 함수
        *args: 함수 인자
        max_retries: 최대 재시도 횟수
        initial_backoff: 첫 재시도 전 대기 시간(초)
        **kwargs: 함수 키워드 인자

    Returns:
        함수 실행 결과
    """
    backoff = initial_backoff
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)  # 성공 시 결과 반환
        except RateLimitError as e:
            if attempt == max_retries - 1:
                # 마지막 시도였으면 예외 다시 발생
                raise

            logger.warning(f"속도 제한 오류 발생 (시도 {attempt + 1}/{max_retries}): {e}")

            # 오류 메시지에서 권장 대기 시간 추출
            wait_time = backoff
            retry_match = re.search(r"Please try again in (\d+)ms", str(e))
            if retry_match:
                suggested_wait = int(retry_match.group(1)) / 1000.0  # ms를 초로 변환
                wait_time = max(backoff, suggested_wait)

            logger.info(f"{wait_time:.2f}초 후에 재시도합니다...")
            time.sleep(wait_time)

            # 지수 백오프: 다음 대기 시간은 2배
            backoff *= 2

    # 모든 재시도가 실패한 경우 - 여기에 명시적 return 또는 예외 발생 필요
    # 이 코드는 실행되지 않아야 하지만, 코드 분석 도구를 위해 추가
    raise Exception(f"최대 재시도 횟수({max_retries})를 초과했습니다.")


# 7. 토론 요약 생성 (메시지 압축 및 재시도 로직 최적화)
def generate_discussion_summary(topic, user_position, ai_position, messages, model="gpt-4o"):
    try:
        # 메시지 압축: 각 메시지의 내용을 최대 200자로 제한
        def truncate_message(text, max_length=200):
            if len(text) <= max_length:
                return text
            return text[:max_length] + "..."

        # 메시지가 15개 이상이면 샘플링
        compressed_messages = []
        if len(messages) > 15:
            # 처음 3개, 마지막 7개, 나머지 중 5개 선택
            important_messages = (
                    messages[:3] +  # 처음 3개
                    messages[3:-7:len(messages[3:-7]) // 5 + 1] +  # 중간에서 샘플링해서 5개
                    messages[-7:]  # 마지막 7개
            )
            compressed_messages = important_messages
        else:
            compressed_messages = messages

        # 내용 압축
        compressed_text = []
        for msg in compressed_messages:
            role = msg["role"]
            content = truncate_message(msg["content"])
            compressed_text.append(f"{role}: {content}")

        conversation_text = "\n".join(compressed_text)

        # 압축 정보 추가
        if len(messages) > len(compressed_messages):
            compression_note = f"\n\n[참고: 원본 대화는 {len(messages)}개의 메시지로 구성되어 있으며, 위 내용은 주요 {len(compressed_messages)}개 메시지의 요약입니다.]"
            conversation_text += compression_note

        system_message = (
            f"다음은 '{topic}'에 관한 토론 내용입니다. "
            f"사용자는 '{user_position}' 입장이었고, AI는 '{ai_position}' 입장이었습니다. "
            f"전체 토론을 간결하게 요약하고, 각 측의 주요 주장과 논점을 정리해 주세요."
        )

        chat_messages: list[ChatCompletionMessageParam] = [
            ChatCompletionSystemMessageParam(
                role="system",
                content=system_message
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content=conversation_text
            )
        ]

        # 재시도 로직을 포함하여 API 호출
        def make_summary_request():
            response = client.chat.completions.create(
                model=model,
                messages=chat_messages,
                temperature=0.7,  # 온도 조절로 다양성 증가
                max_tokens=800  # 최대 토큰 제한
            )
            return response.choices[0].message.content.strip()

        return call_openai_with_retry(make_summary_request)

    except Exception as e:
        logger.error(f"토론 요약 생성 중 오류 발생: {e}")
        # 요약 실패 시 기본 메시지 제공
        return "죄송합니다, 토론 요약을 생성하는데 문제가 발생했습니다. 토론 내용이 너무 길거나 복잡할 수 있습니다."

# API 엔드포인트 정의

# 1. URL 처리 및 키워드/요약 추출 API
@app.route('/api/extract', methods=['POST'])
def extract_api():
    data = request.json
    url = data.get('url')

    if not url:
        return jsonify({"error": "URL이 필요합니다"}), 400

    logger.info(f"Processing URL: {url}")

    # 1. 기사 텍스트 추출
    article_text = extract_article_text(url)
    if not article_text:
        return jsonify({"error": "기사를 추출할 수 없습니다"}), 400

    # 2. 기사 요약
    summary = summarize_text_with_gpt(article_text)
    if not summary:
        return jsonify({"error": "요약을 생성할 수 없습니다"}), 500

    # 3. 키워드 추출
    keywords = extract_keywords_from_summary(summary)

    return jsonify({
        "summary": summary,
        "keywords": keywords
    })


# 2. 토론 주제 생성 API
@app.route('/api/topic', methods=['POST'])
def topic_api():
    data = request.json
    summary = data.get('summary')
    keywords = data.get('keywords')

    if not summary or not keywords:
        return jsonify({"error": "요약문과 키워드가 필요합니다"}), 400

    logger.info("Generating discussion topic")

    topic_data = generate_discussion_topic(summary, keywords)

    return jsonify(topic_data)


# 3. 토론 시작 API
@app.route('/api/discussion/start', methods=['POST'])
def start_discussion_api():
    data = request.json
    topic = data.get('topic')
    user_position = data.get('userPosition')
    ai_position = data.get('aiPosition')
    difficulty = data.get('difficulty')

    if not all([topic, user_position, ai_position, difficulty]):
        return jsonify({"error": "모든 파라미터가 필요합니다"}), 400

    logger.info(f"Starting discussion: {topic}, user: {user_position}, AI: {ai_position}, level: {difficulty}")

    ai_message = generate_first_ai_message(topic, user_position, ai_position, difficulty)

    return jsonify({"message": ai_message})


# 4. 토론 메시지 처리 API
@app.route('/api/discussion/message', methods=['POST'])
def message_api():
    data = request.json
    topic = data.get('topic')
    user_position = data.get('userPosition')
    ai_position = data.get('aiPosition')
    difficulty = data.get('difficulty')
    messages = data.get('messages')

    if not all([topic, user_position, ai_position, difficulty, messages]):
        return jsonify({"error": "모든 파라미터가 필요합니다"}), 400

    logger.info(f"Processing message for topic: {topic}")

    ai_response = generate_ai_response(topic, user_position, ai_position, difficulty, messages)

    return jsonify({"message": ai_response})


# 5. 토론 요약 API
@app.route('/api/discussion/summary', methods=['POST'])
def summary_api():
    data = request.json
    topic = data.get('topic')
    user_position = data.get('userPosition')
    ai_position = data.get('aiPosition')
    messages = data.get('messages')

    if not all([topic, user_position, ai_position, messages]):
        return jsonify({"error": "모든 파라미터가 필요합니다"}), 400

    logger.info(f"Generating summary for topic: {topic}, messages count: {len(messages)}")

    try:
        # 메시지 수가 너무 많을 경우 일부 생략 처리
        if len(messages) > 30:
            logger.warning(f"메시지가 너무 많습니다: {len(messages)}개. 일부만 처리합니다.")
            # 처음 10개, 마지막 10개, 중간에서 10개
            sampled_messages = (
                messages[:10] +
                messages[10:-10:max(1, (len(messages) - 20) // 10)] +
                messages[-10:]
            )
            logger.info(f"메시지 샘플링 완료: {len(sampled_messages)}개")
            messages = sampled_messages

        summary = generate_discussion_summary(topic, user_position, ai_position, messages)
        return jsonify({"summary": summary})
    except RateLimitError as e:
        logger.error(f"Rate limit exceeded during summary generation: {e}")
        error_message = "토론 분석 중 API 요청 한도에 도달했습니다. 잠시 후 다시 시도해주세요."
        return jsonify({"summary": error_message, "error": str(e)}), 200  # 클라이언트에서 처리 가능하도록 200 반환
    except Exception as e:
        logger.error(f"Summary generation error: {e}")
        error_message = "토론 요약 중 오류가 발생했습니다. 토론 내용이 너무 길거나 복잡한 경우 시간이 더 걸릴 수 있습니다."
        return jsonify({"summary": error_message, "error": str(e)}), 200  # 클라이언트에서 처리 가능하도록 200 반환


# 메인 실행 부분
if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)