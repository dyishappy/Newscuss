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
CORS(app)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# OpenAI API 클라이언트 설정
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 상수 정의
MAX_RETRIES = 5
INITIAL_BACKOFF = 1
DEFAULT_MODEL = "gpt-4o"


class OpenAIService:
    """OpenAI API 호출을 담당하는 서비스 클래스"""

    @staticmethod
    def call_with_retry(func, *args, max_retries=MAX_RETRIES, initial_backoff=INITIAL_BACKOFF, **kwargs):
        """OpenAI API 호출 재시도 로직"""
        backoff = initial_backoff
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except RateLimitError as e:
                if attempt == max_retries - 1:
                    raise

                logger.warning(f"속도 제한 오류 발생 (시도 {attempt + 1}/{max_retries}): {e}")

                wait_time = backoff
                retry_match = re.search(r"Please try again in (\d+)ms", str(e))
                if retry_match:
                    suggested_wait = int(retry_match.group(1)) / 1000.0
                    wait_time = max(backoff, suggested_wait)

                logger.info(f"{wait_time:.2f}초 후에 재시도합니다...")
                time.sleep(wait_time)
                backoff *= 2

        raise Exception(f"최대 재시도 횟수({max_retries})를 초과했습니다.")

    @classmethod
    def create_chat_completion(cls, messages, model=DEFAULT_MODEL, **kwargs):
        """공통 채팅 완성 API 호출"""

        def make_request():
            return client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
            )

        response = cls.call_with_retry(make_request)
        return response.choices[0].message.content.strip()


class ArticleProcessor:
    """뉴스 기사 처리를 담당하는 클래스"""

    @staticmethod
    def extract_text(url):
        """URL에서 기사 텍스트 추출"""
        try:
            article = Article(url)
            article.download()
            article.parse()
            return article.text
        except Exception as e:
            logger.error(f"기사 추출 중 오류 발생: {e}")
            return None

    @staticmethod
    def summarize(article_text):
        """기사 요약"""
        try:
            messages = [
                ChatCompletionSystemMessageParam(
                    role="system",
                    content="다음 뉴스 기사를 간결하게 요약해 주세요."
                ),
                ChatCompletionUserMessageParam(
                    role="user",
                    content=article_text
                )
            ]

            return OpenAIService.create_chat_completion(messages)
        except Exception as e:
            logger.error(f"요약 생성 중 오류 발생: {e}")
            return None

    @staticmethod
    def extract_keywords(summary_text, num_keywords=3):
        """요약문에서 핵심 키워드 추출"""
        try:
            messages = [
                ChatCompletionSystemMessageParam(
                    role="system",
                    content=f"다음 요약문에서 핵심 키워드를 {num_keywords}개만 추출해 주세요. "
                            f"단순하게 빈도가 높은 단어가 아니라, 전체 내용에서 중요한 키워드를 추출해주고 "
                            f"키워드만 쉼표로 구분하여 출력해 주세요."
                ),
                ChatCompletionUserMessageParam(
                    role="user",
                    content=summary_text
                )
            ]

            keywords_text = OpenAIService.create_chat_completion(messages)
            return [kw.strip() for kw in keywords_text.split(',')]
        except Exception as e:
            logger.error(f"핵심 키워드 추출 중 오류 발생: {e}")
            return []


class TopicGenerator:
    """토론 주제 생성을 담당하는 클래스"""

    @staticmethod
    def generate(summary_text, keywords):
        """토론 주제 및 설명 생성"""
        try:
            keywords_str = ", ".join(keywords)

            # 토론 주제 생성
            topic_messages = [
                ChatCompletionSystemMessageParam(
                    role="system",
                    content="다음 뉴스 기사 요약문과 핵심 키워드를 바탕으로 "
                            "찬성과 반대로 나뉠 수 있는 토론 주제를 하나만 생성해 주세요.\n"
                            f"요약문: {summary_text}\n"
                            f"핵심 키워드: {keywords_str}\n"
                            "토론 주제:"
                ),
                ChatCompletionUserMessageParam(
                    role="user",
                    content=" "
                )
            ]

            topic = OpenAIService.create_chat_completion(topic_messages)

            # 주제 설명 생성
            description_messages = [
                ChatCompletionSystemMessageParam(
                    role="system",
                    content="당신은 토론 주제에 대한 간략한 설명을 제공하는 도우미입니다."
                ),
                ChatCompletionUserMessageParam(
                    role="user",
                    content=f"다음 토론 주제에 대한 간략한 설명을 100자 이내로 작성해 주세요.\n토론 주제: {topic}"
                )
            ]

            description = OpenAIService.create_chat_completion(description_messages)

            return {
                "topic": topic,
                "description": description
            }
        except Exception as e:
            logger.error(f"토론 주제 생성 중 오류 발생: {e}")
            return {"topic": "생성 실패", "description": "토론 주제 생성에 실패했습니다."}


class DiscussionManager:
    """토론 관리를 담당하는 클래스"""

    @staticmethod
    def create_system_message(topic, user_position, ai_position, difficulty, is_first_message=False):
        """토론용 시스템 메시지 생성"""
        base_message = (
            f"당신은 토론에 참여하는 인공지능 토론자입니다. "
            f"토론 주제는 '{topic}'입니다. "
            f"사용자의 입장은 '{user_position}'이고, 당신은 반대 입장인 '{ai_position}'을 취해야 합니다. "
            f"토론 수준은 '{difficulty}'로 진행합니다. "
        )

        if is_first_message:
            return base_message + "사용자와의 토론을 시작하는 첫 번째 메시지를 작성해주세요. 상대방의 입장을 존중하되, 논리적이고 설득력 있는 주장을 펼쳐야 합니다."
        else:
            return base_message + "사용자의 메시지에 논리적이고 설득력 있게 응답해야 합니다."

    @classmethod
    def start_discussion(cls, topic, user_position, ai_position, difficulty):
        """토론 시작 및 AI 첫 메시지 생성"""
        try:
            system_message = cls.create_system_message(topic, user_position, ai_position, difficulty, True)

            messages = [
                ChatCompletionSystemMessageParam(role="system", content=system_message),
                ChatCompletionUserMessageParam(role="user", content="토론을 시작합니다.")
            ]

            return OpenAIService.create_chat_completion(messages)
        except Exception as e:
            logger.error(f"AI 첫 메시지 생성 중 오류 발생: {e}")
            return "죄송합니다, 토론을 시작하는데 문제가 발생했습니다. 다시 시도해 주세요."

    @classmethod
    def generate_response(cls, topic, user_position, ai_position, difficulty, messages):
        """사용자 메시지에 대한 AI 응답 생성"""
        try:
            system_message = cls.create_system_message(topic, user_position, ai_position, difficulty, False)

            gpt_messages = [
                ChatCompletionSystemMessageParam(role="system", content=system_message)
            ]

            for msg in messages:
                if msg["role"] == "user":
                    gpt_messages.append(ChatCompletionUserMessageParam(role="user", content=msg["content"]))
                else:
                    gpt_messages.append(ChatCompletionAssistantMessageParam(role="assistant", content=msg["content"]))

            return OpenAIService.create_chat_completion(gpt_messages)
        except Exception as e:
            logger.error(f"AI 응답 생성 중 오류 발생: {e}")
            return "죄송합니다, 응답을 생성하는데 문제가 발생했습니다. 다시 시도해 주세요."

    @classmethod
    def generate_comprehensive_summary(cls, topic, user_position, ai_position, messages):
        """전체 토론 내용을 압축 없이 요약 생성"""
        try:
            logger.info(f"전체 토론 요약 시작 - 메시지 수: {len(messages)}")

            # 전체 메시지를 시간순으로 정렬하여 대화 흐름 구성
            conversation_parts = []
            for i, msg in enumerate(messages, 1):
                role = "사용자" if msg["role"] == "user" else "AI"
                conversation_parts.append(f"[{i}] {role}: {msg['content']}")

            # 전체 대화 내용
            full_conversation = "\n\n".join(conversation_parts)

            logger.info(f"전체 대화 길이: {len(full_conversation)} 문자")

            # 포괄적인 요약을 위한 시스템 메시지
            system_message = (
                f"당신은 토론 분석 전문가입니다. 다음은 '{topic}'에 관한 완전한 토론 내용입니다.\n"
                f"사용자는 '{user_position}' 입장이었고, AI는 '{ai_position}' 입장이었습니다.\n\n"
                f"전체 토론을 종합적으로 분석하여 다음 내용을 포함한 상세한 요약을 작성해 주세요:\n"
                f"1. 토론의 전반적인 흐름과 주요 논점들\n"
                f"2. 각 측(사용자와 AI)의 핵심 주장과 근거들\n"
                f"3. 토론 과정에서 제기된 중요한 쟁점들\n"
                f"4. 양측의 논리적 강점과 약점 분석\n"
                f"5. 토론의 결론이나 합의점(있다면)\n\n"
                f"압축하지 말고 충분히 상세하게 분석해 주세요."
            )

            messages_for_gpt = [
                ChatCompletionSystemMessageParam(role="system", content=system_message),
                ChatCompletionUserMessageParam(role="user", content=full_conversation)
            ]

            # 더 긴 응답을 위해 max_tokens 증가
            summary = OpenAIService.create_chat_completion(
                messages_for_gpt,
                temperature=0.7,
                max_tokens=1500  # 더 상세한 요약을 위해 토큰 수 증가
            )

            logger.info(f"요약 생성 완료 - 길이: {len(summary)} 문자")
            return summary

        except Exception as e:
            logger.error(f"토론 요약 생성 중 오류 발생: {e}")
            return "죄송합니다, 토론 요약을 생성하는데 문제가 발생했습니다. 토론 내용이 너무 길거나 복잡할 수 있습니다."


# API 엔드포인트 정의

@app.route('/api/extract', methods=['POST'])
def extract_api():
    """URL 처리 및 키워드/요약 추출 API"""
    data = request.json
    url = data.get('url')

    if not url:
        return jsonify({"error": "URL이 필요합니다"}), 400

    logger.info(f"Processing URL: {url}")

    # 기사 텍스트 추출
    article_text = ArticleProcessor.extract_text(url)
    if not article_text:
        return jsonify({"error": "기사를 추출할 수 없습니다"}), 400

    # 기사 요약
    summary = ArticleProcessor.summarize(article_text)
    if not summary:
        return jsonify({"error": "요약을 생성할 수 없습니다"}), 500

    # 키워드 추출
    keywords = ArticleProcessor.extract_keywords(summary)

    return jsonify({
        "summary": summary,
        "keywords": keywords
    })


@app.route('/api/topic', methods=['POST'])
def topic_api():
    """토론 주제 생성 API"""
    data = request.json
    summary = data.get('summary')
    keywords = data.get('keywords')

    if not summary or not keywords:
        return jsonify({"error": "요약문과 키워드가 필요합니다"}), 400

    logger.info("Generating discussion topic")

    topic_data = TopicGenerator.generate(summary, keywords)
    return jsonify(topic_data)


@app.route('/api/discussion/start', methods=['POST'])
def start_discussion_api():
    """토론 시작 API"""
    data = request.json
    topic = data.get('topic')
    user_position = data.get('userPosition')
    ai_position = data.get('aiPosition')
    difficulty = data.get('difficulty')

    if not all([topic, user_position, ai_position, difficulty]):
        return jsonify({"error": "모든 파라미터가 필요합니다"}), 400

    logger.info(f"Starting discussion: {topic}, user: {user_position}, AI: {ai_position}, level: {difficulty}")

    ai_message = DiscussionManager.start_discussion(topic, user_position, ai_position, difficulty)
    return jsonify({"message": ai_message})


@app.route('/api/discussion/message', methods=['POST'])
def message_api():
    """토론 메시지 처리 API"""
    data = request.json
    topic = data.get('topic')
    user_position = data.get('userPosition')
    ai_position = data.get('aiPosition')
    difficulty = data.get('difficulty')
    messages = data.get('messages')

    if not all([topic, user_position, ai_position, difficulty, messages]):
        return jsonify({"error": "모든 파라미터가 필요합니다"}), 400

    logger.info(f"Processing message for topic: {topic}")

    ai_response = DiscussionManager.generate_response(topic, user_position, ai_position, difficulty, messages)
    return jsonify({"message": ai_response})


@app.route('/api/discussion/summary', methods=['POST'])
def summary_api():
    """전체 토론 요약 API (압축 없음)"""
    data = request.json
    topic = data.get('topic')
    user_position = data.get('userPosition')
    ai_position = data.get('aiPosition')
    messages = data.get('messages')

    if not all([topic, user_position, ai_position, messages]):
        return jsonify({"error": "모든 파라미터가 필요합니다"}), 400

    logger.info(f"Generating comprehensive summary for topic: {topic}, messages count: {len(messages)}")

    try:
        # 전체 메시지를 압축 없이 처리
        summary = DiscussionManager.generate_comprehensive_summary(topic, user_position, ai_position, messages)
        return jsonify({"summary": summary})
    except RateLimitError as e:
        logger.error(f"Rate limit exceeded during summary generation: {e}")
        error_message = "토론 분석 중 API 요청 한도에 도달했습니다. 잠시 후 다시 시도해주세요."
        return jsonify({"summary": error_message, "error": str(e)}), 200
    except Exception as e:
        logger.error(f"Summary generation error: {e}")
        error_message = "토론 요약 중 오류가 발생했습니다. 토론 내용이 너무 길거나 복잡한 경우 시간이 더 걸릴 수 있습니다."
        return jsonify({"summary": error_message, "error": str(e)}), 200


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
