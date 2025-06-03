def generate_discussion_feedback(topic, user_position, ai_position, messages, model="gpt-4o"):
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

        # 사용자 메시지만 필터링 (피드백 대상)
        user_messages = [msg for msg in compressed_messages if msg["role"] == "user"]
        
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
            f"사용자의 토론 능력에 대해 다음 5개 카테고리로 구체적이고 건설적인 피드백을 제공해 주세요:\n\n"
            f"1. **논리적 사고력**: 논증 구조, 논리적 일관성, 인과관계 파악, 반박 논리\n"
            f"2. **근거와 증거 활용**: 사실적 정확성, 근거의 적절성, 출처의 신뢰도, 다각적 접근\n"
            f"3. **의사소통 능력**: 명확성, 설득력, 경청 능력, 응답의 적절성\n"
            f"4. **토론 태도와 매너**: 상호 존중, 공정성, 건설적 자세, 감정 조절\n"
            f"5. **창의성과 통찰력**: 독창적 관점, 문제 해결 능력, 깊이 있는 분석\n\n"
            f"각 카테고리별로 구체적인 예시와 함께 잘한 점과 개선점을 제시하고, "
            f"실질적인 개선 방안을 제안해 주세요. 피드백은 격려적이면서도 구체적이어야 합니다."
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
        def make_feedback_request():
            response = client.chat.completions.create(
                model=model,
                messages=chat_messages,
                temperature=0.3,  # 피드백은 일관성이 중요하므로 낮은 온도
                max_tokens=1200  # 피드백이므로 더 긴 응답 허용
            )
            return response.choices[0].message.content.strip()

        return call_openai_with_retry(make_feedback_request)

    except Exception as e:
        logger.error(f"토론 피드백 생성 중 오류 발생: {e}")
        # 피드백 실패 시 기본 메시지 제공
        return {
            "error": True,
            "message": "죄송합니다, 토론 피드백을 생성하는데 문제가 발생했습니다. 토론 내용이 너무 길거나 복잡할 수 있습니다.",
            "fallback_feedback": {
                "논리적 사고력": "토론 내용을 분석할 수 없어 평가가 어렵습니다.",
                "근거와 증거 활용": "토론 내용을 분석할 수 없어 평가가 어렵습니다.",
                "의사소통 능력": "토론 내용을 분석할 수 없어 평가가 어렵습니다.",
                "토론 태도와 매너": "토론 내용을 분석할 수 없어 평가가 어렵습니다.",
                "창의성과 통찰력": "토론 내용을 분석할 수 없어 평가가 어렵습니다."
            }
        }

# 선택적: 구조화된 피드백을 위한 함수
def generate_structured_discussion_feedback(topic, user_position, ai_position, messages, model="gpt-4o"):
    """
    JSON 형태로 구조화된 피드백을 반환하는 함수
    """
    try:
        # 위와 동일한 메시지 압축 로직
        def truncate_message(text, max_length=200):
            if len(text) <= max_length:
                return text
            return text[:max_length] + "..."

        compressed_messages = []
        if len(messages) > 15:
            important_messages = (
                    messages[:3] +
                    messages[3:-7:len(messages[3:-7]) // 5 + 1] +
                    messages[-7:]
            )
            compressed_messages = important_messages
        else:
            compressed_messages = messages

        compressed_text = []
        for msg in compressed_messages:
            role = msg["role"]
            content = truncate_message(msg["content"])
            compressed_text.append(f"{role}: {content}")

        conversation_text = "\n".join(compressed_text)

        system_message = (
            f"다음은 '{topic}'에 관한 토론 내용입니다. "
            f"사용자는 '{user_position}' 입장이었고, AI는 '{ai_position}' 입장이었습니다. "
            f"사용자의 토론 능력을 다음 JSON 형태로 평가해 주세요:\n\n"
            f"{{\n"
            f'  "논리적_사고력": {{\n'
            f'    "점수": 1-10 점수,\n'
            f'    "잘한_점": "구체적인 칭찬",\n'
            f'    "개선점": "구체적인 개선 사항",\n'
            f'    "개선_방안": "실질적인 조언"\n'
            f'  }},\n'
            f'  "근거와_증거_활용": {{ ... }},\n'
            f'  "의사소통_능력": {{ ... }},\n'
            f'  "토론_태도와_매너": {{ ... }},\n'
            f'  "창의성과_통찰력": {{ ... }},\n'
            f'  "전체_평가": {{\n'
            f'    "총점": "평균 점수",\n'
            f'    "종합_의견": "전반적인 피드백",\n'
            f'    "핵심_개선_포인트": "가장 중요한 개선점 3가지"\n'
            f'  }}\n'
            f"}}\n\n"
            f"반드시 유효한 JSON 형태로만 응답해 주세요."
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

        def make_structured_feedback_request():
            response = client.chat.completions.create(
                model=model,
                messages=chat_messages,
                temperature=0.2,  # JSON 형태이므로 더 낮은 온도
                max_tokens=1500
            )
            return response.choices[0].message.content.strip()

        return call_openai_with_retry(make_structured_feedback_request)

    except Exception as e:
        logger.error(f"구조화된 토론 피드백 생성 중 오류 발생: {e}")
        return None


        


# 토론 점수 평가 생성 (메시지 압축 및 재시도 로직 최적화)
def generate_discussion_scores(topic, user_position, ai_position, messages, model="gpt-4o"):
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

        # 사용자 메시지만 필터링 (피드백 대상)
        user_messages = [msg for msg in compressed_messages if msg["role"] == "user"]
        
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
            f"사용자의 토론 능력을 다음 5개 카테고리별로 0~100점 사이의 점수로 평가해 주세요.\n\n"
            f"평가 기준:\n"
            f"1. **논리적_사고력** (0-100점): 논증 구조, 논리적 일관성, 인과관계 파악, 반박 논리\n"
            f"2. **근거와_증거_활용** (0-100점): 사실적 정확성, 근거의 적절성, 출처의 신뢰도, 다각적 접근\n"
            f"3. **의사소통_능력** (0-100점): 명확성, 설득력, 경청 능력, 응답의 적절성\n"
            f"4. **토론_태도와_매너** (0-100점): 상호 존중, 공정성, 건설적 자세, 감정 조절\n"
            f"5. **창의성과_통찰력** (0-100점): 독창적 관점, 문제 해결 능력, 깊이 있는 분석\n\n"
            f"다음 JSON 형태로만 응답해 주세요:\n"
            f"{{\n"
            f'  "논리적_사고력": 85,\n'
            f'  "근거와_증거_활용": 72,\n'
            f'  "의사소통_능력": 90,\n'
            f'  "토론_태도와_매너": 88,\n'
            f'  "창의성과_통찰력": 75,\n'
            f'  "총점": 82\n'
            f"}}\n\n"
            f"반드시 유효한 JSON 형태로만 응답하고, 각 점수는 0~100 사이의 정수여야 합니다."
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
        def make_score_request():
            response = client.chat.completions.create(
                model=model,
                messages=chat_messages,
                temperature=0.2,  # 점수 평가는 일관성이 중요하므로 낮은 온도
                max_tokens=300   # JSON 형태이므로 짧은 응답
            )
            return response.choices[0].message.content.strip()

        return call_openai_with_retry(make_score_request)

    except Exception as e:
        logger.error(f"토론 점수 평가 생성 중 오류 발생: {e}")
        # 점수 평가 실패 시 기본 JSON 제공
        return {
            "논리적_사고력": 0,
            "근거와_증거_활용": 0,
            "의사소통_능력": 0,
            "토론_태도와_매너": 0,
            "창의성과_통찰력": 0,
            "총점": 0,
            "error": "점수 평가를 생성하는데 문제가 발생했습니다."
        }





# 선택적: 점수와 간단한 코멘트를 함께 제공하는 함수
def generate_discussion_scores_with_comments(topic, user_position, ai_position, messages, model="gpt-4o"):
    """
    점수와 함께 각 카테고리별 간단한 코멘트를 반환하는 함수
    """
    try:
        # 위와 동일한 메시지 압축 로직
        def truncate_message(text, max_length=200):
            if len(text) <= max_length:
                return text
            return text[:max_length] + "..."

        compressed_messages = []
        if len(messages) > 15:
            important_messages = (
                    messages[:3] +
                    messages[3:-7:len(messages[3:-7]) // 5 + 1] +
                    messages[-7:]
            )
            compressed_messages = important_messages
        else:
            compressed_messages = messages

        compressed_text = []
        for msg in compressed_messages:
            role = msg["role"]
            content = truncate_message(msg["content"])
            compressed_text.append(f"{role}: {content}")

        conversation_text = "\n".join(compressed_text)

        system_message = (
            f"다음은 '{topic}'에 관한 토론 내용입니다. "
            f"사용자는 '{user_position}' 입장이었고, AI는 '{ai_position}' 입장이었습니다. "
            f"사용자의 토론 능력을 5개 카테고리별로 0~100점으로 평가하고, 각각에 대해 한 줄 코멘트를 제공해 주세요.\n\n"
            f"다음 JSON 형태로만 응답해 주세요:\n"
            f"{{\n"
            f'  "논리적_사고력": {{\n'
            f'    "점수": 85,\n'
            f'    "코멘트": "논리적 구조가 체계적이나 반박 논리 보완 필요"\n'
            f'  }},\n'
            f'  "근거와_증거_활용": {{\n'
            f'    "점수": 72,\n'
            f'    "코멘트": "사례 제시는 좋으나 출처 명시가 부족함"\n'
            f'  }},\n'
            f'  "의사소통_능력": {{\n'
            f'    "점수": 90,\n'
            f'    "코멘트": "명확하고 설득력 있는 표현력 우수"\n'
            f'  }},\n'
            f'  "토론_태도와_매너": {{\n'
            f'    "점수": 88,\n'
            f'    "코멘트": "상대방 존중하며 건설적인 자세 유지"\n'
            f'  }},\n'
            f'  "창의성과_통찰력": {{\n'
            f'    "점수": 75,\n'
            f'    "코멘트": "독창적 관점은 있으나 깊이 있는 분석 아쉬움"\n'
            f'  }},\n'
            f'  "총점": 82,\n'
            f'  "종합_코멘트": "전반적으로 우수한 토론 실력을 보여주었습니다."\n'
            f"}}\n\n"
            f"반드시 유효한 JSON 형태로만 응답하고, 각 점수는 0~100 사이의 정수여야 합니다."
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

        def make_scores_with_comments_request():
            response = client.chat.completions.create(
                model=model,
                messages=chat_messages,
                temperature=0.2,
                max_tokens=600
            )
            return response.choices[0].message.content.strip()

        return call_openai_with_retry(make_scores_with_comments_request)

    except Exception as e:
        logger.error(f"점수 및 코멘트 생성 중 오류 발생: {e}")
        return {
            "논리적_사고력": {"점수": 0, "코멘트": "평가 불가"},
            "근거와_증거_활용": {"점수": 0, "코멘트": "평가 불가"},
            "의사소통_능력": {"점수": 0, "코멘트": "평가 불가"},
            "토론_태도와_매너": {"점수": 0, "코멘트": "평가 불가"},
            "창의성과_통찰력": {"점수": 0, "코멘트": "평가 불가"},
            "총점": 0,
            "종합_코멘트": "점수 평가에 문제가 발생했습니다.",
            "error": True
        }