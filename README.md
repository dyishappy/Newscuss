# 📰 Newscuss Chatbot  
**AI 뉴스 토론 플랫폼 - Newscuss Chatbot**

---

## 📌 프로젝트 소개  
**Newscuss Chatbot**은 뉴스 기반 AI 토론 챗봇입니다.  
사용자가 뉴스 URL을 입력하면, AI가 해당 기사를 요약 및 분석하여 논쟁 거리를 제공합니다.  
이후 사용자는 찬/반 입장을 선택하고, AI와 논리적인 토론을 실시간으로 진행할 수 있습니다.

---

## 💡 Motivation  
최근 대한민국 사회에서는 정치, 사회, 경제 등 다양한 이슈로 인해 **집단 양극화** 현상이 심화되고 있습니다.  
집단 양극화란, 특정 의견이나 입장을 가진 집단 간에 갈등이 깊어지고 타인의 관점을 수용하지 않는 현상을 말합니다.  

이러한 문제를 해결하기 위해 **Newscuss Chatbot**은 사용자가 다양한 입장에서 뉴스를 바라보고 토론할 수 있는 기회를 제공합니다.  
AI와의 대화를 통해 다른 관점을 이해하고, 자신의 입장을 점검하는 과정을 통해  
더욱 **건전하고 균형 잡힌 소통 문화**를 확산시키는 데 기여하고자 합니다.

---

## 🎯 주요 목표  
- AI가 뉴스 기사를 요약하고 핵심 논쟁 거리를 도출  
- 사용자의 입장 선택(찬성/반대) 후 AI가 반대 논리를 제시  
- 자연어 처리를 활용한 논리적인 토론 진행  
- 최신 뉴스 데이터를 반영하여 신뢰성 높은 토론 제공

---

## 🔥 핵심 기능  
✅ **뉴스 기사 요약** – AI가 입력된 뉴스를 간결하고 정확하게 요약  
✅ **논쟁 거리 생성** – 기사 내용을 바탕으로 토론 주제 자동 생성  
✅ **입장 선택 및 토론 진행** – 사용자가 찬/반을 선택하면, AI가 반대 입장에서 논리 전개  
✅ **대화형 토론 UI** – 웹 기반 인터페이스에서 실시간 토론 지원  
✅ **사용자 맞춤형 논쟁 수준 조정** – 초보자부터 전문가까지 다양한 스타일 지원

---

## 🛠️ 기술 스택

### 프론트엔드
- **Framework**: Next.js (v15.3.0)
- **UI Library**: React (v19.0.0)
- **Styling**: TailwindCSS (v4)
- **State Management**: React Context API

### 백엔드
- **Framework**: Spring Boot (v3.4.4)
- **AI Integration**: Python Flask API
- **Language Models**: OpenAI GPT
- **News Article Processing**: newspaper3k

---

## 💻 아키텍처

```
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│                 │      │                 │      │                 │
│  Next.js        │◄────►│  Spring Boot    │◄────►│  Python Flask   │
│  Frontend       │      │  Backend        │      │  AI Service     │
│                 │      │                 │      │                 │
└─────────────────┘      └─────────────────┘      └─────────────────┘
                                                          ▲
                                                          │
                                                          ▼
                                                  ┌─────────────────┐
                                                  │                 │
                                                  │  OpenAI API     │
                                                  │                 │
                                                  └─────────────────┘
```

---

## 🚀 시작하기

### 프론트엔드 실행
```bash
# Clone frontend repository
git clone https://github.com/sttarrynight/newscuss-fe.git
cd newscuss-fe

# Install dependencies
npm install

# Run development server
npm run dev
```

### 백엔드 실행
```bash
# Clone backend repository
git clone https://github.com/sttarrynight/newscuss-be.git
cd newscuss-be

# Build with Gradle
./gradlew build

# Run Spring Boot application
java -jar build/libs/newscuss-be-0.0.1-SNAPSHOT.jar
```

### Python Flask 서비스 실행
```bash
# Clone backend repository (if not already done)
git clone https://github.com/sttarrynight/newscuss-be.git
cd newscuss-be

# Install Python dependencies
pip install -r requirements.txt

# Run Flask server
python app.py
```

---

## 📋 프로젝트 구조

### 프론트엔드
```
src/
├── app/                 # Next.js 페이지 컴포넌트
├── components/          # 재사용 가능한 UI 컴포넌트
├── context/             # React Context (상태 관리)
├── services/            # API 서비스 통신 로직
└── utils/               # 유틸리티 함수
```

### 백엔드
```
src/main/
├── java/
│   └── com/example/newscussbe/
│       ├── controller/  # REST API 엔드포인트
│       ├── dto/         # 데이터 전송 객체
│       ├── service/     # 비즈니스 로직
│       └── client/      # Python API 클라이언트
└── resources/           # 설정 파일
```

---

## 📚 레포지토리

- **프론트엔드**: [https://github.com/sttarrynight/newscuss-fe](https://github.com/sttarrynight/newscuss-fe)
- **백엔드**: [https://github.com/sttarrynight/newscuss-be](https://github.com/sttarrynight/newscuss-be)

---

## 📝 라이센스

MIT License
