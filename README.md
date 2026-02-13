# Fraud-Guard

# Docker Infrastructure User Guide (Linux)

이 가이드는 Linux 환경에서 프로젝트 Docker 인프라를 배포하고 사용하는 방법을 설명합니다.

## !!사전설정 : Docker/.env 파일 생성 후 아래의 내용을 붙여넣어주세요.
MYSQL_ROOT_PASSWORD=root
MYSQL_DATABASE=fraud_detection

## 기술 스택 (Tech Stack)

### Infrastructure
- **Docker** & **Docker Compose** - 컨테이너 오케스트레이션
- **Kafka** (apache/kafka:4.1.1) - 메시지 브로커
- **Redis** (latest) - 인메모리 캐시
- **MySQL** (8.0) - 관계형 데이터베이스

### Data Processing
- **Apache Spark** (3.5.0) - 분산 스트리밍 처리
- **Python** (3.10+) - 데이터 파이프라인 구현

### Libraries
- **kafka-python** - Kafka Producer/Consumer
- **confluent-kafka** - Kafka 클라이언트
- **pymysql** - MySQL 연결
- **redis-py** - Redis 클라이언트
- **pandas** - 데이터 처리
- **Flask** - API 서버

### Monitoring
- **Kafka UI** - Kafka 모니터링
- **Tableau** - MySQL 데이터 기반 대시보드 시각화

---
## 1. 사전 요구 사항 (Prerequisites)

*   **Docker Engine**: 최신 버전 설치 권장
*   **Docker Compose**: Plugin 또는 Standalone 설치 필요

## 1-2. 환경 설정
```
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. 디렉토리 구조 설정

프로젝트 루트 디렉토리(`2nd_project`)에 `requirements.txt`와 소스 코드 폴더(`src`)가 위치해야 합니다. Docker 설정은 `Docker/` 폴더 내에 있습니다.

```bash
2nd_project/
├── data/                            # 데이터 저장소
│   ├── geo/                         # 지리 데이터 (Tableau 시각화용)
│   │   ├── cb_2023_us_state_20m.shx
│   │   ├── cb_2023_us_state_20m.zip
│   │   └── us_states.geojson
│   ├── ML/                          # 학습된 ML 모델
│   │   ├── tier1model.cbm           # 1차 사기 탐지 모델
│   │   └── tier2model.cbm           # 2차 사기 탐지 모델
│   │
│   └── origin/                      # 원본 데이터셋
│       ├── cards_data.csv           # 카드 정보
│       ├── merchants_data.csv       # 가맹점 정보
│       ├── train_fraud_labels.json  # 사기 라벨 (학습용)
│       ├── transactions_data.csv    # 거래 데이터
│       └── users_data.csv           # 사용자 정보
├── Docker/
│   ├── .env                         # Docker 설정 파일
│   ├── compose.yml                  # Docker Compose 오케스트레이션
│   ├── Dockerfile.python            # Python 컨테이너 이미지
│   └── Dockerfile.spark             # Spark 컨테이너 이미지
│
├── src/                  <-- 개발한 Python 소스 코드를 이곳에 위치
│   ├── ML/                          # 머신러닝 모듈
│   │   └── ML.ipynb                 # 모델 학습 노트북
│   └── utils/
│   │   └── formatter.py             # 데이터 포맷 변환
│   │
│   ├──  __init__.py                 # Python 패키지 초기화
│   ├── consumer1.py                 # Spark Consumer (transactions_data 적재)
│   ├── consumer2.py                 # Spark Consumer (윈도우 집계)
│   ├── init_db.py                   # MySQL 초기 데이터 로딩
│   ├── init_geo.py                  # 지리 데이터
│   ├── init_views.py                # MySQL 뷰 생성
│   ├── redis_warmer.py              # Redis 캐시 사전 로딩
│   ├── terminal.py                  # Flask Producer (Kafka 전송)
│   ├── test_flow.py                 # 프로듀서 테스트
│   └── worker.py                    # 무결성 검증 + ML Worker
│
├── venv/                            # Python 가상환경 (Git 제외)
├── .gitignore                       # Git 제외 파일 목록
├── checkpoint.txt                   # Producer 체크포인트 (재시작용)
├── README.md                        # 프로젝트 문서
└── requirements.txt                 # Python 의존성 목록 (모든 컨테이너 공통)
    
```

## 3. 실행 방법 (Usage)

터미널을 열고 `Docker` 디렉토리로 이동하여 실행합니다.

```bash
# 1. Docker 디렉토리로 이동
cd 2nd_project/Docker

# 2. 컨테이너 빌드 및 백그라운드 실행
# (상위 폴더의 requirements.txt를 참조하여 빌드됩니다)
docker compose up -d --build

# 3. 실행 상태 확인
docker compose ps

# 4. 프로듀서(terminal.py) 플라스크 서버 실행및 로그 확인
docker compose logs -f flask-producer

# 5. consumer1.py 로그 확인
docker-compose logs -f consumer-group-1

# 6. worker.py 로그 확인
docker-compose logs -f ml-worker

# 7. consumer2.py 로그 확인
docker compose logs -f consumer-group-2

# 8. 터미널에서 MySQL 강제종료및 수동 실행
docker stop mysql
docker start mysql
```

## 4. 개발 환경 접속 (Python Dev)

Python 파일 실행, 테스트, 디버깅을 위해 `python-dev` 컨테이너에 접속할 수 있습니다. 이 컨테이너는 프로젝트 루트(`2nd_project`)를 `/app`으로 마운트하고 있습니다.

```bash
# python-dev 컨테이너 내부 쉘 접속
docker exec -it python-dev bash

# 레데스 컨테이너 접속
docker exec -it redis redis-cli

# 1번 유저데이터 삭제
DEL info:user:1

# ID가 1인 유저가 있는지 확인
GET info:user:1

# 접속 후 소스 코드 확인
ls -l /app/src/
```

## 5. 서비스 접속 정보

| 서비스 | 접속 주소 | 설명 |
| :--- | :--- | :--- |
| **Kafka UI** | `http://localhost:8080` | Kafka 토픽 및 메시지 모니터링 |
| **Tableau** | 별도 설치 | MySQL 연결 후 대시보드 구성 |
| **Spark Master** | `http://localhost:8081` | Spark 클러스터 상태 확인 |
| **MySQL** | `localhost:3306` | DB 접근 (User/PW: root/root - compose.yml 참조) |
| **Redis** | `localhost:6379` | 캐시 서버 |

## 6. 종료 방법

```bash
# 컨테이너 종료 및 네트워크 제거
docker compose down

# (옵션) 볼륨 데이터까지 삭제하고 싶을 경우
docker compose down -v
```

## MySQL 간편접속

```bash
docker exec -it mysql mysql -u root -p
```
