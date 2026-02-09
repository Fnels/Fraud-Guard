# Fraud-Guard

# Docker Infrastructure User Guide (Linux)

이 가이드는 Linux 환경에서 프로젝트 Docker 인프라를 배포하고 사용하는 방법을 설명합니다.

## !!사전설정 : Docker/.env 파일 생성 후 아래의 내용을 붙여넣어주세요.
# Database
MYSQL_ROOT_PASSWORD=root
MYSQL_DATABASE=fraud_detection

## 1. 사전 요구 사항 (Prerequisites)

*   **Docker Engine**: 최신 버전 설치 권장
*   **Docker Compose**: Plugin 또는 Standalone 설치 필요

## 2. 디렉토리 구조 설정

프로젝트 루트 디렉토리(`2nd_project`)에 `requirements.txt`와 소스 코드 폴더(`src`)가 위치해야 합니다. Docker 설정은 `Docker/` 폴더 내에 있습니다.

```bash
2nd_project/
├── data/
│   ├── geo/
│   │   ├── cb_2023_us_state_20m.shx
│   │   ├── cb_2023_us_state_20m.zip
│   │   └── us_states.geojson
│   └── origin/
│       ├── cards_data.csv
│       ├── merchants_data.csv
│       ├── train_fraud_labels.json
│       ├── transactions_data.csv
│       └── users_data.csv
├── Docker/
│   ├── .env              <-- 환경 변수 설정
│   ├── compose.yml
│   ├── Dockerfile.python
│   └── Dockerfile.spark
│
├── src/                  <-- 개발한 Python 소스 코드를 이곳에 위치
│   ├── ML/
│   │   ├── ML.ipynb
│   │   └── ML.py
│   └── utils/
│       ├── __pycache__/
│       │   └── formatter.cpython-312.pyc
│       ├── formatter.py
│       ├── __init__.py
│       ├── consumer1.py
│       ├── consumer2.py
│       ├── init_db.py
│       ├── init_geo.py
│       ├── init_views.py
│       ├── redis_warmer.py
│       ├── terminal.py
│       ├── test_flow.py
│       └── worker.py
│
├── venv/
├── .gitignore
├── checkpoint.txt
├── README.md
└── requirements.txt   <-- Python 의존성 목록 (모든 컨테이너 공통)
    
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
docker compose logs -f consumer-group-1

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
SREM check:users "1"

# ID가 1인 유저가 있는지 확인(있으면 1, 없으면 0 반환 )
SISMEMBER check:users "1"

# 접속 후 소스 코드 확인
ls -l /app/src/
```

## 5. 서비스 접속 정보

| 서비스 | 접속 주소 | 설명 |
| :--- | :--- | :--- |
| **Kafka UI** | `http://localhost:8080` | Kafka 토픽 및 메시지 모니터링 |
| **Grafana** | `http://localhost:3000` | 전체 시스템 모니터링 대시보드 |
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
