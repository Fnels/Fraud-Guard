import os
import json
import time
import pymysql
import redis  # 추가
from confluent_kafka import Consumer, Producer
from dotenv import load_dotenv
from pathlib import Path

# ---------------------------------------------------------------------------
# 0. Load Environment Variables (.env)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR.parent / 'Docker' / '.env'

if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
    print(f"[INFO] Loaded .env from: {ENV_PATH}")
else:
    print(f"[WARNING] .env file not found at: {ENV_PATH}")

# ---------------------------------------------------------------------------
# 1. Configuration & Connection Setup
# ---------------------------------------------------------------------------
KAFKA_BROKER = 'kafka:9092'
SOURCE_TOPIC = 'raw-topic'
TARGET_TOPIC = '2nd-topic'
CONSUMER_GROUP = 'fraud-core-group'

DB_HOST = 'mysql'
DB_USER = 'root'
DB_PASSWORD = os.environ.get('MYSQL_ROOT_PASSWORD', 'root') 
DB_NAME = os.environ.get('MYSQL_DATABASE', 'fraud_detection') 

# Redis Configuration
REDIS_HOST = 'redis'
REDIS_PORT = 6379

# Initialize Clients
consumer_conf = {
    'bootstrap.servers': KAFKA_BROKER,
    'group.id': CONSUMER_GROUP,
    'auto.offset.reset': 'latest',
    'enable.auto.commit': True
}
consumer = Consumer(consumer_conf)
consumer.subscribe([SOURCE_TOPIC])

producer_conf = {'bootstrap.servers': KAFKA_BROKER}
producer = Producer(producer_conf)

# Redis Client 추가 (decode_responses=True로 문자열 처리 편하게)
r = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)

# ---------------------------------------------------------------------------
# 2. Helper Functions
# ---------------------------------------------------------------------------

def get_db_connection():
    return pymysql.connect(
        host=DB_HOST, user=DB_USER, password=DB_PASSWORD, db=DB_NAME,
        charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor
    )

def load_data_to_redis():
    """
    [Warming] 시스템 시작 시 MySQL 데이터를 레디스로 1회 적재합니다.
    PM님이 말씀하신 '레디스 캐시화' 단계입니다.
    """
    print("[INFO] Warming up Redis cache from MySQL...")
    start_warm = time.time()
    connection = None
    try:
        connection = get_db_connection()
        with connection.cursor() as cursor:
            # 1. Users 데이터 적재 (Set)
            cursor.execute("SELECT id FROM users_data")
            users = [str(row['id']) for row in cursor.fetchall()]
            if users:
                r.sadd("check:users", *users)

            # 2. Cards 데이터 적재 (Key-Value: card_id -> client_id)
            cursor.execute("SELECT id, client_id FROM cards_data")
            cards = cursor.fetchall()
            for card in cards:
                r.set(f"check:card:{card['id']}", card['client_id'])

            # 3. Merchants 데이터 적재 (Set)
            cursor.execute("SELECT id FROM merchants_data")
            merchants = [str(row['id']) for row in cursor.fetchall()]
            if merchants:
                r.sadd("check:merchants", *merchants)
                
        elapsed = time.time() - start_warm
        print(f"[SUCCESS] Redis Warming Complete! ({elapsed:.2f}s)")
        print(f" - Users: {len(users)}, Cards: {len(cards)}, Merchants: {len(merchants)}")
    except Exception as e:
        print(f"[ERROR] Redis Warming Failed: {e}")
    finally:
        if connection: connection.close()

def check_integrity_redis(data):
    """
    Logic 1: 무결성 검증 (Redis-based)
    MySQL을 전혀 호출하지 않고 레디스 메모리에서만 검사합니다.
    """
    start_time = time.time()
    try:
        # 1. Client ID 검증 (Set 조회)
        if not r.sismember("check:users", str(data['client_id'])):
            # print(f"[FAIL] Invalid Client ID: {data['client_id']}")
            return False

        # 2. Card ID 존재 및 소유주 일치 여부 (String 조회)
        cached_client_id = r.get(f"check:card:{data['card_id']}")
        if cached_client_id != str(data['client_id']):
            # print(f"[FAIL] Invalid Card ID/Owner Mismatch: {data['card_id']}")
            return False

        # 3. Merchant ID 검증 (Set 조회)
        if not r.sismember("check:merchants", str(data['merchant_id'])):
            # print(f"[FAIL] Invalid Merchant ID: {data['merchant_id']}")
            return False

        return True 

    except Exception as e:
        print(f"[ERROR] Redis Check Failed: {e}")
        return False
    finally:
        # 성능 지표를 위해 실행 시간만 계산해서 반환 (로그 출력은 통계에서 처리)
        pass

def delivery_report(err, msg):
    if err is not None:
        print(f'[ERROR] Message delivery failed: {err}')

# ---------------------------------------------------------------------------
# 3. Main Processor Loop
# ---------------------------------------------------------------------------

def main():
    # 1회성 데이터 적재 실행
    load_data_to_redis()
    
    print(f"[INFO] Worker started. Monitoring Redis-based Integrity Check.")
    print("[INFO] Waiting for messages...")
    
    total_checks = 0
    total_time = 0.0
    check_times = []
    
    try:
        while True:
            msg = consumer.poll(1.0)

            if msg is None: continue
            if msg.error():
                print(f"[ERROR] Consumer error: {msg.error()}")
                continue

            try:
                raw_data = json.loads(msg.value().decode('utf-8'))
                order_id = raw_data.get('id')
            except Exception as e:
                print(f"[ERROR] JSON Parsing failed: {e}")
                continue

            # -------------------------------------------------------
            # Step 1.5: Integrity Check (Redis)
            # -------------------------------------------------------
            check_start = time.time()
            is_valid = check_integrity_redis(raw_data) # 레디스 함수로 교체
            check_time = (time.time() - check_start) * 1000

            total_checks += 1
            total_time += check_time
            check_times.append(check_time)

            # Step 2: ML Fraud Check (Placeholder)
            is_fraud = False 
            
            # Kafka 전송 데이터 구성
            output_data = raw_data.copy()
            output_data['is_valid'] = is_valid
            output_data['is_fraud'] = is_fraud
            
            producer.produce(
                TARGET_TOPIC,
                json.dumps(output_data).encode('utf-8'),
                callback=delivery_report
            )
            producer.poll(0)

            # 100건마다 통계 출력 (레디스 성능 체감을 위해)
            if total_checks % 100 == 0:
                avg_time = total_time / total_checks
                recent_100 = check_times[-100:]
                recent_avg = sum(recent_100) / len(recent_100)
                min_time = min(recent_100)
                max_time = max(recent_100)
                
                print("\n" + "⚡" * 30)
                print(f"📊 [Redis 캐시 검증 통계] {total_checks}건 처리")
                print(f"   누적 평균 속도: {avg_time:.4f}ms")
                print(f"   최근 100건 평균: {recent_avg:.4f}ms")
                print(f"   최소/최대 속도: {min_time:.4f}ms / {max_time:.4f}ms")
                print("⚡" * 30 + "\n")

    except KeyboardInterrupt:
        print("[INFO] Aborted by user")
    finally:
        consumer.close()
        producer.flush()
        print("[INFO] Worker shutdown complete")

if __name__ == '__main__':
    time.sleep(12) # MySQL 헬스체크 대기 시간을 고려해 조금 넉넉히
    main()