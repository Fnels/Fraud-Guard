# consumer group 1 의 역할을 수행합니다.

import os
import json
import pymysql
import math
import time
from kafka import KafkaConsumer
from pathlib import Path

# 환경 변수 설정 (docker-compose와 연동)
KAFKA_BOOTSTRAP = os.getenv('KAFKA_BOOTSTRAP', 'kafka:9092')
KAFKA_TOPIC = '2nd-topic'
MYSQL_HOST = 'mysql'
MYSQL_USER = 'root'
MYSQL_PASSWORD = os.getenv('MYSQL_ROOT_PASSWORD', 'root')
MYSQL_DB = 'fraud_guard'

def get_db_connection():
    max_retries = 10
    for i in range(max_retries):
        try:
            conn = pymysql.connect(
                host=MYSQL_HOST,
                user=MYSQL_USER,
                password=MYSQL_PASSWORD,
                db=MYSQL_DB,
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor
            )
            print("✅ DB 연결 성공!")
            return conn
        except pymysql.err.OperationalError as e:
            print(f"⚠️ DB 연결 실패 ({i+1}/{max_retries}): {e}")
            print(f"   현재 설정 - Host: {MYSQL_HOST}, User: {MYSQL_USER}, PW: {'***' if MYSQL_PASSWORD else 'None'}")
            time.sleep(5)  # 5초 쉬고 다시 시도

    raise Exception("❌ DB 연결 시도 횟수 초과!")

def main():
    # 1. 카프카 컨슈머 설정
    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=[KAFKA_BOOTSTRAP],
        auto_offset_reset='earliest', # 처음부터 읽기
        group_id='consumer-group-1',  # 컨슈머 그룹 지정
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))# JSON 역직렬화 설정
    )

    print(f"📥 {KAFKA_TOPIC} 모니터링 시작 및 DB 적재 대기 중...")

    while True:  # 👈 추가: 프로그램이 종료되지 않게 무한 루프
        conn = None

        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            # 1. Transactions Table (최종 적재용)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS transactions_data (
                    id INT PRIMARY KEY,
                    order_id INT,
                    order_time DATETIME(3),
                    client_id INT,
                    card_id INT,
                    merchant_id INT,
                    amount DECIMAL(10, 2),
                    error VARCHAR(100),
                    is_valid BOOLEAN,
                    is_fraud BOOLEAN,
                    is_severe_fraud BOOLEAN,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (client_id) REFERENCES users_data(id),
                    FOREIGN KEY (card_id) REFERENCES cards_data(id),
                    FOREIGN KEY (merchant_id) REFERENCES merchants_data(id)
                )
            """)
            # 2. 메시지 소비 및 적재 루프
            for message in consumer:
                data = message.value
                error_val = data.get('error')
                if error_val is None or (isinstance(error_val, float) and math.isnan(error_val)):
                    # error가 비어있으면(NaN/None) 보통 에러 없음(success)을 의미
                    # 만약 NULL로 넣고 싶다면 None으로 설정하세요.
                    error_val = 'success'
                
                # 2. DB Insert 쿼리
                sql = """
                    INSERT INTO transactions_data (id, order_id, order_time, client_id, card_id, merchant_id, amount, error, is_valid, is_fraud, is_severe_fraud)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE processed_at = CURRENT_TIMESTAMP
                """
                val = (
                    data['id'], 
                    data.get('order_id', 0),# 없을 경우 0 넣기 
                    data['order_time'], 
                    data['client_id'], 
                    data['card_id'], 
                    data['merchant_id'], 
                    data['amount'],
                    error_val,
                    data['is_valid'],
                    data['is_fraud'],
                    data['is_severe_fraud']
                )

                cursor.execute(sql, val)
                conn.commit()
                
                print(f"✅ [DB 저장 완료] ID: {data['id']} | Time: {data['order_time']}")

        except Exception as e:
            print(f"❌ 에러 발생: {e}")
        finally:
            cursor.close()
            conn.close()

if __name__ == "__main__":
    main()