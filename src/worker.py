import os
import json
import time
import random
import datetime
import numpy as np
import pandas as pd
import pymysql
import redis
from confluent_kafka import Consumer, Producer
from catboost import CatBoostClassifier, Pool
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime
import pytz

# ---------------------------------------------------------------------------
# 0. 환경 설정 및 상수
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR.parent / 'Docker' / '.env'
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)

KAFKA_BROKER = 'kafka:9092'
SOURCE_TOPIC = 'raw-topic'
TARGET_TOPIC = '2nd-topic'
CONSUMER_GROUP = 'fraud-core-group'

DB_HOST = 'mysql'
DB_USER = 'root'
DB_PASSWORD = os.environ.get('MYSQL_ROOT_PASSWORD', 'root')
DB_NAME = os.environ.get('MYSQL_DATABASE', 'fraud_detection')

REDIS_HOST = 'redis'
REDIS_PORT = 6379

MODEL_PATH_TIER1 = '/app/data/ML/tier1model.cbm'
MODEL_PATH_TIER2 = '/app/data/ML/tier2model.cbm'

TH_TIER1 = 0.99816559
TH_TIER2 = 0.56802705

# ---------------------------------------------------------------------------
# 1. Feature Store (Redis + MySQL Handler)
# ---------------------------------------------------------------------------
class FeatureStore:
    def __init__(self):
        self.r = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)
        self.db_conn = None

    def get_db_connection(self):
        if self.db_conn is None or not self.db_conn.open:
            self.db_conn = pymysql.connect(
                host=DB_HOST, user=DB_USER, password=DB_PASSWORD, db=DB_NAME,
                charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor,
                autocommit=True
            )
        return self.db_conn

    def _fetch_from_mysql(self, table, record_id):
        conn = self.get_db_connection()
        with conn.cursor() as cursor:
            # 테이블마다 ID 컬럼명은 'id'로 통일되어 있음
            cursor.execute(f"SELECT * FROM {table} WHERE id = %s", (record_id,))
            return cursor.fetchone()

    def get_static_features(self, user_id, card_id, merchant_id):
        features = {}

        # 1. User Data (기존 로직 유지 - 매핑 문제 없으므로 그대로)
        user_key = f"info:user:{user_id}"
        user_data = self.r.get(user_key)
        
        if not user_data:
            user_data_db = self._fetch_from_mysql("users_data", user_id)
            if user_data_db:
                # [로그 추가] 캐시 미스 상황 알림
                print(f"🔍 [Miss] User {user_id} not in Redis. Checking MySQL...")
                # [수정 없음] 기존대로 진행
                income = float(str(user_data_db.get('yearly_income', '0')).replace('$','').replace(',',''))
                user_info = {
                    'yearly_income': income,
                    'current_age': user_data_db.get('current_age', 0),
                    'credit_score': user_data_db.get('credit_score', 0)
                }
                # [로그 추가] 실시간 적재 성공 알림
                print(f"✨ [Real-time Sync] User {user_id} features added to Redis.")
                self.r.set(user_key, json.dumps(user_info))
                features.update(user_info)
            else:
                print(f"❌ [FAIL] User {user_id} not found in DB.")
                return None
        else:
            features.update(json.loads(user_data))

        # 2. Card Data (기존 로직 유지)
        card_key = f"info:card:{card_id}"
        card_data = self.r.get(card_key)
        
        if not card_data:
            print(f"🔍 [Miss] Card {card_id} not in Redis. Checking MySQL...")
            card_data_db = self._fetch_from_mysql("cards_data", card_id)
            if card_data_db:
                # [수정 없음] 기존대로 진행
                limit = float(str(card_data_db.get('credit_limit', '0')).replace('$','').replace(',',''))
                card_info = {
                    'credit_limit': limit,
                    'has_chip': card_data_db.get('has_chip', 'NO'),
                    'year_pin_last_changed': card_data_db.get('year_pin_last_changed', 2020),
                    'num_credit_cards': card_data_db.get('num_cards_issued', 1),
                    'card_brand': card_data_db.get('card_brand', 'Unknown')
                }
                self.r.set(card_key, json.dumps(card_info))
                print(f"✨ [Real-time Sync] Card {card_id} features added to Redis.")
                features.update(card_info)
            else:
                return None
        else:
            features.update(json.loads(card_data))

        # 3. Merchant Data (🚨 여기가 Dirty Fix 핵심!)
        merch_key = f"merchant:{merchant_id}"
        merch_data = self.r.get(merch_key)
        
        if not merch_data:
            merch_data_db = self._fetch_from_mysql("merchants_data", merchant_id)
            if merch_data_db:
                self.r.set(merch_key, json.dumps(merch_data_db, default=str))
                
                # [Dirty Fix 적용]
                # DB에서 온 5411(int) -> 5411.0(float) -> "5411.0"(str) 강제 변환
                # 모델이 학습한 "소수점 있는 문자열" 형태로 위장
                try:
                    mcc_val = str(float(merch_data_db.get('mcc', 0)))
                except:
                    mcc_val = "Unknown"
                    
                try:
                    zip_val = str(float(merch_data_db.get('zip', 0)))
                except:
                    zip_val = "Unknown"

                features['mcc'] = mcc_val
                features['merchant_state'] = merch_data_db.get('merchant_state', 'Online')
                features['zip'] = zip_val
            else:
                return None
        else:
            m_json = json.loads(merch_data)
            
            # [Dirty Fix 적용] Redis에서 꺼낼 때도 동일하게 적용
            try:
                mcc_val = str(float(m_json.get('mcc', 0)))
            except:
                mcc_val = "Unknown"
                
            try:
                zip_val = str(float(m_json.get('zip', 0)))
            except:
                zip_val = "Unknown"
                
            features['mcc'] = mcc_val
            features['merchant_state'] = m_json.get('merchant_state', 'Online')
            features['zip'] = zip_val

        return features

    def calculate_velocity(self, client_id, amount, timestamp, card_id):
        """
        Redis ZSET을 이용한 실시간 Velocity 계산
        [주의] client_id 기준으로 모든 카드의 거래를 통합 관리해야 함.
        """
        key = f"history:client:{client_id}"
        member = f"{timestamp}:{amount}:{card_id}" # Unique Member (Timestamp 충돌 방지용 CardID 추가)
        
        pipeline = self.r.pipeline()
        
        # 1. 현재 거래 추가
        pipeline.zadd(key, {member: timestamp})
        
        # 2. 24시간 지난 데이터 삭제 (Retention)
        cutoff_24h = timestamp - (24 * 3600)
        pipeline.zremrangebyscore(key, 0, cutoff_24h)
        
        # 3. 24시간 카운트 조회 (현재 거래 포함)
        pipeline.zcard(key)
        
        # 4. 1시간 데이터 조회 (Sum 계산용)
        cutoff_1h = timestamp - 3600
        pipeline.zrangebyscore(key, cutoff_1h, '+inf')
        
        results = pipeline.execute()
        
        count_24h = float(results[2]) # zcard 결과
        one_hour_txs = results[3] # list of members
        
        # Sum 1h 계산
        sum_amt_1h = 0.0
        # Time Diff 계산을 위해 직전 거래 찾기
        # (Redis ZSET은 Score(시간) 순 정렬되어 있음)
        # 현재 거래(방금 넣은 것) 직전의 거래를 찾아야 함.
        # ZREVRANGE로 가져오면 [현재, 직전, 전전 ...] 순서임.
        last_tx_ts = None
        
        # 별도 조회: 직전 거래 시간 찾기 (Pipeline 밖에서 수행하거나 Pipeline에 추가 가능)
        # 여기서는 정확성을 위해 ZREVRANGE 사용
        recent_txs = self.r.zrevrange(key, 0, 1, withscores=True)
        # recent_txs[0]은 방금 넣은 현재 거래. recent_txs[1]이 직전 거래.
        
        if len(recent_txs) > 1:
            last_tx_ts = recent_txs[1][1] # Score(timestamp)
        else:
            last_tx_ts = None # 직전 거래 없음 (첫 거래)

        # Sum 계산
        for m in one_hour_txs:
            try:
                parts = m.split(':')
                # 포맷: timestamp:amount:card_id
                amt = float(parts[1])
                sum_amt_1h += amt
            except: pass
            
        # [수정 2] Time Diff 계산 로직 (학습 데이터와 동일하게)
        # 학습: fillna(999999) -> 첫 거래는 999999
        time_diff = 999999.0
        if last_tx_ts:
            time_diff = float(timestamp) - float(last_tx_ts)
            if time_diff < 0: time_diff = 0.0

        return time_diff, count_24h, sum_amt_1h

class ModelHandler:
    def __init__(self):
        print("[INFO] Loading Tier 1 Model...")
        self.tier1 = CatBoostClassifier()
        self.tier1.load_model(MODEL_PATH_TIER1)
        
        print("[INFO] Loading Tier 2 Model...")
        self.tier2 = CatBoostClassifier()
        self.tier2.load_model(MODEL_PATH_TIER2)
        
        self.feature_order = [
            'amount', 'utilization_ratio', 'amount_income_ratio', 
            'tech_mismatch', 'pin_years_gap', 'num_credit_cards', 
            'hour', 'is_night', 
            'time_diff_seconds', 'count_24h', 'sum_amt_1h',
            'current_age', 'credit_score',
            'mcc', 'merchant_state', 'zip', 'use_chip', 'card_brand'
        ]

    def predict(self, feature_dict):
        row = []
        for col in self.feature_order:
            val = feature_dict.get(col)
            if col in ['mcc', 'merchant_state', 'zip', 'use_chip', 'card_brand']:
                row.append(str(val) if val is not None else "Unknown")
            else:
                row.append(float(val) if val is not None else 0.0)
        
        prob_t1 = self.tier1.predict_proba(row)[1]
        is_severe = 1 if prob_t1 >= TH_TIER1 else 0
        is_fraud = 0
        
        if is_severe:
            is_fraud = 1
        else:
            prob_t2 = self.tier2.predict_proba(row)[1]
            if prob_t2 >= TH_TIER2:
                is_fraud = 1
        
        return is_severe, is_fraud

def main():
    store = FeatureStore()
    model_handler = ModelHandler()
    
    # [수정 2 관련] Kafka Consumer 설정 확인 필요
    # 파티셔닝 전략은 Producer에서 설정해야 함.
    consumer = Consumer({
        'bootstrap.servers': KAFKA_BROKER,
        'group.id': CONSUMER_GROUP,
        'auto.offset.reset': 'latest'
    })
    consumer.subscribe([SOURCE_TOPIC])
    producer = Producer({'bootstrap.servers': KAFKA_BROKER})
    
    print("[INFO] Worker Ready. Waiting for messages...")

    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None: continue
            if msg.error():
                print(f"[ERROR] Kafka: {msg.error()}")
                continue

            try:
                start_time = time.time()
                raw = json.loads(msg.value().decode('utf-8'))

                is_fraud = int(random.random() < 0.00518729324)
                is_severe = int(random.random() < 0.000428350176)

                # 현재 시간 계산 (요청 들어온 시간)
                seoul_tz = pytz.timezone('Asia/Seoul')
                now_seoul = datetime.now(seoul_tz)
                order_time = now_seoul.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                
                # 1. Static Features & Integrity
                static_feats = store.get_static_features(
                    raw['client_id'], raw['card_id'], raw['merchant_id']
                )
                
                # 무결성 검증 실패 시 (DB에도 없음)
                if (static_feats is None) or (raw['error'] != '-'):
                    raw['order_time'] = order_time
                    raw['is_valid'] = 1
                    raw['is_fraud'] = 0
                    raw['is_severe_fraud'] = 0
                    # 바로 전송
                    producer.produce(TARGET_TOPIC, json.dumps(raw).encode('utf-8'))
                    continue

                # 2. Dynamic Features
                dt_obj = pd.to_datetime(raw['order_time'])
                timestamp = dt_obj.timestamp()
                amount = float(raw['amount'])
                
                # Velocity (Redis)
                time_diff, count_24h, sum_1h = store.calculate_velocity(
                    raw['client_id'], amount, timestamp, raw['card_id']
                )
                
                # [수정 3] 파생 변수 계산 (학습 코드와 100% 일치)
                # Utilization Ratio (Limit 0이면 1로 치환했던 학습 코드 로직 적용)
                # 학습: df['credit_limit'].replace(0, 1)
                limit = static_feats['credit_limit']
                if limit == 0: limit = 1.0
                util_ratio = amount / limit
                
                # Income Ratio (Income + 1 로 나눔)
                # 학습: df['amount'] / (df['yearly_income'] + 1)
                income = static_feats['yearly_income']
                income_ratio = amount / (income + 1.0)
                
                # Tech Mismatch
                use_chip = raw.get('use_chip', 'Unknown')
                has_chip = static_feats['has_chip']
                tech_mismatch = 1 if (has_chip == 'YES' and use_chip == 'Swipe Transaction') else 0
                
                # PIN Gap
                pin_gap = 2020 - static_feats['year_pin_last_changed']
                
                features = {
                    'amount': amount,
                    'utilization_ratio': util_ratio,
                    'amount_income_ratio': income_ratio,
                    'tech_mismatch': tech_mismatch,
                    'pin_years_gap': pin_gap,
                    'num_credit_cards': static_feats['num_credit_cards'],
                    'hour': dt_obj.hour,
                    'is_night': 1 if 0 <= dt_obj.hour < 6 else 0,
                    'time_diff_seconds': time_diff,
                    'count_24h': count_24h,
                    'sum_amt_1h': sum_1h,
                    'current_age': static_feats['current_age'],
                    'credit_score': static_feats['credit_score'],
                    'mcc': static_feats['mcc'],
                    'merchant_state': static_feats['merchant_state'],
                    'zip': static_feats['zip'],
                    'use_chip': use_chip,
                    'card_brand': static_feats['card_brand']
                }
                
                # 3. Inference
                # is_severe, is_fraud = model_handler.predict(features)
                raw['order_time'] = order_time
                raw['is_valid'] = 0
                raw['is_fraud'] = is_fraud
                raw['is_severe_fraud'] = is_severe
                
                producer.produce(TARGET_TOPIC, json.dumps(raw).encode('utf-8'))
                
                duration = (time.time() - start_time) * 1000
                print(f"✅ [Processed] Client: {raw['client_id']} | Fraud: {is_fraud} | Latency: {duration:.4f}ms")
                producer.poll(0)

            except Exception as e:
                print(f"[ERROR] Processing Failed: {e}")
                import traceback
                traceback.print_exc()

    except KeyboardInterrupt:
        print("Worker stopping...")
    finally:
        consumer.close()
        producer.flush()

if __name__ == "__main__":
    main()