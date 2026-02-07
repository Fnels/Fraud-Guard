import os
import json
import time
import datetime
import numpy as np
import pandas as pd
import pymysql
import redis
from confluent_kafka import Consumer, Producer
from catboost import CatBoostClassifier, Pool
from dotenv import load_dotenv
from pathlib import Path

# ---------------------------------------------------------------------------
# 0. 환경 설정 및 상수
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR.parent / 'Docker' / '.env'
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)

# Kafka & DB Config
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

# Model Paths
MODEL_PATH_TIER1 = '/app/data/ML/tier1model.cbm'
MODEL_PATH_TIER2 = '/app/data/ML/tier2model.cbm'

# Thresholds (사용자가 찾은 최적값 적용)
TH_TIER1 = 0.99816559  # Severe Fraud
TH_TIER2 = 0.56802705  # Probable Fraud (Recall 90% 타겟값으로 교체 권장)
# DW 수정완료 : 찾았던 올바른 threshold값을 입력하였습니다.

# ---------------------------------------------------------------------------
# 1. Feature Store (Redis + MySQL Handler)
# ---------------------------------------------------------------------------
class FeatureStore:
    def __init__(self):
        self.r = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)
        self.db_conn = None

    def get_db_connection(self):
        # 매번 연결/해제하지 않고 재사용 (Production에서는 Connection Pool 권장)
        if self.db_conn is None or not self.db_conn.open:
            self.db_conn = pymysql.connect(
                host=DB_HOST, user=DB_USER, password=DB_PASSWORD, db=DB_NAME,
                charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor,
                autocommit=True
            )
        return self.db_conn

    def _fetch_from_mysql(self, table, record_id):
        """Cache Miss 발생 시 MySQL에서 조회"""
        conn = self.get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute(f"SELECT * FROM {table} WHERE id = %s", (record_id,))
            return cursor.fetchone()

    def get_static_features(self, user_id, card_id, merchant_id):
        """User, Card, Merchant의 정적 정보를 Redis/DB에서 조회"""
        features = {}

        # 1. User Data
        user_key = f"info:user:{user_id}"
        user_data = self.r.get(user_key)
        if not user_data:
            user_data_db = self._fetch_from_mysql("users_data", user_id)
            if user_data_db:
                # 필요한 컬럼만 JSON으로 저장
                # yearly_income 전처리 ('$24,000' -> 24000.0)
                income = str(user_data_db.get('yearly_income', '0')).replace('$','').replace(',','')
                user_info = {
                    'yearly_income': float(income),
                    'current_age': user_data_db.get('current_age', 0),
                    'credit_score': user_data_db.get('credit_score', 0)
                }
                self.r.set(user_key, json.dumps(user_info))
                features.update(user_info)
            else:
                return None # User 없음 (무결성 실패)
        else:
            features.update(json.loads(user_data))

        # 2. Card Data
        card_key = f"info:card:{card_id}"
        card_data = self.r.get(card_key)
        if not card_data:
            card_data_db = self._fetch_from_mysql("cards_data", card_id)
            if card_data_db:
                limit = str(card_data_db.get('credit_limit', '0')).replace('$','').replace(',','')
                card_info = {
                    'credit_limit': float(limit),
                    'has_chip': card_data_db.get('has_chip', 'NO'),
                    'year_pin_last_changed': card_data_db.get('year_pin_last_changed', 2020),
                    'num_credit_cards': card_data_db.get('num_cards_issued', 1), # 대체 컬럼
                    'card_brand': card_data_db.get('card_brand', 'Unknown'),
                    'client_id': card_data_db.get('client_id') # 소유주 확인용
                }
                self.r.set(card_key, json.dumps(card_info))
                features.update(card_info)
            else:
                return None
        else:
            features.update(json.loads(card_data))

        # 3. Merchant Data
        merch_key = f"merchant:{merchant_id}" # redis_warmer 양식에 맞춤
        merch_data = self.r.get(merch_key)
        if not merch_data:
            merch_data_db = self._fetch_from_mysql("merchants_data", merchant_id)
            if merch_data_db:
                # redis_warmer는 통째로 넣었으므로 parsing
                self.r.set(merch_key, json.dumps(merch_data_db, default=str))
                features['mcc'] = str(merch_data_db.get('mcc', '0'))
                features['merchant_state'] = merch_data_db.get('merchant_state', 'Online')
                features['zip'] = str(merch_data_db.get('zip', '00000'))
            else:
                return None
        else:
            m_json = json.loads(merch_data)
            features['mcc'] = str(m_json.get('mcc', '0'))
            features['merchant_state'] = m_json.get('merchant_state', 'Online')
            features['zip'] = str(m_json.get('zip', '00000'))

        return features

    def calculate_velocity(self, client_id, amount, timestamp, card_id):
        """
        Redis ZSET을 이용한 실시간 Velocity 계산
        Key: history:client:{client_id}
        Score: timestamp
        Member: amount (중복 방지를 위해 timestamp:amount 조합 사용)
        """
        key = f"history:client:{client_id}"
        member = f"{timestamp}:{amount}" # Unique Member
        
        pipeline = self.r.pipeline()
        
        # 1. 현재 거래 추가
        pipeline.zadd(key, {member: timestamp})
        
        # 2. 24시간 지난 데이터 삭제 (Retention)
        cutoff_24h = timestamp - (24 * 3600)
        pipeline.zremrangebyscore(key, 0, cutoff_24h)
        
        # 3. 24시간 카운트 조회
        pipeline.zcard(key)
        
        # 4. 1시간 데이터 조회 (Sum 계산용)
        cutoff_1h = timestamp - 3600
        pipeline.zrangebyscore(key, cutoff_1h, '+inf')

        # 5. Last Transaction Time 조회 및 갱신
        last_time_key = f"last_tx:{card_id}"
        pipeline.get(last_time_key)
        pipeline.set(last_time_key, timestamp)
        
        results = pipeline.execute()
        
        # 결과 파싱
        count_24h = results[2] # zcard 결과
        one_hour_txs = results[3] # list of members "ts:amt"
        last_tx_ts = results[4] # get 결과
        
        # Sum 1h 계산
        sum_amt_1h = 0.0
        for m in one_hour_txs:
            try:
                _, amt = m.split(':')
                sum_amt_1h += float(amt)
            except: pass
            
        # Time Diff 계산
        time_diff = 999999.0
        if last_tx_ts:
            time_diff = float(timestamp) - float(last_tx_ts)
            if time_diff < 0: time_diff = 0.0

        return time_diff, float(count_24h), sum_amt_1h

# ---------------------------------------------------------------------------
# 2. ML Handler (Model Loading & Prediction)
# ---------------------------------------------------------------------------
class ModelHandler:
    def __init__(self):
        print("[INFO] Loading Tier 1 Model...")
        self.tier1 = CatBoostClassifier()
        self.tier1.load_model(MODEL_PATH_TIER1)
        
        print("[INFO] Loading Tier 2 Model...")
        self.tier2 = CatBoostClassifier()
        self.tier2.load_model(MODEL_PATH_TIER2)
        
        # 모델이 학습된 컬럼 순서 (반드시 일치해야 함!)
        self.feature_order = [
            'amount', 'utilization_ratio', 'amount_income_ratio', 
            'tech_mismatch', 'pin_years_gap', 'num_credit_cards', 
            'hour', 'is_night', 
            'time_diff_seconds', 'count_24h', 'sum_amt_1h',
            'current_age', 'credit_score',
            'mcc', 'merchant_state', 'zip', 'use_chip', 'card_brand'
        ]

    def predict(self, feature_dict):
        # Dict -> List (순서 보장)
        # 범주형 데이터는 String으로, 결측치는 적절한 값으로
        row = []
        for col in self.feature_order:
            val = feature_dict.get(col)
            if col in ['mcc', 'merchant_state', 'zip', 'use_chip', 'card_brand']:
                row.append(str(val) if val is not None else "Unknown")
            else:
                row.append(float(val) if val is not None else 0.0)
        
        # CatBoost Pool 생성 (1건이라도 Pool 권장)
        # cat_features 인덱스 지정 (뒤에서 5개)
        cat_indices = [13, 14, 15, 16, 17]
        
        # Tier 1 Prediction
        prob_t1 = self.tier1.predict_proba(row)[1]
        
        is_severe = 1 if prob_t1 >= TH_TIER1 else 0
        is_fraud = 0
        
        if is_severe:
            is_fraud = 1
        else:
            # Tier 2 Prediction
            prob_t2 = self.tier2.predict_proba(row)[1]
            if prob_t2 >= TH_TIER2:
                is_fraud = 1
        
        return is_severe, is_fraud

# ---------------------------------------------------------------------------
# 3. Main Logic
# ---------------------------------------------------------------------------
def main():
    store = FeatureStore()
    model_handler = ModelHandler()
    
    # Kafka Setup
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
                raw = json.loads(msg.value().decode('utf-8'))
                
                # 1. Integrity & Static Feature Fetching
                # user_id, card_id, merchant_id로 Redis 조회
                # 하나라도 없으면 DB 조회 후 캐싱 (Fail-over)
                static_feats = store.get_static_features(
                    raw['client_id'], raw['card_id'], raw['merchant_id']
                )
                
                # 무결성 검증 실패 시 (DB에도 없음)
                if (static_feats is None) or (raw['error'] != '-'):
                    raw['is_valid'] = 1
                    raw['is_fraud'] = 0
                    raw['is_severe_fraud'] = 0
                    # 바로 전송
                    producer.produce(TARGET_TOPIC, json.dumps(raw).encode('utf-8'))
                    continue

                # 2. Dynamic Feature Engineering
                # 시간 파싱
                dt_obj = pd.to_datetime(raw['order_time'])
                timestamp = dt_obj.timestamp()
                
                # Velocity 계산 (Redis ZSET)
                amount = float(raw['amount'])
                time_diff, count_24h, sum_1h = store.calculate_velocity(
                    raw['client_id'], amount, timestamp, raw['card_id']
                )
                
                # 파생 변수 계산
                # utilization_ratio
                util_ratio = amount / static_feats['credit_limit'] if static_feats['credit_limit'] > 0 else 0
                
                # amount_income_ratio
                income_ratio = amount / static_feats['yearly_income'] if static_feats['yearly_income'] > 0 else 0.0
                
                # tech_mismatch
                # use_chip은 raw data에는 없으므로(가정), 만약 raw에 있다면 그것 사용.
                # 예시 데이터에는 raw에 use_chip이 없었음. (보통 transaction에 포함됨)
                # 여기서는 raw에 'use_chip'이 있다고 가정 (없으면 Unknown)
                # DW 수정완료 use_chip도 raw에 들어오도록 수정함.
                use_chip = raw.get('use_chip', 'Unknown') # Raw에 있어야 함!
                has_chip = static_feats['has_chip']
                tech_mismatch = 1 if (has_chip == 'YES' and use_chip == 'Swipe Transaction') else 0
                
                # pin_years_gap
                pin_gap = 2020 - static_feats['year_pin_last_changed'] # 2020년 기준
                
                # 3. Final Feature Vector Construction
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
                
                # 4. Inference
                is_severe, is_fraud = model_handler.predict(features)
                
                # 5. Send Result
                raw['is_valid'] = 0
                raw['is_fraud'] = is_fraud
                raw['is_severe_fraud'] = is_severe
                
                producer.produce(TARGET_TOPIC, json.dumps(raw).encode('utf-8'))
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