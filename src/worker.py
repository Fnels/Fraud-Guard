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

    # [Safe Helper] 달러, 콤마 제거 및 안전한 float 변환
    def clean_dollar(self, x):
        if x is None: return 0.0
        s = str(x).replace('$', '').replace(',', '')
        try:
            return float(s)
        except:
            return 0.0

    def get_static_features(self, user_id, card_id, merchant_id):
        features = {}

        # 1. User Data (기존 로직 유지)
        user_key = f"info:user:{user_id}"
        user_data = self.r.get(user_key)
        
        if not user_data:
            user_data_db = self._fetch_from_mysql("users_data", user_id)
            if user_data_db:
                # [로그 추가] 캐시 미스 상황 알림
                print(f"🔍 [Miss] User {user_id} not in Redis. Checking MySQL...")
                
                # [수정] clean_dollar 적용 및 int 명시
                income = self.clean_dollar(user_data_db.get('yearly_income'))
                user_info = {
                    'yearly_income': income,
                    'current_age': int(user_data_db.get('current_age', 0)),
                    'credit_score': int(user_data_db.get('credit_score', 0))
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
                # [수정] clean_dollar 적용 및 int 명시
                limit = self.clean_dollar(card_data_db.get('credit_limit'))
                card_info = {
                    'credit_limit': limit,
                    'has_chip': card_data_db.get('has_chip', 'NO'),
                    'year_pin_last_changed': int(card_data_db.get('year_pin_last_changed', 2020)),
                    'num_credit_cards': int(card_data_db.get('num_cards_issued', 1)),
                    'card_brand': card_data_db.get('card_brand', 'Unknown')
                }
                self.r.set(card_key, json.dumps(card_info))
                print(f"✨ [Real-time Sync] Card {card_id} features added to Redis.")
                features.update(card_info)
            else:
                return None
        else:
            features.update(json.loads(card_data))

        # 3. Merchant Data (★ 핵심 수정: 포맷 통일)
        merch_key = f"merchant:{merchant_id}"
        merch_data = self.r.get(merch_key)
        
        # Redis/DB 데이터 로드 로직
        m_data = None
        if not merch_data:
            merch_data_db = self._fetch_from_mysql("merchants_data", merchant_id)
            if merch_data_db:
                # Redis엔 원본 저장 (기존 로직 유지)
                self.r.set(merch_key, json.dumps(merch_data_db, default=str))
                m_data = merch_data_db
            else:
                return None
        else:
            m_data = json.loads(merch_data)
        print(m_data)

        # [Dirty Fix 수정 -> Training Format 일치화]
        # MCC: Training is "5300" (Stringified Int) -> DB 5300.0/5300 -> "5300"
        raw_mcc = m_data.get('mcc')
        try:
            if raw_mcc is None or str(raw_mcc).lower() == 'nan':
                mcc_val = "Unknown"
            else:
                # float -> int -> str (소수점 제거)
                mcc_val = str(int(float(str(raw_mcc))))
        except:
            mcc_val = "Unknown"

        # Zip: Training is "4105.0" (Stringified Float) -> DB 4105 -> "4105.0"
        raw_zip = m_data.get('zip')
        try:
            if raw_zip is None or str(raw_zip).lower() == 'nan':
                zip_val = "Unknown"
            else:
                # float -> str (소수점 유지)
                zip_val = str(float(str(raw_zip)))
        except:
            zip_val = "Unknown"
            
        # Merchant State
        raw_state = m_data.get('merchant_state')
        if raw_state is None or str(raw_state).lower() == 'nan':
            state_val = "Online"
        else:
            state_val = str(raw_state)

        features['mcc'] = mcc_val
        features['merchant_state'] = state_val
        features['zip'] = zip_val

        return features

    def calculate_velocity(self, client_id, amount, timestamp, card_id):
        """
        Redis ZSET을 이용한 실시간 Velocity 계산
        """
        key = f"history:client:{client_id}"
        member = f"{timestamp}:{amount}:{card_id}" # Unique Member
        
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
        
        results = pipeline.execute()
        
        count_24h = float(results[2]) # zcard 결과
        one_hour_txs = results[3] # list of members
        
        # Sum 1h 계산
        sum_amt_1h = 0.0
        # Time Diff 계산을 위해 직전 거래 찾기
        last_tx_ts = None
        
        # 별도 조회: 직전 거래 시간 찾기
        recent_txs = self.r.zrevrange(key, 0, 1, withscores=True)
        if len(recent_txs) > 1:
            last_tx_ts = recent_txs[1][1] # Score(timestamp)

        for m in one_hour_txs:
            try:
                parts = m.split(':')
                amt = float(parts[1])
                sum_amt_1h += amt
            except: pass
            
        # Time Diff 계산
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
        
        # [수정] Int형으로 처리해야 할 컬럼 목록 정의
        self.int_columns = [
            'tech_mismatch', 'pin_years_gap', 'num_credit_cards', 
            'hour', 'is_night', 'current_age', 'credit_score'
        ]

    def predict(self, feature_dict):
        row = []
        for col in self.feature_order:
            val = feature_dict.get(col)
            
            # 1. 범주형 (String)
            if col in ['mcc', 'merchant_state', 'zip', 'use_chip', 'card_brand']:
                row.append(str(val) if val is not None else "Unknown")
            # 2. 정수형 (Int) - [수정] float으로 변환되는 것 방지
            elif col in self.int_columns:
                row.append(int(val) if val is not None else 0)
            # 3. 실수형 (Float)
            else:
                row.append(float(val) if val is not None else 0.0)
        
        # [로그] 모델 입력값 확인
        print("\n" + "="*50)
        print(f"🕵️ [Live Debug] Worker Input Vector (Client ID: {feature_dict.get('client_id', 'Unknown')})")
        print("="*50)
        debug_view = dict(zip(self.feature_order, row))
        print(json.dumps(debug_view, indent=4, default=str))
        print("="*50 + "\n")

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

                # 현재 시간 계산 (요청 들어온 시간)
                seoul_tz = pytz.timezone('Asia/Seoul')
                now_seoul = datetime.now(seoul_tz)
                order_time = now_seoul.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                
                # 1. Static Features & Integrity
                static_feats = store.get_static_features(
                    raw['client_id'], raw['card_id'], raw['merchant_id']
                )
                
                # 무결성 검증 실패 시 (DB에도 없음) or Error Check
                if (static_feats is None) or (raw['error'] != '-'):
                    raw['order_time'] = order_time
                    raw['is_valid'] = 1 # 1 = Invalid (Team Convention)
                    raw['is_fraud'] = 0
                    raw['is_severe_fraud'] = 0
                    producer.produce(TARGET_TOPIC, json.dumps(raw).encode('utf-8'))
                    continue

                # 2. Dynamic Features
                dt_obj = pd.to_datetime(raw['order_time'])
                timestamp = dt_obj.timestamp()
                
                # [수정 1] Amount 절대값 처리 (Critical Fix)
                amount = abs(float(raw['amount']))
                
                # Velocity (Redis)
                time_diff, count_24h, sum_amt_1h = store.calculate_velocity(
                    raw['client_id'], amount, timestamp, raw['card_id']
                )
                
                # [수정 2] 파생 변수 계산 & Type Casting
                # Utilization Ratio
                limit = static_feats['credit_limit']
                if limit == 0: limit = 1.0
                util_ratio = amount / limit
                
                # Income Ratio
                income = static_feats['yearly_income']
                income_ratio = amount / (income + 1.0)
                
                # Tech Mismatch
                use_chip = raw.get('use_chip', 'Unknown')
                has_chip = static_feats['has_chip']
                tech_mismatch = 1 if (has_chip == 'YES' and use_chip == 'Swipe Transaction') else 0
                
                # PIN Gap
                pin_gap = 2020 - static_feats['year_pin_last_changed']
                
                # Feature Vector Construction (Type 명시 적용)
                features = {
                    'amount': float(amount),
                    'utilization_ratio': float(util_ratio),
                    'amount_income_ratio': float(income_ratio),
                    'tech_mismatch': int(tech_mismatch),
                    'pin_years_gap': int(pin_gap),
                    'num_credit_cards': int(static_feats['num_credit_cards']),
                    'hour': int(dt_obj.hour),
                    'is_night': int(1 if 0 <= dt_obj.hour < 6 else 0),
                    'time_diff_seconds': float(time_diff),
                    'count_24h': float(count_24h),
                    'sum_amt_1h': float(sum_amt_1h),
                    'current_age': int(static_feats['current_age']),
                    'credit_score': int(static_feats['credit_score']),
                    'mcc': static_feats['mcc'], # String
                    'merchant_state': static_feats['merchant_state'], # String
                    'zip': static_feats['zip'], # String
                    'use_chip': use_chip, # String
                    'card_brand': static_feats['card_brand'], # String
                    'client_id': raw['client_id'] # Debugging용
                }
                
                # 3. Inference
                is_severe, is_fraud = model_handler.predict(features)
                raw['order_time'] = order_time
                raw['is_valid'] = 0 # 0 = Valid (Team Convention)
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