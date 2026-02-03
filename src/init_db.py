import os
import csv
import pymysql
from dotenv import load_dotenv
from pathlib import Path

# .env 로드
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR.parent / 'Docker' / '.env'

if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)

# 설정
DB_HOST = 'mysql'
DB_USER = 'root'
DB_PASSWORD = os.environ.get('MYSQL_ROOT_PASSWORD', 'root')
DB_NAME = os.environ.get('MYSQL_DATABASE', 'fraud_detection')
DATA_DIR = Path('/app/data/origin') 

def get_connection():
    return pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )

def create_database_and_tables():
    """DB 및 테이블 스키마 생성 (DDL)"""
    conn = pymysql.connect(host=DB_HOST, user=DB_USER, password=DB_PASSWORD)
    try:
        with conn.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
            cursor.execute(f"USE {DB_NAME}")
            
            # 1. Users Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users_data (
                    id INT PRIMARY KEY,
                    current_age INT,
                    retirement_age INT,
                    birth_year INT,
                    birth_month INT,
                    gender VARCHAR(10),
                    address VARCHAR(255),
                    latitude FLOAT,
                    longitude FLOAT,
                    per_capita_income INT,   -- $ 제거 후 저장됨
                    yearly_income INT,       -- $ 제거 후 저장됨
                    total_debt INT,          -- $ 제거 후 저장됨
                    credit_score INT,
                    num_credit_cards INT
                )
            """)

            # 2. Merchants Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS merchants_data (
                    id INT PRIMARY KEY,
                    business_name VARCHAR(255),
                    business_type VARCHAR(100),
                    address VARCHAR(255),
                    latitude FLOAT,
                    longitude FLOAT
                )
            """)

            # 3. Cards Table (컬럼 추가됨: card_on_dark_web)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cards_data (
                    id INT PRIMARY KEY,
                    client_id INT,
                    card_brand VARCHAR(50),
                    card_type VARCHAR(50),
                    card_number INT,
                    expires VARCHAR(10),
                    cvv INT,
                    has_chip VARCHAR(10),
                    num_cards_issued INT,
                    credit_limit FLOAT,      -- $ 제거 후 저장됨
                    acct_open_date VARCHAR(20),
                    year_pin_last_changed INT,
                    card_on_dark_web VARCHAR(10), -- [추가된 컬럼]
                    FOREIGN KEY (client_id) REFERENCES users_data(id)
                )
            """)
            
        conn.commit()
        print("[INIT] Schema creation checked/completed.")
    finally:
        conn.close()

def clean_currency(value):
    """$ 기호 제거 및 빈 문자열 처리"""
    if value is None:
        return None
    if isinstance(value, str):
        # $ 기호 제거
        clean_val = value.replace('$', '')
        # 빈 문자열이면 None 반환 (DB에서 NULL로 처리되도록)
        if clean_val.strip() == '':
            return None
        return clean_val
    return value

def load_csv(table_name, file_name):
    """CSV 파일을 읽어 DB에 적재"""
    file_path = DATA_DIR / file_name
    
    if not file_path.exists():
        print(f"[WARN] File not found: {file_path}. Skipping {table_name} load.")
        return
    
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(f"USE {DB_NAME}")
            
            # 데이터 존재 여부 확인
            cursor.execute(f"SELECT COUNT(*) as cnt FROM {table_name}")
            result = cursor.fetchone()
            
            if result['cnt'] > 0:
                print(f"[SKIP] Table '{table_name}' already has {result['cnt']} rows.")
                return

            print(f"[LOAD] Loading {file_name} into {table_name}...")
            
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
                if not rows:
                    return

                columns = rows[0].keys()
                cols_str = ', '.join(columns)
                vals_str = ', '.join(['%s'] * len(columns))
                
                sql = f"INSERT INTO {table_name} ({cols_str}) VALUES ({vals_str})"
                
                batch_size = 1000
                data_tuples = []
                for row in rows:
                    # [핵심 수정] 모든 값에 대해 $ 기호 제거 등 클리닝 수행
                    cleaned_row = [clean_currency(row[col]) for col in columns]
                    data_tuples.append(tuple(cleaned_row))
                
                for i in range(0, len(data_tuples), batch_size):
                    batch = data_tuples[i:i + batch_size]
                    cursor.executemany(sql, batch)
                    conn.commit()
                    print(f"   - Inserted {len(batch)} rows...")
            
            print(f"[SUCCESS] Loaded {len(rows)} rows into {table_name}.")
            
    except Exception as e:
        print(f"[ERROR] Failed to load {table_name}: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    print("--- DB Initializer Started ---")
    
    # 1. DB/Table 생성
    create_database_and_tables()
    
    # 2. CSV 적재
    load_csv('users_data', 'users_data.csv')
    load_csv('merchants_data', 'merchants_data.csv')
    load_csv('cards_data', 'cards_data.csv')
    
    print("--- DB Initializer Completed ---")