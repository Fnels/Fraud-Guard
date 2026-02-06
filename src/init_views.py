import os
import time
import pymysql
from dotenv import load_dotenv
from pathlib import Path

# ---------------------------------------------------------------------------
# 0) Load .env
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR.parent / "Docker" / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)

DB_HOST = os.getenv("MYSQL_HOST", "mysql")
DB_USER = os.getenv("MYSQL_USER", "root")
DB_PASSWORD = os.getenv("MYSQL_ROOT_PASSWORD", "root")
DB_NAME = os.getenv("MYSQL_DATABASE", "fraud_guard")


# ---------------------------------------------------------------------------
# 1) DB Wait / Connect Utils
# ---------------------------------------------------------------------------
def wait_for_db(retries: int = 30, sleep_sec: int = 2) -> None:
    """MySQL 서버가 쿼리를 받을 준비가 될 때까지 대기"""
    for i in range(retries):
        try:
            conn = pymysql.connect(
                host=DB_HOST,
                user=DB_USER,
                password=DB_PASSWORD,
                charset="utf8mb4",
            )
            conn.close()
            print("✅ MySQL is ready!")
            return
        except pymysql.MySQLError as e:
            left = retries - i - 1
            print(f"⏳ Waiting for MySQL... ({left} retries left) | {e}")
            time.sleep(sleep_sec)
    raise RuntimeError("❌ MySQL Connection Timed Out.")


def get_conn():
    return pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        db=DB_NAME,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
    )


def exists_table_or_view(conn, name: str) -> bool:
    sql = """
    SELECT COUNT(*) AS cnt
    FROM information_schema.tables
    WHERE table_schema = %s AND table_name = %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (DB_NAME, name))
        return cur.fetchone()["cnt"] > 0


# ---------------------------------------------------------------------------
# 2) DDL (Agg tables / Views)
# ---------------------------------------------------------------------------
AGG_TABLE_DDL = {
    "agg_txn_1m": """
    CREATE TABLE IF NOT EXISTS agg_txn_1m (
        window_start DATETIME(3) NOT NULL,
        window_end   DATETIME(3) NOT NULL,
        merchant_id  INT NOT NULL,
        txn_cnt     BIGINT NOT NULL,
        amount_sum  DECIMAL(18,2) NOT NULL,
        fraud_cnt   BIGINT NOT NULL,
        valid_cnt   BIGINT NOT NULL,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                   ON UPDATE CURRENT_TIMESTAMP,
        PRIMARY KEY (window_start, window_end, merchant_id)
    );
    """,
    "agg_txn_1h": """
    CREATE TABLE IF NOT EXISTS agg_txn_1h (
        window_start DATETIME(3) NOT NULL,
        window_end   DATETIME(3) NOT NULL,
        merchant_id  INT NOT NULL,
        txn_cnt     BIGINT NOT NULL,
        amount_sum  DECIMAL(18,2) NOT NULL,
        fraud_cnt   BIGINT NOT NULL,
        valid_cnt   BIGINT NOT NULL,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                   ON UPDATE CURRENT_TIMESTAMP,
        PRIMARY KEY (window_start, window_end, merchant_id)
    );
    """,
    "agg_txn_1d": """
    CREATE TABLE IF NOT EXISTS agg_txn_1d (
        window_start DATETIME(3) NOT NULL,
        window_end   DATETIME(3) NOT NULL,
        merchant_id  INT NOT NULL,
        txn_cnt     BIGINT NOT NULL,
        amount_sum  DECIMAL(18,2) NOT NULL,
        fraud_cnt   BIGINT NOT NULL,
        valid_cnt   BIGINT NOT NULL,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                   ON UPDATE CURRENT_TIMESTAMP,
        PRIMARY KEY (window_start, window_end, merchant_id)
    );
    """,
}

VW_AGG_SQL = {
    "vw_agg_1m": "CREATE OR REPLACE VIEW vw_agg_1m AS SELECT * FROM agg_txn_1m;",
    "vw_agg_1h": "CREATE OR REPLACE VIEW vw_agg_1h AS SELECT * FROM agg_txn_1h;",
    "vw_agg_1d": "CREATE OR REPLACE VIEW vw_agg_1d AS SELECT * FROM agg_txn_1d;",
}


VW_CUSTOMER_MARKETING_SQL = """
CREATE OR REPLACE VIEW vw_customer_marketing AS
SELECT
  u.id AS client_id,
  u.current_age,
  CASE
    WHEN u.current_age < 20 THEN '<20'
    WHEN u.current_age < 30 THEN '20s'
    WHEN u.current_age < 40 THEN '30s'
    WHEN u.current_age < 50 THEN '40s'
    WHEN u.current_age < 60 THEN '50s'
    ELSE '60+'
  END AS age_group,
  u.gender,
  u.address,
  u.latitude,
  u.longitude,
  u.per_capita_income,
  u.yearly_income,
  u.total_debt,
  CASE
    WHEN u.yearly_income IS NULL OR u.yearly_income = 0 THEN NULL
    ELSE ROUND(u.total_debt / u.yearly_income, 5)
  END AS dti_ratio,
  u.credit_score,
  u.num_credit_cards,

  -- ✅ Geo 기반 주(state)
  usm.state_code,
  usm.state_name,

  -- 카드(유저당 여러 장일 수 있으니 LEFT JOIN)
  c.id AS card_id,
  c.card_brand,
  c.card_type,
  c.credit_limit
FROM users_data u
LEFT JOIN user_state_map usm
  ON usm.client_id = u.id
LEFT JOIN cards_data c
  ON c.client_id = u.id;
"""


# ---------------------------------------------------------------------------
# 3) Main
# ---------------------------------------------------------------------------
def main():
    print("--- View Initializer Started ---")
    wait_for_db()

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            # 1) agg tables ensure
            for name, ddl in AGG_TABLE_DDL.items():
                cur.execute(ddl)
            print("✅ agg_txn tables ensured.")

            # 2) vw_agg ensure
            for name, sql in VW_AGG_SQL.items():
                cur.execute(sql)
            print("✅ vw_agg views ensured.")

            # 3) vw_customer_marketing ensure
            #    (필수 테이블 없으면 에러나므로 사전 체크)
            required = ["users_data", "cards_data", "user_state_map"]
            missing = [t for t in required if not exists_table_or_view(conn, t)]
            if missing:
                # 여기서 실패시키는 대신, 로그만 남기고 넘어가고 싶으면 continue 처리 가능
                raise RuntimeError(f"❌ Missing required table/view(s): {missing}. "
                                   f"init_geo.py가 먼저 실행되어 user_state_map이 생성되어야 합니다.")

            cur.execute(VW_CUSTOMER_MARKETING_SQL)
            print("✅ vw_customer_marketing ensured.")

        print("--- View Initializer Completed ---")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
