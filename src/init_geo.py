import os
import json
import time
import pymysql
from pathlib import Path
from dotenv import load_dotenv

# .env 로드 (너 프로젝트 구조 기준)
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR.parent / "Docker" / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)

DB_HOST = os.getenv("MYSQL_HOST", "mysql")
DB_USER = os.getenv("MYSQL_USER", "root")
DB_PASSWORD = os.environ.get("MYSQL_ROOT_PASSWORD", "root")
DB_NAME = os.environ.get("MYSQL_DATABASE", "fraud_guard")

GEOJSON_PATH = Path("/app/data/geo/us_states.geojson")  # 너가 만든 파일 경로

def wait_for_db():
    retries = 30
    while retries > 0:
        try:
            conn = pymysql.connect(host=DB_HOST, user=DB_USER, password=DB_PASSWORD, charset="utf8mb4")
            conn.close()
            print("✅ MySQL is ready!")
            return
        except pymysql.MySQLError as e:
            print(f"⏳ Waiting for MySQL... ({retries} left) / {e}")
            time.sleep(2)
            retries -= 1
    raise RuntimeError("❌ MySQL Connection Timed Out")

def get_conn():
    return pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        db=DB_NAME,
        charset="utf8mb4",
        autocommit=True,
        cursorclass=pymysql.cursors.DictCursor,
    )

def ensure_dim_us_state():
    ddl = """
    CREATE TABLE IF NOT EXISTS dim_us_state (
      stusps  CHAR(2) PRIMARY KEY,
      name    VARCHAR(64) NOT NULL,
      statefp CHAR(2) NULL,
      geom    GEOMETRY NOT NULL SRID 0,
      SPATIAL INDEX idx_state_geom (geom)
    ) ENGINE=InnoDB;
    """

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(ddl)
    finally:
        conn.close()

def load_states_geojson_if_empty():
    if not GEOJSON_PATH.exists():
        raise FileNotFoundError(f"❌ GeoJSON not found: {GEOJSON_PATH}")

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS cnt FROM dim_us_state;")
            cnt = cur.fetchone()["cnt"]
            if cnt > 0:
                print(f"[SKIP] dim_us_state already has {cnt} rows.")
                return

            print(f"[LOAD] Loading states from {GEOJSON_PATH} ...")
            data = json.loads(GEOJSON_PATH.read_text(encoding="utf-8"))
            features = data.get("features", [])

            sql = """
            INSERT INTO dim_us_state (stusps, name, statefp, geom)
            VALUES (%s, %s, %s, ST_GeomFromGeoJSON(%s, 1, 0))
            ON DUPLICATE KEY UPDATE
                name=VALUES(name),
                statefp=VALUES(statefp),
                geom=VALUES(geom);
            """
            rows = []
            for f in features:
                props = f.get("properties", {}) or {}
                geom = f.get("geometry", {}) or {}
                stusps = props.get("STUSPS")
                name = props.get("NAME")
                statefp = props.get("STATEFP")

                if not stusps or not name or not geom:
                    continue

                rows.append((stusps, name, statefp, json.dumps(geom)))

            cur.executemany(sql, rows)
            print(f"[SUCCESS] Inserted {len(rows)} states into dim_us_state.")
    finally:
        conn.close()

def ensure_user_state_map():
    ddl = """
    CREATE TABLE IF NOT EXISTS user_state_map (
      client_id  INT PRIMARY KEY,
      state_code CHAR(2) NULL,
      state_name VARCHAR(64) NULL,
      updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
      INDEX idx_state_code (state_code)
    ) ENGINE=InnoDB;
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(ddl)
    finally:
        conn.close()

def refresh_user_state_map():
    """
    users_data(latitude/longitude) 점이 dim_us_state(폴리곤) 안에 들어가면 그 주로 매핑
    - SRID 0: POINT(lon, lat) 그대로 사용
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE user_state_map;")

            sql = """
            INSERT INTO user_state_map (client_id, state_code, state_name)
            SELECT
                u.id AS client_id,
                s.stusps AS state_code,
                s.name AS state_name
            FROM users_data u
            JOIN dim_us_state s
                ON ST_Contains(s.geom, ST_SRID(POINT(u.longitude, u.latitude), 0))
            WHERE u.longitude IS NOT NULL
                AND u.latitude IS NOT NULL;
            """

            cur.execute(sql)

            cur.execute("SELECT COUNT(*) AS cnt FROM user_state_map;")
            mapped = cur.fetchone()["cnt"]

            cur.execute("""
              SELECT COUNT(*) AS cnt
              FROM users_data
              WHERE longitude IS NOT NULL AND latitude IS NOT NULL;
            """)
            total_pts = cur.fetchone()["cnt"]

            print(f"[DONE] user_state_map mapped {mapped}/{total_pts} points.")
    finally:
        conn.close()

def main():
    print("--- Geo Initializer Started ---")
    wait_for_db()
    ensure_dim_us_state()
    load_states_geojson_if_empty()
    ensure_user_state_map()
    refresh_user_state_map()
    print("--- Geo Initializer Completed ---")

if __name__ == "__main__":
    main()
