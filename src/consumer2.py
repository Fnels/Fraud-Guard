# consumer group 2 의 역할을 수행합니다.
# 윈도우 집계 처리. 1분, 1시간, 24시간 단위 3개로 쪼개야 합니다.
'''
1분 윈도우 집계 테이블: agg_txn_1m
1시간 윈도우 집계 테이블: agg_txn_1h
24시간 윈도우 집계 테이블: agg_txn_1d
'''

# src/consumer2.py
import os
import json
import pymysql
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DoubleType, BooleanType
)
from pyspark.sql.functions import (
    col, from_json, to_timestamp, window,
    count as f_count, sum as f_sum, when
)

# ----------------------------
# 0) 환경 변수/상수
# ----------------------------
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "kafka:9092")
SOURCE_TOPIC = os.getenv("KAFKA_TOPIC", "2nd-topic")

MYSQL_HOST = os.getenv("MYSQL_HOST", "mysql")
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_ROOT_PASSWORD", "root")
MYSQL_DB = os.getenv("MYSQL_DATABASE", "fraud_guard")

# 윈도우 종류(요구사항: 1분/1시간/24시간)
WINDOW_SPECS = [
    ("1 minute", "agg_txn_1m"),
    ("1 hour",   "agg_txn_1h"),
    ("1 day",    "agg_txn_1d"),
]

# MySQL 업서트 배치 크기(너무 크게 잡으면 메모리/패킷 문제)
UPSERT_BATCH_SIZE = 1000


# ----------------------------
# 1) MySQL 유틸 함수
# ----------------------------
def get_mysql_conn():
    """
    MySQL 연결을 생성하는 함수.
    - foreachBatch에서 매 배치마다 연결해도 되지만,
      실무에서는 커넥션 풀을 두는 편이 더 좋습니다.
    - 데모/MVP에서는 단순 연결로도 충분.
    """
    return pymysql.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        db=MYSQL_DB,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True
    )


def create_agg_table_if_not_exists(table_name: str):
    """
    집계 결과를 저장할 테이블을 MySQL에 생성.
    - PK(window_start, window_end, merchant_id)로 잡아서 업서트 가능하게 설계
    """
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
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
    """
    conn = get_mysql_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(ddl)
    finally:
        conn.close()


def upsert_agg_rows(table_name: str, rows: list):
    """
    집계 DataFrame의 결과(rows)를 MySQL에 업서트(INSERT ... ON DUPLICATE KEY UPDATE)로 저장.
    - 집계 결과는 보통 row 수가 상대적으로 적어서(merchant_id별 window) 드라이버 업서트 방식이 MVP에 적합
    - 실무에서 대규모라면 JDBC sink + merge 전략(또는 Delta/Hudi/Iceberg)도 고려
    """
    if not rows:
        return

    sql = f"""
    INSERT INTO {table_name}
      (window_start, window_end, merchant_id, txn_cnt, amount_sum, fraud_cnt, valid_cnt)
    VALUES
      (%s, %s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
      txn_cnt = VALUES(txn_cnt),
      amount_sum = VALUES(amount_sum),
      fraud_cnt = VALUES(fraud_cnt),
      valid_cnt = VALUES(valid_cnt),
      updated_at = CURRENT_TIMESTAMP;
    """

    conn = get_mysql_conn()
    try:
        with conn.cursor() as cur:
            # executemany가 너무 큰 리스트를 받으면 부담될 수 있으니 chunk로 나눔
            for i in range(0, len(rows), UPSERT_BATCH_SIZE):
                chunk = rows[i:i + UPSERT_BATCH_SIZE]
                cur.executemany(sql, chunk)
    finally:
        conn.close()


def dt_to_mysql_str(dt: datetime) -> str:
    """
    Spark Timestamp -> MySQL DATETIME(3) 문자열로 변환.
    - DATETIME(3)은 밀리초(3자리)까지만 저장.
    """
    # dt가 None일 가능성은 낮지만 방어
    if dt is None:
        return None
    # 마이크로초 6자리 중 앞 3자리만 사용(밀리초)
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


# ----------------------------
# 2) Spark: Kafka 메시지 스키마 정의
# ----------------------------
def build_value_schema():
    """
    2nd-topic에 들어오는 JSON value 스키마 정의.
    현재 worker.py가 output_data에 넣는 필드 기준으로 구성.
    """
    return StructType([
        StructField("id", IntegerType(), True),
        StructField("order_id", IntegerType(), True),
        StructField("order_time", StringType(), True),
        StructField("client_id", IntegerType(), True),
        StructField("card_id", IntegerType(), True),
        StructField("merchant_id", IntegerType(), True),
        StructField("amount", DoubleType(), True),
        StructField("error", StringType(), True),
        StructField("is_valid", BooleanType(), True),
        StructField("is_fraud", BooleanType(), True),
    ])


# ----------------------------
# 3) foreachBatch writer 생성
# ----------------------------
def make_foreach_batch_writer(table_name: str):
    """
    foreachBatch에서 사용할 writer 함수를 만들어 반환.
    - 반환된 함수는 Spark가 micro-batch마다 호출함.
    """
    def write_batch(batch_df, batch_id: int):
        """
        batch_df: 집계가 끝난 DataFrame
        batch_id: 마이크로배치 번호(디버깅용)
        """
        # 집계 결과가 없으면 바로 종료
        if batch_df.rdd.isEmpty():
            return

        # window struct를 평탄화해서 driver로 수집
        flat = (
            batch_df
            .select(
                col("window.start").alias("window_start"),
                col("window.end").alias("window_end"),
                col("merchant_id").cast("int").alias("merchant_id"),
                col("txn_cnt").cast("long").alias("txn_cnt"),
                col("amount_sum").alias("amount_sum"),
                col("fraud_cnt").cast("long").alias("fraud_cnt"),
                col("valid_cnt").cast("long").alias("valid_cnt"),
            )
        )

        # 여기서는 집계 row 수가 제한적이라고 가정(MVP)
        collected = flat.collect()

        rows = []
        for r in collected:
            rows.append((
                dt_to_mysql_str(r["window_start"]),
                dt_to_mysql_str(r["window_end"]),
                int(r["merchant_id"]) if r["merchant_id"] is not None else 0,
                int(r["txn_cnt"]),
                float(r["amount_sum"]) if r["amount_sum"] is not None else 0.0,
                int(r["fraud_cnt"]),
                int(r["valid_cnt"]),
            ))

        upsert_agg_rows(table_name, rows)
        print(f"[BATCH:{batch_id}] upserted {len(rows)} rows into {table_name}")

    return write_batch


# ----------------------------
# 4) 메인: 3개 윈도우 스트림 실행
# ----------------------------
def main():
    # (1) 테이블 생성
    for _, table in WINDOW_SPECS:
        create_agg_table_if_not_exists(table)

    # (2) Spark 세션 생성
    spark = (
        SparkSession.builder
        .appName("consumer2-window-aggregations")
        # 시간대 맞추기: order_time이 로컬 시간이라면 Asia/Seoul 유지가 보통 편함
        .config("spark.sql.session.timeZone", "Asia/Seoul")
        # shuffle 파티션: 로컬/데모면 너무 크지 않게
        .config("spark.sql.shuffle.partitions", "16")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    value_schema = build_value_schema()

    # (3) Kafka에서 스트리밍 읽기
    raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
        .option("subscribe", SOURCE_TOPIC)
        .option("startingOffsets", "latest")
        .load()
    )

    # (4) value(JSON) 파싱 + order_time을 timestamp로 변환
    parsed = (
        raw.selectExpr("CAST(value AS STRING) AS value_str")
           .select(from_json(col("value_str"), value_schema).alias("v"))
           .select("v.*")
    )

    # order_time 포맷이 "YYYY-MM-DD HH:MM:SS.SSS" 형태라고 가정
    # 만약 포맷이 다르면 pattern을 맞춰야 함(여기서 가장 많이 막힘)
    base = (
        parsed
        .withColumn("order_ts", to_timestamp(col("order_time"), "yyyy-MM-dd HH:mm:ss.SSS"))
        .filter(col("order_ts").isNotNull())
    )

    # (5) 윈도우별 집계 DF 생성 및 스트림 시작
    queries = []
    for win_duration, table_name in WINDOW_SPECS:
        agg = (
            base
            # 워터마크: 늦게 도착하는 데이터 허용 범위(데모면 5분 정도면 충분)
            .withWatermark("order_ts", "5 minutes")
            .groupBy(
                window(col("order_ts"), win_duration),
                col("merchant_id")
            )
            .agg(
                f_count("*").alias("txn_cnt"),
                f_sum(col("amount")).alias("amount_sum"),
                f_sum(when(col("is_fraud") == True, 1).otherwise(0)).alias("fraud_cnt"),
                f_sum(when(col("is_valid") == True, 1).otherwise(0)).alias("valid_cnt"),
            )
        )

        writer = make_foreach_batch_writer(table_name)

        q = (
            agg.writeStream
            .outputMode("update")  # window aggregation은 update/append 전략을 상황에 맞게 선택
            .foreachBatch(writer)
            .option("checkpointLocation", f"/tmp/checkpoints/{table_name}")
            .start()
        )

        print(f"[STARTED] window={win_duration} -> table={table_name}")
        queries.append(q)

    # (6) 종료까지 대기
    for q in queries:
        q.awaitTermination()


if __name__ == "__main__":
    main()
