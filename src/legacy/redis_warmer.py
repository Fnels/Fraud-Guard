# src/redis_warmer.py
import pymysql
import redis
import os
import json, time

def warm_up():
    # 1. 연결 설정
    mysql_conn = pymysql.connect(
        host='mysql', user='root', 
        password=os.getenv('MYSQL_ROOT_PASSWORD', 'root'),
        db=os.getenv('MYSQL_DATABASE', 'fraud_detection'),
        cursorclass=pymysql.cursors.DictCursor
    )
    r = redis.StrictRedis(host='redis', port=6379, db=0)

    try:
        with mysql_conn.cursor() as cursor:
            # 유저 데이터 적재
            cursor.execute("SELECT id FROM users_data")
            users = [row['id'] for row in cursor.fetchall()]
            if users: r.sadd("check:users", *users)
            
            # 카드 데이터 적재 (ID와 소유주 관계)
            cursor.execute("SELECT id, client_id FROM cards_data")
            for row in cursor.fetchall():
                r.set(f"check:card:{row['id']}", row['client_id'])

            # 소상공인 데이터 적재
            cursor.execute("SELECT id FROM merchants_data")
            merchants = cursor.fetchall()

            for merchant in merchants:
                r.set(f"merchant:{merchant['id']}", json.dumps(merchant))
            print(f"✅ Loaded {len(merchants)} merchants")
                
            print(f"[SUCCESS] Redis Warming: {len(users)} users and cards loaded.")

        print("=" * 50)
        print("✅ Redis 캐싱 완료!")
        print("=" * 50)    
    finally:
        mysql_conn.close()

if __name__ == "__main__":
    # MySQL 준비될 때까지 대기
    max_retries = 30
    for i in range(max_retries):
        try:
            conn = pymysql.connect(
                host=DB_HOST,
                user=DB_USER,
                password=DB_PASSWORD,
                db=DB_NAME
            )
            conn.close()
            print("[INFO] MySQL is ready!")
            break
        except Exception as e:
            print(f"[WAIT] MySQL not ready ({i+1}/{max_retries})...")
            time.sleep(2)
    warm_up()