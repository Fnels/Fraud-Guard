import redis
import json
# from typing import Optional
import  logging

# Redis 연결 설정 

# 로깅 설정 (에러 발생 시 추적용)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RedisClient:
    # 1. Connection Pool 사용: 매번 연결을 맺고 끊는 비용을 줄임 (DE적 관점)
    def __init__(self, host='localhost', port=6379, db=0):
        self.pool = redis.ConnectionPool(
            host=host,
            port=port,
            # 1. 소켓 타임아웃: Redis가 응답 없으면 무한정 대기하지 않고 끊기
            socket_timeout=2.0,
            db=db,
            # 2. 재시도 전략: 네트워크 불안정 시 자동 재시도
            retry_on_timeout=True,
            # 3. 최대 연결 수: 시스템 자원 보호
            max_connections=100,
            decode_responses=True  # 데이터를 문자열로 바로 받아옴
        )
        self.conn = redis.StrictRedis(connection_pool=self.pool)

    # 연결 확인
    def ping(self):
            """Redis 연결 확인"""
            return self.pool.ping()


    # 결과 저장  Kafka에서 데이터를 꺼내 사기 판별을 끝낸 후, 그 결과값을 Redis에 저장할 때 사용.
    def set_result(self, transaction_id, data, expire=300):
            """ML 결과를 Redis에 저장 (TTL 5분 설정)"""
            try:
                key = f"tx_res:{transaction_id}"
                # 데이터를 JSON 문자열로 직렬화하여 저장
                self.conn.set(key, json.dumps(data), ex=expire)
            except Exception as e:
                logger.error(f"Redis 저장 실패: {e}")

    # 결과 확인
    def get_result(self, transaction_id):
            """Flask에서 특정 거래의 결과를 조회"""
            try:
                key = f"tx_res:{transaction_id}"
                result = self.conn.get(key)
                if result:
                    return json.loads(result)
                return None
            except Exception as e:
                logger.error(f"Redis 조회 실패: {e}")
                return None


    def exists(self, transaction_id: str) -> bool:
            """키 존재 여부"""
            return self.conn.exists(transaction_id) > 0

    def delete_result(self, transaction_id: str):
        """결과 삭제"""
        self.conn.delete(transaction_id)


# 싱글톤 패턴처럼 사용하기 위해 객체 생성
# Docker Compose 사용 시 host='redis'로 변경하세요.
redis_api = RedisClient(host='localhost', port=6379)