from flask import Flask, jsonify
import pandas as pd
import time
import logging
from utils.formatter import get_scaled_timestamp

app = Flask(__name__)

# 도커 데스크탑 로그(표준 출력) 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s' # 로그 시간 제외, 메시지만 깔끔하게 출력
)
logger = logging.getLogger(__name__)

@app.route('/')
def index():
    return "Terminal Server Ready. Access /start to view data logs in Docker Desktop."

@app.route('/start', methods=['GET', 'POST'])
def start_simulation():
    try:
        # 1. 데이터 로드
        df = pd.read_csv('data/transactions_data.csv')
        
        logger.info("==================================================")
        logger.info(f"🚀 데이터 가공 시뮬레이션 시작 (총 {len(df)}건)")
        logger.info("==================================================")

        for i, row in df.iterrows():
            # (시간 형식: 시:분은 현재, 초.밀리초는 인덱스 기반)
            data = get_scaled_timestamp(row, i)
            
            # 3. 도커 로그로 한 줄씩 출력 (줄줄이 찍히는 핵심 부분)
            # JSON 모양을 한 줄로 예쁘게 정렬해서 출력합니다. 고유id 부여, 초 변경,
            log_msg = f"📤 [IDX:{i:04d}] | {data['id']} | {data['order_time']} | Client:{data['client_id']} | CardId:{data['card_id']}| MerchantId:{data['merchant_id']}｜Amt:{data['amount']}"
            logger.info(log_msg)
            
            # 4. 실시간 느낌을 위한 딜레이 (0.1초)
            time.sleep(0.1)

        logger.info("==================================================")
        logger.info("✅ 모든 데이터 가공 및 출력 완료")
        logger.info("==================================================")

        return jsonify({"status": "success", "processed": len(df)})

    except FileNotFoundError:
        logger.error("❌ 에러: CSV 파일을 찾을 수 없습니다 (data/transactions_data.csv)")
        return "File Not Found", 404
    except Exception as e:
        logger.error(f"🔥 예상치 못한 오류: {e}")
        return str(e), 500

if __name__ == '__main__':
    # 도커 환경 포트 5000번 사용
    app.run(host='0.0.0.0', port=5000)