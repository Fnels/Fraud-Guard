from datetime import datetime
import time

def get_scaled_timestamp(row, index):
    now = datetime.now()
    order_time = now.strftime('%Y-%m-%d %H:%M:%S') + f".{index % 1000:03d}"
    
    # --- 금액 데이터 정제 로직 추가 ---
    raw_amount = str(row['amount'])
    # '$', ',', ' ' 등 숫자/소수점/마이너스 외의 문자 제거
    clean_amount = "".join(c for c in raw_amount if c.isdigit() or c in ['.', '-'])
    
    try:
        amount_float = float(clean_amount)
    except ValueError:
        amount_float = 0.0  # 변환 실패 시 기본값
    # ------------------------------

    return {
        "order_time": order_time,
        "client_id": int(row['client_id']),
        "card_id": int(row['card_id']),
        "merchant_id": int(row['merchant_id']),
        "amount": amount_float
    }