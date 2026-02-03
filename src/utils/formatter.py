from datetime import datetime

def get_scaled_timestamp(row, index):
    """
    요구사항: 시:분까지는 현재 시간, 초와 밀리초는 index 기반으로 생성
    """
    now = datetime.now()
    
    # 1. 날짜와 시:분 추출 (예: 2026-02-02 21:40)
    order_time = now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

    
    # 금액 데이터 정제 ($ 기호 등 제거)
    raw_amount = str(row['amount'])
    clean_amount = "".join(c for c in raw_amount if c.isdigit() or c in ['.', '-'])
    
    return {
        "order_time": order_time,
        "id":int(row['id']),
        "client_id": int(row['client_id']),
        "card_id": int(row['card_id']),
        "merchant_id": int(row['merchant_id']),
        "amount": float(clean_amount),
        "error": row['errors']
    }