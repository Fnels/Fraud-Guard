# from datetime import datetime
# import pytz

def get_scaled_timestamp(row, index):
    
    # 현재시간 계산

    # seoul_tz = pytz.timezone('Asia/Seoul')
    # now_seoul = datetime.now(seoul_tz)
    # order_time = now_seoul.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

    
    # 금액 데이터 정제 ($ 기호 등 제거)
    raw_amount = str(row['amount'])
    clean_amount = "".join(c for c in raw_amount if c.isdigit() or c in ['.', '-'])
    
    return {
        "order_time": row['date'],
        "id":int(row['id']),
        "client_id": int(row['client_id']),
        "card_id": int(row['card_id']),
        "merchant_id": int(row['merchant_id']),
        "amount": float(clean_amount),
        "use_chip": row['use_chip'],
        "error": row['errors']
    }