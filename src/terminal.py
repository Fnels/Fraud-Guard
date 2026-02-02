import pandas as pd
import requests
import json, time, os 
from utils.formatter import get_scaled_timestamp

def send_data():
    df = pd.read_csv('data/raw_data.csv')
    # url = "http://본사서버주소:포트/endpoint" # producer_raw.py의 API 주소
    
    for i, row in df.iterrows():
        # 1. 데이터 정제 (JSON 형태)
        data = row.to_dict()
        data['timestamp'] = get_scaled_timestamp(i) # 가공된 시간 추가
    
        # 2. POST 전송
        try:
            response = requests.post(url, json=data)
            print(f"[{i}] 전송 완료: {response.status_code}")
        except Exception as e:
            print(f"전송 실패: {e}")
            
        # 3. 단말기처럼 보이게 약간의 딜레이 (예: 0.1초)
        time.sleep(0.1)


