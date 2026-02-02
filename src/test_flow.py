import pandas as pd
import json
import time
from utils.formatter import get_scaled_timestamp

def test_data_flow(limit=5):
    """
    URL 전송 없이 데이터 가공 로직만 5건 테스트
    """
    print("=== [테스트 시작] 데이터 가공 시뮬레이션 ===")
    
    try:
        # 1. CSV 로드
        df = pd.read_csv('data/transactions_data.csv')
        
        # 2. 상위 n건만 루프
        for i, row in df.head(limit).iterrows():
            # formatter의 get_scaled_timestamp 호출
            # (작성하신 코드보니 row와 index를 인자로 받게 설계하셨네요!)
            formatted_data = get_scaled_timestamp(row, i)
            
            # 3. JSON 형태로 예쁘게 출력 (indent=4 사용)
            json_output = json.dumps(formatted_data, indent=4, ensure_ascii=False)
            
            print(f"\n[Index {i}] 가공 결과:")
            print(json_output)
            
            # 타입 검증 (DE는 데이터 타입을 항상 체크해야 함)
            print(f"- order_time 타입: {type(formatted_data['order_time'])}")
            print(f"- amount 타입: {type(formatted_data['amount'])}")
            
            time.sleep(0.1)
            
    except FileNotFoundError:
        print("에러: 'data/raw_data.csv' 파일을 찾을 수 없습니다.")
    except Exception as e:
        print(f"테스트 중 오류 발생: {e}")

if __name__ == "__main__":
    test_data_flow()