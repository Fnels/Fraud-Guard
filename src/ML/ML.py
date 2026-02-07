import pandas as pd
import numpy as np

# 1. 데이터 로드 (파일 경로는 실제 환경에 맞춰주세요)
# transactions_data는 워낙 크니 chunk로 나누거나 dask를 써야 할 수도 있지만, 
# 여기선 pandas로 로드 가능하다고 가정하고 진행합니다.
print("⏳ Loading Data...")

# (1) Users & Cards
users = pd.read_csv('data/users_data.csv')
cards = pd.read_csv('data/cards_data.csv')
users = users.rename(columns={'id':'client_id'})
cards = cards.rename(columns={'id':'card_id'})

# (2) Transactions (예시: 필요한 컬럼만 불러와서 메모리 절약 가능)
# use_chip, merchant_city, zip, mcc 등 버리지 말고 다 가져옵니다.
trans = pd.read_csv('data/transactions_data.csv') 

# (3) Labels (Target)
labels = pd.read_csv('data/train_fraud_labels.csv') # id, target('Yes'/'No')

# 2. 데이터 병합 (Merge)
print("⏳ Merging Data...")

# Transactions + Labels (id 기준)
# 타겟이 없는 데이터는 학습에 쓸 수 없으므로 Inner Join
df = pd.merge(trans, labels, on='id', how='inner')

# + Users (client_id 기준)
df = pd.merge(df, users, on='client_id', how='left')

# + Cards (card_id 기준)
# client_id도 같이 키로 잡으면 더 안전합니다.
df = pd.merge(df, cards, on=['card_id', 'client_id'], how='left')

print(f"✅ Merged Shape: {df.shape}")

# 3. 기초 정제 함수
def clean_dollar(x):
    if isinstance(x, str):
        return float(x.replace('$', '').replace(',', ''))
    return x

print("⏳ Cleaning Data...")

# (1) 금액 관련 컬럼 $ 제거 및 실수형 변환
money_cols = ['amount', 'per_capita_income', 'yearly_income', 'total_debt', 'credit_limit']
for col in money_cols:
    if col in df.columns:
        df[col] = df[col].apply(clean_dollar)

# (2) Amount 절대값 처리 (환불 데이터도 규모로 판단)
df['amount'] = df['amount'].abs()

# (3) Target 변환 (Yes/No -> 1/0)
df['is_fraud'] = df['target'].apply(lambda x: 1 if x == 'Yes' else 0)
df = df.drop(columns=['target']) # 기존 문자열 컬럼 제거

# (4) Online Transaction 처리
# merchant_state가 비어있으면(NaN) -> 'Online'으로 채움 (작성자님 전략 유지)
df['merchant_state'] = df['merchant_state'].fillna('Online')

# (5) 카드 정보 결측치 처리 (혹시 Join 안 된 경우)
df['has_chip'] = df['has_chip'].fillna('NO') # 보수적으로 없음 처리

print("✅ Cleaning Complete")

print("⏳ Engineering Sniper Features...")

# [Feature 1] 기술적 모순 (Tech Mismatch)
# 칩이 있는 카드(YES)인데, 결제는 긁어서(Swipe) 했다? -> 복제카드 의심
# use_chip: 'Swipe Transaction', 'Chip Transaction', 'Online Transaction'
# has_chip: 'YES', 'NO'
def check_tech_mismatch(row):
    # 온라인 거래는 칩 여부와 상관 없음
    if row['use_chip'] == 'Online Transaction':
        return 0
    # 칩이 있는데 스와이프 했으면 의심 (1)
    if row['has_chip'] == 'YES' and row['use_chip'] == 'Swipe Transaction':
        return 1
    return 0

df['tech_mismatch'] = df.apply(check_tech_mismatch, axis=1)

# [Feature 2] 한도 소진율 (Utilization Ratio)
# 500달러 결제라도, 한도가 500달러인 카드를 꽉 채워 쓴 거면 위험함 (속칭 '카드깡' 등)
df['utilization_ratio'] = df['amount'] / df['credit_limit']

# [Feature 3] 소득 대비 지출 비율 (Income Ratio)
# 연봉 3만불인 사람이 한 번에 1만불을 긁으면 이상함
# yearly_income이 0일 수도 있으니 안전하게 처리
df['amount_income_ratio'] = df['amount'] / (df['yearly_income'] + 1) 

# [Feature 4] 시간대 정보 (Hour)
# 문자열 date를 datetime으로 변환
df['date_obj'] = pd.to_datetime(df['date'])
df['hour'] = df['date_obj'].dt.hour
df['is_night'] = df['hour'].apply(lambda x: 1 if 0 <= x < 6 else 0) # 새벽 0시~6시

# [Feature 5] PIN 변경 경과 기간
# 최근에 PIN을 바꿨는데(해킹 등) 고액 결제? 
# 현재 시점을 데이터의 최대 연도로 가정 (2020년)
current_year = df['date_obj'].dt.year.max()
df['pin_years_gap'] = current_year - df['year_pin_last_changed']

# [Feature 6] MCC Risk Mapping (Target Encoding)
# MCC 코드 자체가 중요한 게 아니라, "그 업종에서 사기가 얼마나 빈번한가"가 중요함.
# Train 데이터 기준으로 각 MCC별 사기 확률을 계산해서 매핑합니다.
# 주의: Data Leakage 방지를 위해 원칙적으로는 Train Split 후에 해야 하지만, 
# 데이터가 워낙 크고 분포가 일정하다면 전체 기준으로 맵을 만들어도 무방합니다.
mcc_risk_map = df.groupby('mcc')['is_fraud'].mean()
df['mcc_risk'] = df['mcc'].map(mcc_risk_map)

# [Feature 7] State Risk Mapping
# "CA는 199번"이라는 숫자보다 "CA는 사기율 0.2%"라는 숫자가 모델에겐 훨씬 유익함.
state_risk_map = df.groupby('merchant_state')['is_fraud'].mean()
df['state_risk'] = df['merchant_state'].map(state_risk_map)

# [Feature 8] Zip code 활용 (Distance 대체제)
# Zip code는 너무 많으므로, Zip code별 사기 확률(Risk)로 변환
zip_risk_map = df.groupby('zip')['is_fraud'].mean()
df['zip_risk'] = df['zip'].map(zip_risk_map)
# Zip이 처음 보는 값이라 매핑이 안되면(NaN), 평균 사기율로 채움
df['zip_risk'] = df['zip_risk'].fillna(df['is_fraud'].mean())

print("✅ Feature Engineering Complete")

# 시간벡터 생성
# df_final을 만들기 전, 기초 정제가 끝난 df 상태에서 수행해야 합니다.
# (Users, Cards가 병합되고, date가 datetime으로 변환된 상태)

print("⏳ Generating Velocity Features...")

# 1. 정렬 (User별, 시간순)
# 시간 차이를 계산하려면 반드시 정렬되어 있어야 합니다.
df['date_obj'] = pd.to_datetime(df['date']) # 만약 안 되어 있다면
df = df.sort_values(['client_id', 'date_obj'])

# 2. [핵심 1] 직전 거래와의 시간 차이 (Time Since Last Transaction)
# 사기꾼은 짧은 시간에 여러 번 긁습니다.
# client_id별로 그룹화하여 diff 계산 (초 단위)
df['time_diff_seconds'] = df.groupby('client_id')['date_obj'].diff().dt.total_seconds()
df['time_diff_seconds'] = df['time_diff_seconds'].fillna(999999) # 첫 거래는 큰 값으로

# 3. [핵심 2] 지난 24시간 동안 거래 횟수 (Velocity Count)
# "지난 24시간 동안 100번 긁었다" -> 사기일 확률 매우 높음
# set_index 후 rolling을 쓰면 빠릅니다.
df_sorted = df.set_index('date_obj').sort_index()
# client_id별로 24시간 window count
# (데이터가 커서 오래 걸리면 1시간(1h)으로 줄여도 좋습니다)
count_24h = df_sorted.groupby('client_id')['amount'].rolling('24h').count()

# 인덱스 리셋 후 원본에 병합하기 위해 정리
count_24h = count_24h.reset_index()
# 컬럼명 정리 (client_id, date_obj, amount -> count_24h)
count_24h.columns = ['client_id', 'date_obj', 'count_24h']

# 원본 df에 병합 (Key: client_id, date_obj)
# 주의: 중복 timestamp가 있을 수 있으니 drop_duplicates 고려
df = pd.merge(df, count_24h, on=['client_id', 'date_obj'], how='left')

# 4. [핵심 3] 지난 1시간 동안 결제 금액 합계 (Velocity Amount)
# "평소 100불 쓰는데 지난 1시간 동안 5000불 썼다"
amt_1h = df_sorted.groupby('client_id')['amount'].rolling('1h').sum()
amt_1h = amt_1h.reset_index()
amt_1h.columns = ['client_id', 'date_obj', 'sum_amt_1h']

df = pd.merge(df, amt_1h, on=['client_id', 'date_obj'], how='left')

# 5. 기존 파생변수와 결합
# 이전에 만든 tech_mismatch, utilization_ratio 등도 당연히 포함해야 합니다.

print("✅ Velocity Features Generated")

# 1. 원본 범주형 데이터 복구
# df_final에는 risk 점수만 있고 원본 코드가 없을 수 있으니, 
# 전처리 단계의 df에서 가져오거나 다시 합쳐야 합니다.
# (df는 velocity feature까지 다 만들어진 상태라고 가정)

# 사용할 피처 재선정 (Risk 변수 제거 -> 원본 변수 투입)
raw_cat_features = [
    'amount', 'utilization_ratio', 'amount_income_ratio', 
    'tech_mismatch', 'pin_years_gap', 'num_credit_cards', 
    'hour', 'is_night', 
    'time_diff_seconds', 'count_24h', 'sum_amt_1h', # Velocity
    'current_age', 'credit_score',
    
    # [핵심 변경] Risk 변수 대신 원본 범주형 변수 사용
    'mcc',            # mcc_risk 대신 (숫자형이라도 범주형으로 취급)
    'merchant_state', # state_risk 대신 (문자열)
    'zip',            # zip_risk 대신 (숫자형이라도 범주형으로 취급)
    'use_chip',       # tech_mismatch가 있지만 원본도 같이 줘봅니다
    'card_brand',     # 브랜드 정보도 추가
    
    'is_fraud' # Target
]

# 해당 컬럼들로 데이터셋 구성
# (주의: Null 값이 있으면 안 됩니다. 문자열은 'Unknown', 숫자는 -1 등으로 채우세요)
df_raw_cat = df[raw_cat_features].copy()
df_raw_cat['merchant_state'] = df_raw_cat['merchant_state'].fillna('Unknown')
df_raw_cat['card_brand'] = df_raw_cat['card_brand'].fillna('Unknown')
df_raw_cat['mcc'] = df_raw_cat['mcc'].astype(str) # 숫자가 아닌 범주로 인식되게 문자열 변환 추천
df_raw_cat['zip'] = df_raw_cat['zip'].astype(str)