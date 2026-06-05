import streamlit as st

PASSWORD = "Concrete2026"

pwd = st.text_input(
    "비밀번호를 입력하세요",
    type="password"
)

if pwd != PASSWORD:
    st.warning("비밀번호를 입력해야 접속할 수 있습니다.")
    st.stop()

import pandas as pd
import pickle
from xgboost import XGBRegressor

# 모델 불러오기
model = XGBRegressor()
model.load_model("xgb_concrete_strength_final.json")

# 학습 당시 사용한 변수명 불러오기
with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

st.title("화재 후 콘크리트 압축강도 예측")

st.write("입력값을 넣으면 XGBoost 모델이 화재 후 압축강도를 예측합니다.")

# 기본 입력란
temperature = st.number_input("Temperature (°C)", value=600.0)
fc_28 = st.number_input("FC_28 (MPa)", value=40.0)
cement = st.number_input("Cement (kg/m³)", value=400.0)
water = st.number_input("Water (kg/m³)", value=180.0)
aggregate = st.number_input("Coarse aggregate / 굵은골재 (kg/m³)", value=1000.0)
sand = st.number_input("Fine aggregate / 잔골재 (kg/m³)", value=700.0)
aggregate_type = st.selectbox("Aggregate type / 골재 종류", ["Carbonate", "Siliceous", "Unknown"])

if cement > 0:
    wc = water / cement
else:
    wc = 0

st.write(f"W/C = {wc:.3f}")

if st.button("예측하기"):
    # 학습 때 사용한 변수 구조와 동일한 빈 데이터 생성
    input_df = pd.DataFrame(columns=feature_columns)
    input_df.loc[0] = 0

    # 사용자가 입력한 값 반영
    input_df.loc[0, "Temperature"] = temperature
    input_df.loc[0, "FC_28"] = fc_28
    input_df.loc[0, "W/C"] = wc
    input_df.loc[0, "Cement"] = cement
    input_df.loc[0, "Water"] = water
    input_df.loc[0, "Aggregate"] = aggregate
    input_df.loc[0, "Sand"] = sand
    input_df.loc[0, "Aggregate_type_Carbonate"] = 0
    input_df.loc[0, "Aggregate_type_Siliceous"] = 0
    input_df.loc[0, "Aggregate_type_Unknown"] = 0

    if aggregate_type == "Carbonate":
        input_df.loc[0, "Aggregate_type_Carbonate"] = 1
    elif aggregate_type == "Siliceous":
        input_df.loc[0, "Aggregate_type_Siliceous"] = 1
    else:
        input_df.loc[0, "Aggregate_type_Unknown"] = 1

    # 예측
    prediction = model.predict(input_df)[0]

    st.subheader("예측 결과")
    st.write(f"화재 후 압축강도 예측값: {prediction:.2f} MPa")
