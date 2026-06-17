import streamlit as st
import matplotlib.pyplot as plt

PASSWORD = "Concrete2026"

st.title("화재 후 콘크리트 압축강도 예측")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    pwd = st.text_input(
        "비밀번호를 입력하세요",
        type="password",
        key="password_input")

    if st.button("로그인"):
        if pwd == PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("비밀번호가 틀렸습니다.")

    st.stop()

import pandas as pd
import pickle
from xgboost import XGBRegressor

def optional_float(value):
    value = value.strip()

    if value == "":
        return None

    return float(value)

# 모델 불러오기
model = XGBRegressor()
model.load_model("xgb_concrete_strength_final.json")

# 학습 당시 사용한 변수명 불러오기
with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

st.write("입력값을 넣으면 화재 후 압축강도를 예측합니다.")

# 기본 입력란
category = st.selectbox("Concrete category / 콘크리트 종류", ["Concrete", "Lightweight concrete", "Cementless concrete"])
temperature = st.number_input("Temperature (°C)", min_value=0, value=600, step=10, format="%d")
fc_28 = st.number_input("28 day compressive strength (MPa)", min_value=0.0, value=40.0, step=1.0, format="%.1f")
fc_90 = st.text_input("90 day compressive strength (MPa)")
cement = st.number_input("Cement (kg/m³)", min_value=0.0, value=400.0, step=1.0, format="%.1f")
water = st.number_input("Water (kg/m³)", min_value=0.0, value=180.0, step=1.0, format="%.1f")

if cement > 0:
    wc = water / cement
else:
    wc = 0

st.write(f"W/C = {wc:.3f}")

aggregate = st.number_input("Coarse aggregate / 굵은골재 (kg/m³)", min_value=0.0, value=1000.0, step=1.0, format="%.1f")
sand = st.number_input("Fine aggregate / 잔골재 (kg/m³)", min_value=0.0, value=700.0, step=1.0, format="%.1f")
aggregate_type = st.selectbox("Aggregate type / 골재 종류", ["Carbonate", "Siliceous"])
fly_ash = st.number_input("Fly Ash (kg/m³)", min_value=0.0, value=0.0, step=1.0, format="%.1f")
slag = st.number_input("Slag (kg/m³)", min_value=0.0, value=0.0, step=1.0, format="%.1f")
silica_fume = st.number_input("Silica Fume (kg/m³)", min_value=0.0, value=0.0, step=1.0, format="%.1f")
superplasticizer = st.number_input("Superplasticizer (kg/m³)", min_value=0.0, value=0.0, step=1.0, format="%.1f")

test_day = st.number_input("Test age / 실험 재령 (days)", min_value=1, value=28, step=1, format="%d")

if temperature <= 50:
    st.info("50℃ 이하의 시편은 비가열 시편으로 처리되며, 냉각 조건은 적용하지 않습니다.")
    cooling_time = 0
else:
    cooling_condition = st.selectbox("Cooling condition", ["Hot", "Ambient", "Cooling period"])

    if cooling_condition == "Hot":
        cooling_time = 0
    elif cooling_condition == "Ambient":
        cooling_time = 0.2
    else:
        cooling_time = st.number_input("Cooling period (days)", min_value=1, value=1, step=1, format="%d")

st.write("Fiber information")

st.write("동일한 종류의 섬유를 여러 길이로 사용한 경우, 함량은 합산하여 입력하고, 길이는 함량 가중평균 길이를 입력하세요.")

fiber_types = ["None", "Steel", "PVA", "PP", "Glass", "Basalt"]

if "fiber_count" not in st.session_state:
    st.session_state.fiber_count = 0

if st.button("+ Add Fiber"):
    if st.session_state.fiber_count < 5:
        st.session_state.fiber_count += 1
    else:
        st.warning("최대 5개 섬유까지만 추가할 수 있습니다.")

fibers = []

for i in range(st.session_state.fiber_count):
    st.write(f"Fiber {i+1}")

    fiber_type = st.selectbox(f"Fiber {i+1} type", fiber_types, key=f"fiber_type_{i}")

    fiber_content = st.number_input(f"Fiber {i+1} content (kg/m³)", min_value=0.0, value=0.0, step=0.1,format="%.1f", key=f"fiber_content_{i}")

    fiber_length = st.number_input(f"Fiber {i+1} length (mm)", min_value=0.0, value=0.0, step=0.1, format="%.1f", key=f"fiber_length_{i}")

    fibers.append({"type": fiber_type, "content": fiber_content, "length": fiber_length})

if st.button("예측하기"):
    # 학습 때 사용한 변수 구조와 동일한 빈 데이터 생성
    input_df = pd.DataFrame(columns=feature_columns)
    input_df.loc[0] = 0

    # 입력한 값 반영
    input_df.loc[0, "Category_Cementless_concrete"] = 0
    input_df.loc[0, "Category_Concrete"] = 0
    input_df.loc[0, "Category_Lightweight_concrete"] = 0

    if category == "Concrete":
        input_df.loc[0, "Category_Concrete"] = 1
    elif category == "Lightweight_concrete":
        input_df.loc[0, "Category_Lightweight_concrete"] = 1
    elif category == "Cementless_concrete":
        input_df.loc[0, "Category_Cementless_concrete"] = 1

    input_df.loc[0, "Data_format_Code"] = 0
    input_df.loc[0, "Data_format_Experiment"] = 1

    input_df.loc[0, "Temperature"] = temperature
    input_df.loc[0, "FC_28"] = fc_28

    input_df.loc[0, "FC_90"] = optional_float(fc_90)

    input_df.loc[0, "W/C"] = wc
    input_df.loc[0, "Cement"] = cement
    input_df.loc[0, "Water"] = water
    input_df.loc[0, "Aggregate"] = aggregate
    input_df.loc[0, "Sand"] = sand
    input_df.loc[0, "Aggregate_type_Carbonate"] = 0
    input_df.loc[0, "Aggregate_type_Siliceous"] = 0
    input_df.loc[0, "Aggregate_type_Unknown"] = 0
    input_df.loc[0, "Fly_ash"] = fly_ash
    input_df.loc[0, "Slag"] = slag
    input_df.loc[0, "Silica_fume"] = silica_fume
    input_df.loc[0, "Superplasticizer"] = superplasticizer
    input_df.loc[0, "Cooling_Time"] = cooling_time
    input_df.loc[0, "Test_day"] = test_day

    if aggregate_type == "Carbonate":
        input_df.loc[0, "Aggregate_type_Carbonate"] = 1
    elif aggregate_type == "Siliceous":
        input_df.loc[0, "Aggregate_type_Siliceous"] = 1

# 섬유 변수 초기화
    fiber_columns = [
        "Steel_fibre", "Steel_fibre_length",
        "PVA_fibre", "PVA_fibre_length",
        "PP_fibre", "PP_fibre_length",
        "Glass_fibre", "Glass_fibre_length",
        "Basalt_fibre", "Basalt_fibre_length"]

    for col in fiber_columns:
        input_df.loc[0, col] = 0

# 섬유
    for fiber in fibers:
        ftype = fiber["type"]
        content = fiber["content"]
        length = fiber["length"]

        if ftype == "Steel":
            input_df.loc[0, "Steel_fibre"] += content
            input_df.loc[0, "Steel_fibre_length"] = length

        elif ftype == "PVA":
            input_df.loc[0, "PVA_fibre"] += content
            input_df.loc[0, "PVA_fibre_length"] = length

        elif ftype == "PP":
            input_df.loc[0, "PP_fibre"] += content
            input_df.loc[0, "PP_fibre_length"] = length

        elif ftype == "Glass":
            input_df.loc[0, "Glass_fibre"] += content
            input_df.loc[0, "Glass_fibre_length"] = length

        elif ftype == "Basalt":
            input_df.loc[0, "Basalt_fibre"] += content
            input_df.loc[0, "Basalt_fibre_length"] = length

    residual_strength = model.predict(input_df)[0]

    control_df = input_df.copy()
    control_df.loc[0, "Temperature"] = 20
    control_df.loc[0, "Cooling_Time"] = 0

    control_strength = model.predict(control_df)[0]

    residual_ratio = residual_strength / control_strength * 100

    temp_list = [20, 100, 200, 300, 400, 500, 600, 800, 1000, 1200]

    graph_data = []

    for temp in temp_list:
        temp_df = input_df.copy()
        temp_df.loc[0, "Temperature"] = temp

        if temp <= 50:
            temp_df.loc[0, "Cooling_Time"] = 0
        else:
            temp_df.loc[0, "Cooling_Time"] = cooling_time

        pred_strength = model.predict(temp_df)[0]

        graph_data.append({
            "Temperature": temp,
            "Compressive Strength (MPa)": pred_strength})

    graph_df = pd.DataFrame(graph_data)

    # 예측
    prediction = model.predict(input_df)[0]

    st.subheader("예측 결과")
    st.write(f"화재 후 압축강도 예측값: {prediction:.2f} MPa")
    st.write(f"잔존강도율: {residual_ratio:.1f} %")

    st.subheader("Temperature-strength curve")

    fig, ax = plt.subplots()

    ax.plot(
        graph_df["Temperature"],
        graph_df["Compressive Strength (MPa)"],
        marker="o")

    ax.set_xlabel("Temperature (℃)")
    ax.set_ylabel("Predicted compressive strength (MPa)")
    ax.set_title("Temperature-strength curve")
    ax.grid(True)

    st.pyplot(fig)
