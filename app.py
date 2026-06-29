import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from xgboost import XGBRegressor

st.set_page_config(page_title="Concrete Residual Strength Prediction", layout="wide")
st.markdown("""<style>.block-container { max-width: 1350px; padding-top: 2rem; padding-left: 4rem; padding-right: 4rem; } </style> """, unsafe_allow_html=True)
input_area, result_area = st.columns([1.7, 1], gap="large")

def optional_float(value):
    value = value.strip()

    if value == "":
        return None

    return float(value)

PASSWORD = "Concrete2026"

with input_area:

    st.title("화재 후 콘크리트 압축강도 예측")

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        pwd = st.text_input(
            "비밀번호를 입력하세요",
            type="password", key="password_input")
        if st.button("로그인"):
            if pwd == PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("비밀번호가 틀렸습니다.")
        st.stop()

    #모델 불러오기
    model = XGBRegressor()
    model.load_model("xgb_concrete_strength_final.json")

    with open("feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)

    st.write("입력값을 넣으면 화재 후 압축강도를 예측합니다.")

    # 기본 입력
    col1, col2 = st.columns(2)
    with col1:
        prediction_method = st.selectbox("Prediction method / 예측 방법", ["Experiment-based", "Code-based"])
    with col2:
        design_code = st.selectbox("Design code / 적용 기준", ["Eurocode 2", "ACI", "fib Model Code"], disabled=True)
    if prediction_method == "Code-based": st.info("현재 Code 기반 예측 압축강도는 검토 중으로, Experiment 기반 예측만 가능합니다.")

    col1, col2 = st.columns(2)
    with col1:
        category = st.selectbox("Concrete category / 콘크리트 종류", ["Concrete", "Lightweight concrete", "Cementless concrete"])
    with col2:
        temperature = st.number_input("Temperature (°C)", min_value=0, value=600, step=10, format="%d")

    col1, col2 = st.columns(2)
    with col1:
        fc_28 = st.number_input("28 day compressive strength (MPa)", min_value=0.0, value=40.0, step=1.0, format="%.1f")
    with col2:
        fc_90 = st.text_input("90 day compressive strength (MPa)")

    col1, col2 = st.columns(2)
    with col1:
        cement = st.number_input("Cement (kg/m³)", min_value=0.0, value=400.0, step=1.0, format="%.1f")
    with col2:
        aggregate_type = st.selectbox("Aggregate type / 골재 종류", ["Carbonate", "Siliceous"])

    col1, col2 = st.columns(2)
    with col1:
        water = st.number_input("Water (kg/m³)", min_value=0.0, value=180.0, step=1.0, format="%.1f")
    with col2:
        if cement > 0:
            wc = water / cement
        else:
            wc = 0
        st.write(f"W/C = {wc:.3f}")

    col1, col2 = st.columns(2)
    with col1:
        aggregate = st.number_input("Coarse aggregate / 굵은골재 (kg/m³)", min_value=0.0, value=1000.0, step=1.0, format="%.1f")
    with col2:
        sand = st.number_input("Fine aggregate / 잔골재 (kg/m³)", min_value=0.0, value=700.0, step=1.0, format="%.1f")

    col1, col2 = st.columns(2)
    with col1:
        fly_ash = st.number_input("Fly Ash (kg/m³)", min_value=0.0, value=0.0, step=1.0, format="%.1f")
    with col2:
        slag = st.number_input("Slag (kg/m³)", min_value=0.0, value=0.0, step=1.0, format="%.1f")

    col1, col2 = st.columns(2)
    with col1:
        silica_fume = st.number_input("Silica Fume (kg/m³)", min_value=0.0, value=0.0, step=1.0, format="%.1f")
    with col2:
        superplasticizer = st.number_input("Superplasticizer (kg/m³)", min_value=0.0, value=0.0, step=1.0, format="%.1f")

    col1, col2 = st.columns(2)
    with col1:
        test_day = st.number_input("Test age / 실험 재령 (days)", min_value=1, value=28, step=1, format="%d")
    with col2:
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
    st.caption("동일한 종류의 섬유를 여러 길이로 사용한 경우, 함량은 합산하여 입력하고, 길이는 함량 가중평균 길이를 입력하세요.")

    fiber_types = ["None", "Steel", "PVA", "PP", "Glass", "Basalt"]

    if "fiber_count" not in st.session_state:
        st.session_state.fiber_count = 0

    if st.button("+ Add Fiber"):
        if st.session_state.fiber_count < 5:
            st.session_state.fiber_count += 1
        else:
            st.warning("섬유 최대 5개까지만 추가할 수 있습니다.")

    fibers = []

    for i in range(st.session_state.fiber_count):
        st.write(f"Fiber {i+1}")

        col1, col2, col3 = st.columns(3)

        with col1:
            fiber_type = st.selectbox(f"Fiber {i+1} type", fiber_types, key=f"fiber_type_{i}")
        with col2:
            fiber_content = st.number_input(f"Fiber {i+1} content (kg/m³)", min_value=0.0, value=0.0, step=0.1,format="%.1f", key=f"fiber_content_{i}")
        with col3:
            fiber_length = st.number_input(f"Fiber {i+1} length (mm)", min_value=0.0, value=0.0, step=0.1, format="%.1f", key=f"fiber_length_{i}")
        fibers.append({"type": fiber_type, "content": fiber_content, "length": fiber_length})

    if st.button("예측하기"):
        input_df = pd.DataFrame(columns=feature_columns)
        input_df.loc[0] = 0

        input_df.loc[0, "Data_format_Code"] = 0
        input_df.loc[0, "Data_format_Experiment"] = 0

        if prediction_method == "Experiment-based":
            input_df.loc[0, "Data_format_Experiment"] = 1

        elif prediction_method == "Code-based":
            input_df.loc[0, "Data_format_Code"] = 1

        input_df.loc[0, "Category_Cementless_concrete"] = 0
        input_df.loc[0, "Category_Concrete"] = 0
        input_df.loc[0, "Category_Lightweight_concrete"] = 0

        if category == "Concrete":
            input_df.loc[0, "Category_Concrete"] = 1
        elif category == "Lightweight_concrete":
            input_df.loc[0, "Category_Lightweight_concrete"] = 1
        elif category == "Cementless_concrete":
            input_df.loc[0, "Category_Cementless_concrete"] = 1

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

    # 섬유
        fiber_columns = [
            "Steel_fibre", "Steel_fibre_length",
            "PVA_fibre", "PVA_fibre_length",
            "PP_fibre", "PP_fibre_length",
            "Glass_fibre", "Glass_fibre_length",
            "Basalt_fibre", "Basalt_fibre_length"]

        for col in fiber_columns:
            input_df.loc[0, col] = 0

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

with result_area:
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
