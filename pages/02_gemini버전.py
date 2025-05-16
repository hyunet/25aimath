import streamlit as st
from sympy import symbols, diff, sympify, lambdify, re, im
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize
import time

st.set_page_config(layout="wide", page_title="경사 하강법 체험", page_icon="🎢")

# --- 0. 교육적 콘텐츠 및 정적 옵션 정의 ---
st.markdown("""
<style>
    .stAlert p {
        font-size: 14px;
    }
    .custom-caption {
        font-size: 0.9em;
        color: gray;
        text-align: center;
        margin-top: 20px;
    }
    .highlight {
        font-weight: bold;
        color: #FF4B4B;
    }
    .math-formula {
        font-family: 'Computer Modern', 'Serif';
        font-size: 1.1em;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎢 딥러닝 경사 하강법 체험 (교육용)")
st.caption("제작: 서울고 송석리 선생님 | 교육적 개선: Gemini AI & 구글")

st.info("""
**🎯 이 앱의 목표:**
경사 하강법(Gradient Descent)은 딥러닝 모델을 학습시키는 핵심 알고리즘입니다.
이 도구를 통해 다음을 직접 체험하고 이해할 수 있습니다:
1.  경사 하강법이 어떻게 함수의 최저점(또는 안장점)을 찾아가는지 시각적으로 확인합니다.
2.  **학습률(Learning Rate)**, **시작점**, **반복 횟수** 등 주요 파라미터가 최적화 과정에 미치는 영향을 탐구합니다.
3.  다양한 형태의 함수(볼록 함수, 안장점, 복잡한 함수 등)에서 경사 하강법이 어떻게 작동하는지 비교해봅니다.

**👇 사용 방법:**
1.  왼쪽 사이드바에서 **함수 유형**을 선택하고, 필요하면 **함수 수식**을 직접 입력하세요.
2.  **그래프 시점**, **x, y 범위**를 조절하여 원하는 형태로 그래프를 관찰하세요.
3.  **경사 하강법 파라미터**(시작 위치, 학습률, 최대 반복 횟수)를 설정하세요.
4.  **[🚶 한 스텝 이동]** 버튼으로 단계별 과정을, **[▶️ 전체 실행]** 버튼으로 애니메이션을 확인하세요.
5.  메인 화면의 **3D 그래프**와 하단의 **함숫값 변화 그래프**를 함께 관찰하며 학습하세요!
""")


angle_options = {
    "사선(전체 보기)": dict(x=1.7, y=1.7, z=1.2),
    "정면(x+방향)": dict(x=2.0, y=0.0, z=0.5),
    "정면(y+방향)": dict(x=0.0, y=2.0, z=0.5),
    "위에서 내려다보기": dict(x=0.0, y=0.0, z=3.0),
    "뒤쪽(x-방향)": dict(x=-2.0, y=0.0, z=0.5),
    "옆(y-방향)": dict(x=0.0, y=-2.0, z=0.5)
}
default_angle_option_name = "정면(x+방향)"

# 함수 설명 추가
default_funcs_info = {
    "볼록 함수 (최적화 쉬움, 예: x²+y²)": {
        "func": "x**2 + y**2",
        "desc": "가장 기본적인 형태로, 하나의 전역 최저점을 가집니다. 경사 하강법이 안정적으로 최저점을 찾아가는 과정을 관찰하기 좋습니다. <br>🔍 **학습 포인트:** 학습률에 따른 수렴 속도 변화, 시작점에 관계없이 동일한 최저점으로 수렴하는지 확인해보세요.",
        "preset": {"x_range": (-6.0, 6.0), "y_range": (-6.0, 6.0), "start_x": 5.0, "start_y": -4.0, "lr": 0.1, "steps": 25, "camera": "정면(x+방향)"}
    },
    "안장점 함수 (예: 0.3x²-0.3y²)": {
        "func": "0.3*x**2 - 0.3*y**2",
        "desc": "안장점(Saddle Point)을 가집니다. 특정 방향으로는 내려가지만 다른 방향으로는 올라가는 지점입니다. 경사 하강법이 안장점 근처에서 정체될 수 있습니다.<br>🔍 **학습 포인트:** 안장점 주변에서 경사 하강 경로가 어떻게 움직이는지, 학습률이나 시작점에 따라 안장점을 벗어날 수 있는지 관찰하세요.",
        "preset": {"x_range": (-4.0, 4.0), "y_range": (-4.0, 4.0), "start_x": 2.0, "start_y": 1.0, "lr": 0.1, "steps": 40, "camera": "정면(y+방향)"}
    },
    "Himmelblau 함수 (다중 최적점)": {
        "func": "(x**2 + y - 11)**2 + (x + y**2 - 7)**2",
        "desc": "여러 개의 지역 최저점(Local Minima)을 가집니다. 경사 하강법은 시작점에 따라 다른 지역 최저점으로 수렴할 수 있습니다.<br>🔍 **학습 포인트:** 시작점을 다르게 설정했을 때 어떤 최저점으로 수렴하는지, 전역 최저점을 항상 찾을 수 있는지 확인해보세요. (4개의 동일한 최저점이 존재합니다.)",
        "preset": {"x_range": (-6.0, 6.0), "y_range": (-6.0, 6.0), "start_x": 1.0, "start_y": 1.0, "lr": 0.01, "steps": 60, "camera": "사선(전체 보기)"}
    },
    "복잡한 함수 (Rastrigin 유사)": {
        "func": "20 + (x**2 - 10*cos(2*3.14159*x)) + (y**2 - 10*cos(2*3.14159*y))",
        "desc": "매우 많은 지역 최적점을 가지는 비볼록 함수(Non-convex Function)입니다. 경사 하강법이 전역 최저점을 찾기 매우 어려운 예시입니다.<br>🔍 **학습 포인트:** 경사 하강 경로가 쉽게 지역 최저점에 갇히는 현상을 관찰하고, 파라미터 조정으로 이를 개선할 수 있는지 실험해보세요.",
        "preset": {"x_range": (-5.0, 5.0), "y_range": (-5.0, 5.0), "start_x": 3.5, "start_y": -2.5, "lr": 0.02, "steps": 70, "camera": "사선(전체 보기)"}
    },
    "사용자 정의 함수 입력": {
        "func": "", # 사용자가 입력
        "desc": "Python의 `numpy`에서 사용 가능한 연산자(예: `+`, `-`, `*`, `/`, `**`, `cos`, `sin`, `exp`, `sqrt`, `pi`)를 사용하여 자신만의 함수 `f(x,y)`를 정의해보세요. <br>⚠️ **주의:** 복잡하거나 미분 불가능한 지점이 많은 함수는 오류가 발생하거나 경사 하강법이 제대로 작동하지 않을 수 있습니다.",
        "preset": {"x_range": (-6.0, 6.0), "y_range": (-6.0, 6.0), "start_x": 5.0, "start_y": -4.0, "lr": 0.1, "steps": 25, "camera": "정면(x+방향)"}
    }
}
func_options = list(default_funcs_info.keys())
default_func_type = func_options[0]

# --- 1. 모든 UI 제어용 세션 상태 변수 최상단 초기화 ---
if "selected_func_type" not in st.session_state:
    st.session_state.selected_func_type = default_func_type
if "selected_camera_option_name" not in st.session_state:
    st.session_state.selected_camera_option_name = default_funcs_info[default_func_type]["preset"]["camera"]
if "user_func_input" not in st.session_state:
    st.session_state.user_func_input = "x**2 + y**2"
if "current_step_info" not in st.session_state:
    st.session_state.current_step_info = {}
if "function_values_history" not in st.session_state:
    st.session_state.function_values_history = []


def apply_preset_for_func_type(func_type_name):
    preset = default_funcs_info[func_type_name]["preset"]
    st.session_state.x_min_max_slider = preset["x_range"]
    st.session_state.y_min_max_slider = preset["y_range"]
    st.session_state.start_x_slider = preset["start_x"]
    st.session_state.start_y_slider = preset["start_y"]
    st.session_state.selected_camera_option_name = preset["camera"]
    st.session_state.steps_slider = preset["steps"]
    st.session_state.learning_rate_input = preset["lr"]

    # 시작점이 새 범위 내에 있도록 최종 조정
    new_x_min, new_x_max = st.session_state.x_min_max_slider
    new_y_min, new_y_max = st.session_state.y_min_max_slider
    st.session_state.start_x_slider = max(new_x_min, min(new_x_max, st.session_state.start_x_slider))
    st.session_state.start_y_slider = max(new_y_min, min(new_y_max, st.session_state.start_y_slider))
    st.session_state.current_step_info = {} # 프리셋 변경시 스텝 정보 초기화
    st.session_state.function_values_history = [] # 함숫값 기록 초기화


param_keys_to_check = ["x_min_max_slider", "y_min_max_slider", "start_x_slider", "start_y_slider", "learning_rate_input", "steps_slider"]
if not all(key in st.session_state for key in param_keys_to_check):
    apply_preset_for_func_type(st.session_state.selected_func_type)


# --- 2. 현재 설정값 결정 (세션 상태 기반) ---
camera_eye = angle_options[st.session_state.selected_camera_option_name]
if st.session_state.selected_func_type == "사용자 정의 함수 입력":
    func_input_str = st.session_state.user_func_input
else:
    func_input_str = default_funcs_info.get(st.session_state.selected_func_type, {"func": "x**2+y**2"})["func"]

x_min, x_max = st.session_state.x_min_max_slider
y_min, y_max = st.session_state.y_min_max_slider
start_x = st.session_state.start_x_slider
start_y = st.session_state.start_y_slider
learning_rate = st.session_state.learning_rate_input
steps = st.session_state.steps_slider

x_sym, y_sym = symbols('x y')

# --- 3. 경로 관련 세션 상태 초기화 (조건부) ---
# 함수, 시작점, 학습률이 변경되면 경로 초기화
reset_path_condition = (
    "gd_path" not in st.session_state or
    st.session_state.get("last_func_eval", "") != func_input_str or
    st.session_state.get("last_start_x_eval", 0.0) != start_x or
    st.session_state.get("last_start_y_eval", 0.0) != start_y or
    st.session_state.get("last_lr_eval", 0.0) != learning_rate
)

if reset_path_condition:
    st.session_state.gd_path = [(float(start_x), float(start_y))]
    st.session_state.gd_step = 0
    st.session_state.play = False
    st.session_state.last_func_eval = func_input_str
    st.session_state.last_start_x_eval = start_x
    st.session_state.last_start_y_eval = start_y
    st.session_state.last_lr_eval = learning_rate
    st.session_state.animation_camera_eye = camera_eye
    st.session_state.messages = []
    st.session_state.current_step_info = {}
    # 초기 함숫값 기록
    try:
        f_sym_temp = sympify(func_input_str)
        f_np_temp = lambdify((x_sym, y_sym), f_sym_temp, modules=['numpy', {'cos': np.cos, 'sin': np.sin, 'exp': np.exp, 'sqrt': np.sqrt, 'pi': np.pi}])
        initial_z = f_np_temp(float(start_x), float(start_y))
        # 복소수 결과의 실수부만 사용 (교육적 단순화)
        if isinstance(initial_z, complex): initial_z = initial_z.real
        st.session_state.function_values_history = [initial_z] if not np.isnan(initial_z) else []
    except Exception:
        st.session_state.function_values_history = []


# --- 사이드바 UI 구성 ---
with st.sidebar:
    st.header("⚙️ 설정 및 파라미터")

    with st.expander("💡 경사 하강법이란?", expanded=False):
        st.markdown("""
        경사 하강법(Gradient Descent)은 함수의 값을 최소화하는 지점을 찾기 위한 반복적인 최적화 알고리즘입니다. 마치 안개 속에서 산의 가장 낮은 지점을 찾아 내려가는 것과 비슷합니다.

        1.  **현재 위치에서 가장 가파른 경사(기울기, Gradient)를 계산합니다.**
            - 기울기는 각 변수(여기서는 $x, y$)에 대한 편미분 $(\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y})$ 값으로, 함수 값이 가장 빠르게 증가하는 방향을 나타냅니다.
        2.  **기울기의 반대 방향으로 조금 이동합니다.**
            - 함수 값을 줄여야 하므로, 기울기의 반대 방향으로 이동합니다.
            - 이동하는 거리(보폭)는 **학습률(Learning Rate, $\alpha$)**에 의해 조절됩니다.
            <div class='math-formula'>$x_{new} = x_{old} - \alpha \cdot \frac{\partial f}{\partial x}$</div>
            <div class='math-formula'>$y_{new} = y_{old} - \alpha \cdot \frac{\partial f}{\partial y}$</div>
        3.  **이 과정을 반복합니다.**
            - 새로운 위치에서 다시 기울기를 계산하고 이동하는 과정을 반복하여, 함수 값이 더 이상 줄어들지 않는 지점(최저점 또는 안장점 등)에 도달하려고 시도합니다.

        딥러닝에서는 손실 함수(Loss Function)의 값을 최소화하여 모델의 성능을 최적화하는 데 핵심적으로 사용됩니다.
        """)

    with st.expander("📖 주요 파라미터 가이드", expanded=True):
        st.markdown("""
        - **함수 유형**: 다양한 형태의 함수 표면에서 경사 하강법이 어떻게 작동하는지 관찰합니다.
            - <span class='highlight'>볼록 함수</span>: 하나의 최저점을 가져 쉽게 최적화됩니다.
            - <span class='highlight'>안장점 함수</span>: 특정 지점에서 기울기가 0이지만 최저점이 아닐 수 있습니다.
            - <span class='highlight'>Himmelblau/복잡한 함수</span>: 여러 개의 지역 최저점을 가져, 시작점에 따라 다른 결과가 나올 수 있습니다.
        - **그래프 시점**: 3D 그래프를 보는 각도를 조절합니다.
        - **x, y 범위**: 그래프에 표시될 $x, y$ 좌표의 범위를 설정합니다.
        - **시작 $x, y$ 위치**: 경사 하강법 탐색을 시작할 초기 좌표입니다. <span class='highlight'>시작점에 따라 최종 도착점이 달라질 수 있습니다 (특히 비볼록 함수에서).</span>
        - **학습률 ($\alpha$)**: 한 번의 스텝에서 이동하는 거리의 크기를 결정합니다.
            - <span class='highlight'>너무 크면</span>: 최저점을 지나쳐 발산하거나, 진동할 수 있습니다.
            - <span class='highlight'>너무 작으면</span>: 학습 속도가 매우 느려지거나, 지역 최저점에서 벗어나기 어려울 수 있습니다.
        - **최대 반복 횟수**: 경사 하강법을 몇 번이나 반복할지 최대치를 설정합니다.
        """)

    st.subheader("📊 함수 및 그래프 설정")

    def handle_func_type_change():
        new_func_type = st.session_state.func_radio_key_widget
        st.session_state.selected_func_type = new_func_type
        apply_preset_for_func_type(new_func_type)
        # 경로 초기화 트리거 (위의 reset_path_condition에서 처리)
        st.session_state.gd_path = [] # 강제 초기화 신호


    st.radio(
        "그래프 시점(카메라 각도)",
        options=list(angle_options.keys()),
        index=list(angle_options.keys()).index(st.session_state.selected_camera_option_name),
        key="camera_angle_radio_key_widget",
        on_change=lambda: setattr(st.session_state, "selected_camera_option_name", st.session_state.camera_angle_radio_key_widget)
    )
    st.radio(
        "함수 유형",
        func_options,
        index=func_options.index(st.session_state.selected_func_type),
        key="func_radio_key_widget",
        on_change=handle_func_type_change
    )

    # 선택된 함수에 대한 설명 표시
    selected_func_info = default_funcs_info[st.session_state.selected_func_type]
    st.markdown(f"**선택된 함수 정보:**<div style='font-size:0.9em; margin-bottom:10px; padding:8px; background-color:#f0f2f6; border-radius:5px;'>{selected_func_info['desc']}</div>", unsafe_allow_html=True)


    if st.session_state.selected_func_type == "사용자 정의 함수 입력":
        st.text_input("함수 f(x, y) 입력 (예: x**2 + y**2 + sin(x))",
                      value=st.session_state.user_func_input,
                      key="user_func_text_input_key_widget",
                      on_change=lambda: setattr(st.session_state, "user_func_input", st.session_state.user_func_text_input_key_widget)
                      )
    else:
        st.text_input("선택된 함수 f(x, y)", value=selected_func_info["func"], disabled=True)

    st.slider("x 범위", -20.0, 20.0, st.session_state.x_min_max_slider, step=0.1,
              key="x_slider_key_widget",
              on_change=lambda: setattr(st.session_state, "x_min_max_slider", st.session_state.x_slider_key_widget))
    st.slider("y 범위", -20.0, 20.0, st.session_state.y_min_max_slider, step=0.1,
              key="y_slider_key_widget",
              on_change=lambda: setattr(st.session_state, "y_min_max_slider", st.session_state.y_slider_key_widget))

    st.subheader("🔩 경사 하강법 파라미터")
    current_x_min_ui, current_x_max_ui = st.session_state.x_min_max_slider
    current_y_min_ui, current_y_max_ui = st.session_state.y_min_max_slider

    # 시작점 슬라이더의 min/max가 현재 x,y 범위 내에 있도록 보정
    start_x_val = float(st.session_state.start_x_slider)
    start_y_val = float(st.session_state.start_y_slider)
    start_x_val = max(current_x_min_ui, min(current_x_max_ui, start_x_val))
    start_y_val = max(current_y_min_ui, min(current_y_max_ui, start_y_val))


    st.slider("시작 x 위치", float(current_x_min_ui), float(current_x_max_ui), start_x_val, step=0.01, # 정밀도 향상
              key="start_x_key_widget",
              on_change=lambda: setattr(st.session_state, "start_x_slider", st.session_state.start_x_key_widget))
    st.slider("시작 y 위치", float(current_y_min_ui), float(current_y_max_ui), start_y_val, step=0.01, # 정밀도 향상
              key="start_y_key_widget",
              on_change=lambda: setattr(st.session_state, "start_y_slider", st.session_state.start_y_key_widget))

    st.number_input("학습률 (Learning Rate, α)", min_value=0.00001, max_value=5.0, value=st.session_state.learning_rate_input, step=0.0001, format="%.5f", # 정밀도 및 범위 확장
                    key="lr_key_widget",
                    on_change=lambda: setattr(st.session_state, "learning_rate_input", st.session_state.lr_key_widget),
                    help="너무 크면 발산, 너무 작으면 학습이 느립니다. 0.001 ~ 0.5 사이 값을 추천합니다.")
    st.slider("최대 반복 횟수", 1, 200, st.session_state.steps_slider, # 최대 반복 횟수 증가
              key="steps_key_widget",
              on_change=lambda: setattr(st.session_state, "steps_slider", st.session_state.steps_key_widget))

    st.sidebar.subheader("🔬 SciPy 최적화 결과 (참고용)")
    scipy_result_placeholder = st.sidebar.empty()

# --- sympy 함수 파싱 및 numpy 변환 ---
min_point_scipy_coords = None
parse_error = False
try:
    f_sym = sympify(func_input_str)
    # 입력된 함수에 x 또는 y가 없는 경우 에러 처리
    if not (f_sym.has(x_sym) or f_sym.has(y_sym)):
        st.error(f"🚨 함수 정의 오류: 함수에 변수 'x' 또는 'y'가 포함되어야 합니다. 입력: {func_input_str}")
        st.stop()

    f_np = lambdify((x_sym, y_sym), f_sym, modules=['numpy', {'cos': np.cos, 'sin': np.sin, 'exp': np.exp, 'sqrt': np.sqrt, 'pi': np.pi, 'Abs':np.abs}])
    dx_f_sym = diff(f_sym, x_sym)
    dy_f_sym = diff(f_sym, y_sym)
    dx_np = lambdify((x_sym, y_sym), dx_f_sym, modules=['numpy', {'cos': np.cos, 'sin': np.sin, 'exp': np.exp, 'sqrt': np.sqrt, 'pi': np.pi, 'Abs':np.abs, 'sign': np.sign}])
    dy_np = lambdify((x_sym, y_sym), dy_f_sym, modules=['numpy', {'cos': np.cos, 'sin': np.sin, 'exp': np.exp, 'sqrt': np.sqrt, 'pi': np.pi, 'Abs':np.abs, 'sign': np.sign}])

    # SciPy 최적화 (오류 처리 강화)
    try:
        def min_func_scipy(vars_list):
            val = f_np(vars_list[0], vars_list[1])
            if isinstance(val, complex): # 복소수 결과 처리
                val = val.real if np.isreal(val.real) else np.inf # 실수부만, 허수부 있으면 무한대로
            return val if np.isfinite(val) else np.inf # NaN, inf 처리


        potential_starts = [[float(start_x), float(start_y)], [0.0, 0.0]] # 현재 시작점과 원점
        if "Himmelblau" in st.session_state.selected_func_type:
            potential_starts.extend([[3,2], [-2.805, 3.131], [-3.779, -3.283], [3.584, -1.848]])

        best_res = None
        for p_start in potential_starts:
            # 시작점이 현재 x,y 범위 밖에 있으면 SciPy 최적화에서 제외
            if not (x_min <= p_start[0] <= x_max and y_min <= p_start[1] <= y_max):
                continue
            try:
                res_temp = minimize(min_func_scipy, p_start, method='Nelder-Mead', tol=1e-7, options={'maxiter': 500, 'adaptive': True})
                if best_res is None or (res_temp.success and res_temp.fun < best_res.fun) or \
                   (res_temp.success and not best_res.success and np.isfinite(res_temp.fun)):
                    if np.isfinite(res_temp.fun): # 유효한 결과만 저장
                         best_res = res_temp
            except Exception: # 개별 minimize 오류는 무시
                pass

        if best_res and best_res.success and np.isfinite(best_res.fun):
            min_x_sp, min_y_sp = best_res.x
            # SciPy 결과가 현재 x,y 범위를 벗어나면 표시 안 함
            if x_min <= min_x_sp <= x_max and y_min <= min_y_sp <= y_max:
                min_z_sp_val = f_np(min_x_sp, min_y_sp)
                if isinstance(min_z_sp_val, complex): min_z_sp_val = min_z_sp_val.real
                if np.isfinite(min_z_sp_val):
                    min_point_scipy_coords = (min_x_sp, min_y_sp, min_z_sp_val)
                    scipy_result_placeholder.markdown(f"""- **위치 (x, y)**: `({min_x_sp:.3f}, {min_y_sp:.3f})` <br> - **함수 값 f(x,y)**: `{min_z_sp_val:.4f}`""", unsafe_allow_html=True)
                else:
                    scipy_result_placeholder.info("SciPy 최적점의 함수 값이 유효하지 않습니다.")
            else:
                scipy_result_placeholder.info("SciPy 최적점이 현재 그래프 범위 밖에 있습니다.")
        else:
            scipy_result_placeholder.info("SciPy 최적점을 찾지 못했거나, 결과가 유효하지 않습니다.")
    except Exception as e_scipy:
        scipy_result_placeholder.warning(f"SciPy 최적화 중 오류: {str(e_scipy)[:100]}...")

except Exception as e_parse:
    st.error(f"🚨 함수 정의 오류: '{func_input_str}'을(를) 해석할 수 없습니다. 수식을 확인해주세요. (오류: {e_parse})")
    parse_error = True
    # 오류 발생 시 더미 함수로 대체하여 앱 중단 방지
    x_sym, y_sym = symbols('x y')
    f_sym = x_sym**2 + y_sym**2
    f_np = lambdify((x_sym, y_sym), f_sym, 'numpy')
    dx_f_sym = diff(f_sym, x_sym); dy_f_sym = diff(f_sym, y_sym)
    dx_np = lambdify((x_sym, y_sym), dx_f_sym, 'numpy'); dy_np = lambdify((x_sym, y_sym), dy_f_sym, 'numpy')
    st.session_state.gd_path = [(0.,0.)] # 경로도 초기화
    st.session_state.function_values_history = [0.]


# --- 그래프 그리기 함수 ---
def plot_graphs(f_np_func, dx_np_func, dy_np_func, x_min_curr, x_max_curr, y_min_curr, y_max_curr,
                gd_path_curr, function_values_hist_curr, min_point_scipy_curr, current_camera_eye_func, current_step_info_func):
    fig_3d = go.Figure()
    X_plot = np.linspace(x_min_curr, x_max_curr, 80)
    Y_plot = np.linspace(y_min_curr, y_max_curr, 80)
    Xs_plot, Ys_plot = np.meshgrid(X_plot, Y_plot)

    try:
        Zs_plot_raw = f_np_func(Xs_plot, Ys_plot)
        # 복소수 결과 처리: 실수부만 사용, 허수부가 크면 NaN으로 처리하여 시각화 문제 방지
        if np.iscomplexobj(Zs_plot_raw):
            Zs_plot_real = np.real(Zs_plot_raw)
            Zs_plot_imag = np.imag(Zs_plot_raw)
            Zs_plot = np.where(np.abs(Zs_plot_imag) < 1e-9, Zs_plot_real, np.nan) # 허수부가 매우 작으면 실수부 사용
        else:
            Zs_plot = Zs_plot_raw
        Zs_plot = np.nan_to_num(Zs_plot, nan=np.nanmin(Zs_plot)-1 if np.sum(np.isfinite(Zs_plot)) > 0 else 0) # NaN 값 처리
    except Exception:
        Zs_plot = np.zeros_like(Xs_plot) # 오류 시 0으로 채움

    # 3D 표면
    fig_3d.add_trace(go.Surface(x=X_plot, y=Y_plot, z=Zs_plot, opacity=0.75, colorscale='Viridis',
                                contours_z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True),
                                name="함수 표면 f(x,y)", showscale=False))
    # 경사 하강 경로
    if gd_path_curr and len(gd_path_curr) > 0:
        px, py = zip(*gd_path_curr)
        try:
            pz_raw = [f_np_func(pt_x, pt_y) for pt_x, pt_y in gd_path_curr]
            pz = []
            for val in pz_raw:
                if isinstance(val, complex): pz.append(val.real if np.isreal(val.real) else np.nan)
                else: pz.append(val)
            pz = [np.nan_to_num(p, nan=Zs_plot.min()-1 if np.sum(np.isfinite(Zs_plot)) > 0 else 0) for p in pz] # NaN 값 처리
        except Exception:
            pz = [np.nan] * len(px)

        path_texts = [f"S{idx}<br>({pt_x:.2f}, {pt_y:.2f})<br>f={p_z:.2f}" if not np.isnan(p_z) else f"S{idx}<br>({pt_x:.2f}, {pt_y:.2f})" for idx, ((pt_x, pt_y), p_z) in enumerate(zip(gd_path_curr, pz))]

        fig_3d.add_trace(go.Scatter3d(
            x=px, y=py, z=pz, mode='lines+markers+text',
            marker=dict(size=5, color='red', symbol='circle'), line=dict(color='red', width=4), # 경로 강조
            name="경사 하강 경로", text=path_texts, textposition="top right", textfont=dict(size=10, color='black')
        ))

        # 현재 점에서의 기울기 벡터 (Cone으로 표현)
        if len(gd_path_curr) > 0 and not st.session_state.get("play", False): # 애니메이션 중에는 그리지 않음
            last_x_gd, last_y_gd = gd_path_curr[-1]
            try:
                last_z_gd = pz[-1]
                if not np.isnan(last_z_gd): # 유효한 z값일 때만 기울기 표시
                    grad_x_arrow = dx_np_func(last_x_gd, last_y_gd)
                    grad_y_arrow = dy_np_func(last_x_gd, last_y_gd)
                    if isinstance(grad_x_arrow, complex): grad_x_arrow = grad_x_arrow.real
                    if isinstance(grad_y_arrow, complex): grad_y_arrow = grad_y_arrow.real

                    if not (np.isnan(grad_x_arrow) or np.isnan(grad_y_arrow) or np.isinf(grad_x_arrow) or np.isinf(grad_y_arrow)):
                        arrow_scale = 0.3 * learning_rate / 0.1 # 학습률에 따라 화살표 길이 약간 조절 (너무 길어지지 않게)
                        arrow_scale = min(arrow_scale, 0.5) # 최대 길이 제한
                        fig_3d.add_trace(go.Cone(
                            x=[last_x_gd], y=[last_y_gd], z=[last_z_gd + 0.02 * np.abs(last_z_gd) if last_z_gd != 0 else 0.02], # z 살짝 띄우기
                            u=[-grad_x_arrow * arrow_scale], v=[-grad_y_arrow * arrow_scale], w=[0], # 기울기 반대 방향
                            sizemode="absolute", sizeref=0.2, colorscale=[[0, 'magenta'], [1, 'magenta']],
                            showscale=False, anchor="tail", name="현재 기울기 방향",
                            hoverinfo='text', hovertext=f"기울기: ({-grad_x_arrow:.2f}, {-grad_y_arrow:.2f})"
                        ))
            except Exception:
                pass # 기울기 시각화 실패 시 무시

        # GD 최종점 강조
        last_x_gd, last_y_gd = gd_path_curr[-1]
        last_z_gd = pz[-1]
        fig_3d.add_trace(go.Scatter3d(
            x=[last_x_gd], y=[last_y_gd], z=[last_z_gd if not np.isnan(last_z_gd) else (Zs_plot.min() if np.sum(np.isfinite(Zs_plot)) > 0 else 0)],
            mode='markers+text',
            marker=dict(size=8, color='orange', symbol='diamond', line=dict(color='black', width=1.5)),
            text=["현재 위치"], textposition="top left", name="GD 현재 위치"
        ))


    # SciPy 최적점
    if min_point_scipy_curr:
        min_x_sp, min_y_sp, min_z_sp = min_point_scipy_curr
        if not (np.isnan(min_x_sp) or np.isnan(min_y_sp) or np.isnan(min_z_sp)):
            fig_3d.add_trace(go.Scatter3d(
                x=[min_x_sp], y=[min_y_sp], z=[min_z_sp], mode='markers+text',
                marker=dict(size=10, color='cyan', symbol='star', line=dict(color='black',width=1)),
                text=["SciPy 최적점"], textposition="bottom center", name="SciPy 최적점"
            ))

    fig_3d.update_layout(
        scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='f(x, y)',
                   camera=dict(eye=current_camera_eye_func),
                   aspectmode='cube',
                   # Z축 범위 동적 조절 (너무 넓거나 좁아지는 것 방지)
                   zaxis=dict(range=[min(Zs_plot.min(), min(pz) if pz and any(np.isfinite(pz)) else Zs_plot.min()) - abs(Zs_plot.std()*0.2) if np.sum(np.isfinite(Zs_plot)) > 0 else -1,
                                     max(Zs_plot.max(), max(pz) if pz and any(np.isfinite(pz)) else Zs_plot.max()) + abs(Zs_plot.std()*0.2) if np.sum(np.isfinite(Zs_plot)) > 0 else 1])
                  ),
        height=550, margin=dict(l=0, r=0, t=40, b=0),
        title_text="3D 함수 표면 및 경사 하강 경로", title_x=0.5,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # 2D 함숫값 변화 그래프
    fig_2d = go.Figure()
    if function_values_hist_curr and any(np.isfinite(function_values_hist_curr)):
        valid_history = [val for val in function_values_hist_curr if np.isfinite(val)] # NaN/inf 제외
        if valid_history:
            fig_2d.add_trace(go.Scatter(y=valid_history, mode='lines+markers', name='함숫값 f(x,y) 변화',
                                     marker=dict(color='green')))
    fig_2d.update_layout(
        height=250, title_text="반복에 따른 함숫값(손실) 변화", title_x=0.5,
        xaxis_title="반복 횟수 (Step)", yaxis_title="함숫값 f(x,y)",
        margin=dict(l=20, r=20, t=50, b=20)
    )
    if len(valid_history) > 1: # 데이터가 2개 이상일 때만 y축 범위 자동 조절 (약간의 여유)
        min_val = np.min(valid_history)
        max_val = np.max(valid_history)
        padding = (max_val - min_val) * 0.1 if (max_val - min_val) > 1e-6 else 0.1
        fig_2d.update_yaxes(range=[min_val - padding, max_val + padding])


    # 현재 스텝 정보 표시
    current_info_md = "#### 📌 현재 스텝 정보\n"
    if not current_step_info_func:
        current_info_md += "경사 하강을 시작하세요 (한 스텝 또는 전체 실행)."
    else:
        curr_x_info = current_step_info_func.get('curr_x', 'N/A')
        curr_y_info = current_step_info_func.get('curr_y', 'N/A')
        f_val_info = current_step_info_func.get('f_val', 'N/A')
        grad_x_info = current_step_info_func.get('grad_x', 'N/A')
        grad_y_info = current_step_info_func.get('grad_y', 'N/A')
        next_x_info = current_step_info_func.get('next_x', 'N/A')
        next_y_info = current_step_info_func.get('next_y', 'N/A')

        current_info_md += f"- **현재 스텝:** {st.session_state.gd_step}/{steps}\n"
        current_info_md += f"- **현재 위치 $(x, y)$:** `({curr_x_info:.3f}, {curr_y_info:.3f})`\n"
        current_info_md += f"- **현재 함숫값 $f(x,y)$:** `{f_val_info:.4f}`\n"
        current_info_md += f"- **기울기 $(\\frac{{\partial f}}{{\partial x}}, \\frac{{\partial f}}{{\partial y}})$:** `({grad_x_info:.3f}, {grad_y_info:.3f})`\n"
        if st.session_state.gd_step < steps :
             current_info_md += f"- **학습률 $\\alpha$ :** `{learning_rate:.5f}`\n"
             current_info_md += f"- **업데이트:** $x_{{new}} = {curr_x_info:.3f} - ({learning_rate:.4f}) \\times ({grad_x_info:.3f}) = {next_x_info:.3f}$ \n"
             current_info_md += f"            $y_{{new}} = {curr_y_info:.3f} - ({learning_rate:.4f}) \\times ({grad_y_info:.3f}) = {next_y_info:.3f}$ \n"
        current_info_md += f"- **다음 위치 $(x_{{new}}, y_{{new}})$:** `({next_x_info:.3f}, {next_y_info:.3f})`"

    return fig_3d, fig_2d, current_info_md


# --- 메인 페이지 레이아웃 및 나머지 로직 ---
if parse_error: # 함수 파싱 실패 시 여기서 멈춤
    st.warning("함수 오류로 인해 시뮬레이션을 진행할 수 없습니다. 사이드바에서 함수 정의를 수정해주세요.")
    st.stop()

if st.session_state.get("play", False):
    st.info("🎥 애니메이션 실행 중... 현재 카메라 각도로 고정됩니다.")

st.markdown("---")
col_btn1, col_btn2, col_btn3, col_info = st.columns([1.2, 1.5, 1, 2.8]) # 정보 표시 컬럼 추가
with col_btn1: step_btn = st.button("🚶 한 스텝 이동", use_container_width=True, disabled=st.session_state.get("play", False))
with col_btn2: play_btn = st.button("▶️ 전체 실행" if not st.session_state.get("play", False) else "⏹️ 중지", key="playbtn_widget_key", use_container_width=True)
with col_btn3: reset_btn = st.button("🔄 초기화", key="resetbtn_widget_key", use_container_width=True, disabled=st.session_state.get("play", False))

# 현재 스텝 정보 표시를 위한 플레이스홀더
step_info_placeholder = col_info.empty()

graph_placeholder_3d = st.empty()
graph_placeholder_2d = st.empty()


def perform_one_step():
    if st.session_state.gd_step < steps:
        curr_x, curr_y = st.session_state.gd_path[-1]
        try:
            # 현재 값 계산
            current_f_val = f_np(curr_x, curr_y)
            if isinstance(current_f_val, complex): current_f_val = current_f_val.real
            if not np.isfinite(current_f_val):
                st.session_state.messages.append(("error", f"현재 위치 ({curr_x:.2f}, {curr_y:.2f})에서 함수 값이 발산(NaN/inf)하여 중단합니다."))
                st.session_state.play = False
                return False # 진행 불가

            grad_x_val = dx_np(curr_x, curr_y)
            grad_y_val = dy_np(curr_x, curr_y)
            if isinstance(grad_x_val, complex): grad_x_val = grad_x_val.real
            if isinstance(grad_y_val, complex): grad_y_val = grad_y_val.real


            if np.isnan(grad_x_val) or np.isnan(grad_y_val) or np.isinf(grad_x_val) or np.isinf(grad_y_val):
                st.session_state.messages.append(("error", "기울기 계산 결과가 NaN 또는 무한대입니다. 진행을 중단합니다."))
                st.session_state.play = False
                return False # 진행 불가
            else:
                next_x = curr_x - learning_rate * grad_x_val
                next_y = curr_y - learning_rate * grad_y_val

                st.session_state.gd_path.append((next_x, next_y))
                st.session_state.gd_step += 1

                # 다음 스텝의 함숫값 계산 및 기록
                next_f_val = f_np(next_x, next_y)
                if isinstance(next_f_val, complex): next_f_val = next_f_val.real

                if np.isfinite(next_f_val):
                     st.session_state.function_values_history.append(next_f_val)
                else: # 발산하면 이전 값으로 유지 또는 NaN 추가
                     st.session_state.function_values_history.append(np.nan)


                # 현재 스텝 정보 업데이트
                st.session_state.current_step_info = {
                    'curr_x': curr_x, 'curr_y': curr_y, 'f_val': current_f_val,
                    'grad_x': grad_x_val, 'grad_y': grad_y_val,
                    'next_x': next_x, 'next_y': next_y
                }
                return True # 성공적으로 한 스텝 진행

        except Exception as e:
            st.session_state.messages.append(("error", f"스텝 진행 중 오류: {str(e)[:100]}..."))
            st.session_state.play = False
            return False # 진행 불가
    return False # 이미 최대 스텝 도달


if reset_btn:
    st.session_state.selected_func_type = default_func_type
    apply_preset_for_func_type(st.session_state.selected_func_type)
    st.session_state.user_func_input = default_funcs_info["사용자 정의 함수 입력"]["preset"]["func"] if default_funcs_info["사용자 정의 함수 입력"]["preset"]["func"] else "x**2+y**2"

    # 적용된 프리셋 값으로 경로 및 기타 상태 초기화
    current_start_x_on_reset = st.session_state.start_x_slider
    current_start_y_on_reset = st.session_state.start_y_slider
    current_func_input_on_reset = default_funcs_info.get(st.session_state.selected_func_type)["func"] \
                                  if st.session_state.selected_func_type != "사용자 정의 함수 입력" \
                                  else st.session_state.user_func_input

    st.session_state.gd_path = [(float(current_start_x_on_reset), float(current_start_y_on_reset))]
    st.session_state.gd_step = 0
    st.session_state.play = False
    st.session_state.animation_camera_eye = angle_options[st.session_state.selected_camera_option_name]
    st.session_state.messages = []
    st.session_state.current_step_info = {} # 스텝 정보 초기화

    # 초기 함숫값 기록
    try:
        f_sym_reset = sympify(current_func_input_on_reset)
        f_np_reset = lambdify((x_sym, y_sym), f_sym_reset, modules=['numpy', {'cos': np.cos, 'sin': np.sin, 'exp': np.exp, 'sqrt': np.sqrt, 'pi': np.pi}])
        initial_z_reset = f_np_reset(float(current_start_x_on_reset), float(current_start_y_on_reset))
        if isinstance(initial_z_reset, complex): initial_z_reset = initial_z_reset.real
        st.session_state.function_values_history = [initial_z_reset] if np.isfinite(initial_z_reset) else []
    except Exception:
        st.session_state.function_values_history = []


    st.session_state.last_func_eval = current_func_input_on_reset
    st.session_state.last_start_x_eval = current_start_x_on_reset
    st.session_state.last_start_y_eval = current_start_y_on_reset
    st.session_state.last_lr_eval = st.session_state.learning_rate_input
    st.rerun()

if step_btn:
    if not st.session_state.get("play", False): # 애니메이션 중에는 한 스텝 이동 비활성화
        st.session_state.play = False # 혹시 모를 play 상태 해제
        perform_one_step()
        st.rerun()

if play_btn:
    if not st.session_state.get("play", False):
        st.session_state.play = True
        st.session_state.animation_camera_eye = camera_eye # 애니메이션 시작 시 카메라 고정
        st.session_state.messages = [] # 이전 메시지 초기화
    else: # 중지 버튼 눌렀을 때
        st.session_state.play = False
    st.rerun()


# --- 메인 로직 실행 ---
if st.session_state.get("play", False) and st.session_state.gd_step < steps:
    current_animation_cam = st.session_state.get("animation_camera_eye", camera_eye)
    if perform_one_step():
        fig3d, fig2d, info_md = plot_graphs(f_np, dx_np, dy_np, x_min, x_max, y_min, y_max,
                                            st.session_state.gd_path, st.session_state.function_values_history,
                                            min_point_scipy_coords, current_animation_cam, st.session_state.current_step_info)
        graph_placeholder_3d.plotly_chart(fig3d, use_container_width=True)
        graph_placeholder_2d.plotly_chart(fig2d, use_container_width=True)
        step_info_placeholder.markdown(info_md, unsafe_allow_html=True)
        time.sleep(0.25) # 애니메이션 속도 조절 (기존 0.18보다 약간 더 느리게 하여 관찰 용이)
        if st.session_state.gd_step < steps:
            st.rerun()
        else: # 최대 스텝 도달 시 play 해제
            st.session_state.play = False
            st.session_state.play_just_finished = True # 애니메이션 방금 끝났다는 플래그
            st.rerun()
    else: # perform_one_step 실패 (오류 등)
        st.session_state.play = False
        st.rerun() # 오류 메시지 등을 표시하기 위해 rerun
else: # 애니메이션 중이 아닐 때 (정적 표시)
    current_display_cam = camera_eye
    if st.session_state.get("play_just_finished", False): # 애니메이션이 방금 끝났으면
        current_display_cam = st.session_state.get("animation_camera_eye", camera_eye) # 마지막 애니메이션 카메라 유지
        st.session_state.play_just_finished = False # 플래그 해제

    fig3d_static, fig2d_static, info_md_static = plot_graphs(f_np, dx_np, dy_np, x_min, x_max, y_min, y_max,
                                                             st.session_state.gd_path, st.session_state.function_values_history,
                                                             min_point_scipy_coords, current_display_cam, st.session_state.current_step_info)
    graph_placeholder_3d.plotly_chart(fig3d_static, use_container_width=True, key="main_chart_static")
    graph_placeholder_2d.plotly_chart(fig2d_static, use_container_width=True, key="loss_chart_static")
    step_info_placeholder.markdown(info_md_static, unsafe_allow_html=True)

# --- 메시지 및 학습 피드백 표시 ---
temp_messages = st.session_state.get("messages", [])
for msg_type, msg_content in temp_messages:
    if msg_type == "error": st.error(msg_content)
    elif msg_type == "warning": st.warning(msg_content)
    elif msg_type == "success": st.success(msg_content)
    elif msg_type == "info": st.info(msg_content)

if not st.session_state.get("play", False): # 애니메이션 중이 아닐 때만 메시지 초기화 및 최종 상태 분석
    st.session_state.messages = [] # 이미 표시된 메시지는 초기화
    if len(st.session_state.gd_path) > 1: # 한 스텝 이상 진행되었을 때
        last_x_final, last_y_final = st.session_state.gd_path[-1]
        try:
            last_z_final = f_np(last_x_final, last_y_final)
            if isinstance(last_z_final, complex): last_z_final = last_z_final.real
            grad_x_final = dx_np(last_x_final, last_y_final)
            grad_y_final = dy_np(last_x_final, last_y_final)
            if isinstance(grad_x_final, complex): grad_x_final = grad_x_final.real
            if isinstance(grad_y_final, complex): grad_y_final = grad_y_final.real

            grad_norm_final = np.sqrt(grad_x_final**2 + grad_y_final**2) if np.isfinite(grad_x_final) and np.isfinite(grad_y_final) else np.inf

            if not np.isfinite(last_z_final):
                st.error(f"🚨 최종 위치 ({last_x_final:.2f}, {last_y_final:.2f})에서 함수 값이 발산했습니다! (NaN 또는 무한대). 학습률을 줄이거나 시작점을 변경해보세요.")
            elif grad_norm_final < 1e-3 : # 기울기가 매우 작을 때 (수렴 판정 기준 완화)
                 st.success(f"🎉 최적화 완료! 현재 위치 ({last_x_final:.2f}, {last_y_final:.2f}), 함숫값: {last_z_final:.4f}, 기울기 크기: {grad_norm_final:.4f}. \n 기울기가 매우 작아 최저점, 최고점 또는 안장점에 근접한 것으로 보입니다. SciPy 결과와 비교해보세요!")
            elif st.session_state.gd_step >= steps:
                 st.warning(f"⚠️ 최대 반복({steps}회) 도달. 현재 위치 ({last_x_final:.2f}, {last_y_final:.2f}), 함숫값: {last_z_final:.4f}, 기울기 크기: {grad_norm_final:.4f}. \n 아직 기울기가 충분히 작지 않습니다. 반복 횟수를 늘리거나 학습률을 조정해보세요.")

            # 학습률 관련 피드백 (발산 감지)
            if len(st.session_state.function_values_history) > 5: # 최근 5개 함숫값으로 발산 경향 판단
                recent_values = [v for v in st.session_state.function_values_history[-5:] if np.isfinite(v)]
                if len(recent_values) > 1 and np.all(np.diff(recent_values) > 0) and np.abs(recent_values[-1]) > np.abs(recent_values[0]) * 1.5 : # 계속 증가하고, 초기값보다 1.5배 커지면
                     if learning_rate > 0.1: # 비교적 큰 학습률일 때
                        st.warning(f"📈 함숫값이 계속 증가하고 있습니다 (현재: {last_z_final:.2e}). 학습률({learning_rate:.4f})이 너무 클 수 있습니다. 줄여보세요.")
            if learning_rate > 0.8: # 매우 큰 학습률에 대한 직접 경고
                 st.warning(f"🔥 학습률({learning_rate:.4f})이 매우 큽니다! 최적점을 지나쳐 발산하거나 진동할 가능성이 높습니다.")


        except Exception:
            pass # 최종 상태 분석 중 오류는 무시

# 생각할 거리 (교육적 질문)
st.markdown("---")
st.subheader("🤔 더 생각해 볼까요?")
questions = [
    "1. **학습률($\\alpha$)**을 매우 크게 또는 매우 작게 변경하면 경로가 어떻게 달라지나요? 어떤 문제가 발생할 수 있나요?",
    "2. **시작점**을 다르게 설정하면 모든 함수에서 항상 같은 최저점으로 수렴하나요? 그렇지 않다면 이유는 무엇일까요?",
    "3. '안장점 함수'에서 경사 하강법은 왜 안장점 근처에서 오래 머무르거나 특정 방향으로만 움직이는 경향을 보일까요?",
    "4. 'Himmelblau 함수'나 '복잡한 함수'처럼 지역 최저점이 많은 경우, 경사 하강법만으로 **전역 최저점(Global Minimum)**을 항상 찾을 수 있을까요? 어떻게 하면 더 나은 최저점을 찾을 수 있을까요?",
    "5. 현재 스텝 정보의 **기울기 값**과 3D 그래프에 표시된 **기울기 화살표**는 어떤 관계가 있나요?"
]
for q in questions:
    st.markdown(q)

st.markdown("<p class='custom-caption'>이 도구를 통해 경사 하강법의 원리와 특징을 깊이 이해하는 데 도움이 되기를 바랍니다.</p>", unsafe_allow_html=True)
