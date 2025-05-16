import streamlit as st
from sympy import symbols, diff, sympify, lambdify
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize

# ------------------------------------------------------------------------------
# 0. 기본 설정 및 공통 스타일
# ------------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="경사 하강법 체험", page_icon="🎢")
st.markdown("""
<style>
    .stAlert p {font-size: 14px;}
    .custom-caption {font-size: 0.9em; color: gray; text-align: center; margin-top: 20px;}
    .highlight {font-weight: bold; color: #FF4B4B;}
    .math-formula {font-family: 'Computer Modern', 'Serif'; font-size: 1.1em; margin: 5px 0;}
</style>
""", unsafe_allow_html=True)
st.title("🎢 딥러닝 경사 하강법 체험 (교육용)")

# ------------------------------------------------------------------------------
# 1. 앱 소개 & 사용 방법 (expander)
# ------------------------------------------------------------------------------
with st.expander("🎯 이 앱의 목표 : "):
    st.markdown("""
0. 경사 하강법(Gradient Descent)은 딥러닝 모델을 학습시키는 핵심 알고리즘입니다. 이 도구를 통해 **직접 체험**하며 이해할 수 있습니다.  
1. 경사 하강법이 **어떻게 함수의 최저점(또는 안장점)을 찾아가는지** 시각적으로 확인  
2. **학습률·시작점·반복 횟수** 등이 최적화 과정에 미치는 영향 탐구  
3. 다양한 형태의 함수(볼록·안장점·복잡한 함수 등)에서 경사 하강법 비교
""")
with st.expander("👇사용 방법 자세히 보기"):
    st.markdown("""
1. **함수 유형** 선택 후, 필요하면 직접 수식을 입력  
2. **그래프 시점**과 **x·y 범위** 조절  
3. **시작 위치, 학습률, 최대 반복** 설정  
4. **🚶 한 스텝 이동** 으로 단계별, **🚀 전체 경로 계산** 으로 빠른 확인  
5. 메인 **3D 그래프**와 **함숫값 변화 그래프**를 함께 관찰
""")

# ------------------------------------------------------------------------------
# 2. 카메라·함수 preset 딕셔너리
# ------------------------------------------------------------------------------
angle_options = {
    "사선(전체 보기)": dict(x=1.7, y=1.7, z=1.2),
    "정면(x+방향)": dict(x=2.0, y=0.0, z=0.5),
    "정면(y+방향)": dict(x=0.0, y=2.0, z=0.5),
    "위에서 내려다보기": dict(x=0.0, y=0.0, z=3.0),
    "뒤쪽(x-방향)": dict(x=-2.0, y=0.0, z=0.5),
    "옆(y-방향)": dict(x=0.0, y=-2.0, z=0.5)
}
default_angle_option_name = "정면(x+방향)"

default_funcs_info = {
    "볼록 함수 (최적화 쉬움, 예: x²+y²)": {
        "func": "x**2 + y**2",
        "desc": "가장 기본적인 형태로, 하나의 전역 최저점을 가집니다.",
        "preset": {"x_range": (-6, 6), "y_range": (-6, 6),
                   "start_x": 5, "start_y": -4, "lr": 0.1, "steps": 25, "camera": "정면(x+방향)"}
    },
    "안장점 함수 (예: 0.3x²-0.3y²)": {
        "func": "0.3*x**2 - 0.3*y**2",
        "desc": "안장점(Saddle Point)을 가집니다.",
        "preset": {"x_range": (-4, 4), "y_range": (-4, 4),
                   "start_x": 2, "start_y": 1, "lr": 0.1, "steps": 40, "camera": "정면(y+방향)"}
    },
    "Himmelblau 함수 (다중 최적점)": {
        "func": "(x**2 + y - 11)**2 + (x + y**2 - 7)**2",
        "desc": "여러 개의 지역 최저점을 가집니다.",
        "preset": {"x_range": (-6, 6), "y_range": (-6, 6),
                   "start_x": 1, "start_y": 1, "lr": 0.01, "steps": 60, "camera": "사선(전체 보기)"}
    },
    "복잡한 함수 (Rastrigin 유사)": {
        "func": "20 + (x**2 - 10*np.cos(2*np.pi*x)) + (y**2 - 10*np.cos(2*np.pi*y))",
        "desc": "매우 많은 지역 최저점을 가지는 비볼록 함수입니다.",
        "preset": {"x_range": (-5, 5), "y_range": (-5, 5),
                   "start_x": 3.5, "start_y": -2.5, "lr": 0.02, "steps": 70, "camera": "사선(전체 보기)"}
    },
    "사용자 정의 함수 입력": {
        "func": "",
        "desc": "파이썬 수식으로 직접 입력하세요.",
        "preset": {"x_range": (-6, 6), "y_range": (-6, 6),
                   "start_x": 5, "start_y": -4, "lr": 0.1, "steps": 25, "camera": "정면(x+방향)"}
    }
}
func_options = list(default_funcs_info.keys())
default_func_type = func_options[0]

# ------------------------------------------------------------------------------
# 3. 세션 상태 초기화
# ------------------------------------------------------------------------------
if "selected_func_type" not in st.session_state:
    st.session_state.selected_func_type = default_func_type
if "selected_camera_option_name" not in st.session_state:
    st.session_state.selected_camera_option_name = default_angle_option_name
if "user_func_input" not in st.session_state:
    st.session_state.user_func_input = "x**2 + y**2"
if "learning_rate_input" not in st.session_state:
    st.session_state.learning_rate_input = 0.1
if "steps_slider" not in st.session_state:
    st.session_state.steps_slider = 25
if "x_min_max_slider" not in st.session_state:
    st.session_state.x_min_max_slider = (-6.0, 6.0)
if "y_min_max_slider" not in st.session_state:
    st.session_state.y_min_max_slider = (-6.0, 6.0)
if "start_x_slider" not in st.session_state:
    st.session_state.start_x_slider = 5.0
if "start_y_slider" not in st.session_state:
    st.session_state.start_y_slider = -4.0
if "gd_path" not in st.session_state:
    st.session_state.gd_path = []
if "gd_step" not in st.session_state:
    st.session_state.gd_step = 0
if "function_values_history" not in st.session_state:
    st.session_state.function_values_history = []
if "is_calculating_all_steps" not in st.session_state:
    st.session_state.is_calculating_all_steps = False
if "current_step_info" not in st.session_state:
    st.session_state.current_step_info = {}
if "messages" not in st.session_state:
    st.session_state.messages = []

# ------------------------------------------------------------------------------
# 4. 사이드바 (설정)
# ------------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ 설정 및 파라미터")

    # 4-1 함수·카메라 선택
    def on_change_func():
        st.session_state.selected_func_type = st.session_state.func_radio_key_widget
        preset = default_funcs_info[st.session_state.selected_func_type]["preset"]
        st.session_state.x_min_max_slider = preset["x_range"]
        st.session_state.y_min_max_slider = preset["y_range"]
        st.session_state.start_x_slider = preset["start_x"]
        st.session_state.start_y_slider = preset["start_y"]
        st.session_state.learning_rate_input = preset["lr"]
        st.session_state.steps_slider = preset["steps"]
        st.session_state.selected_camera_option_name = preset["camera"]
        st.session_state.gd_path = []
        st.session_state.function_values_history = []
        st.session_state.gd_step = 0
        st.session_state.current_step_info = {}

    st.radio("함수 유형", func_options,
             index=func_options.index(st.session_state.selected_func_type),
             key="func_radio_key_widget", on_change=on_change_func)

    st.radio("그래프 시점", list(angle_options.keys()),
             index=list(angle_options.keys()).index(st.session_state.selected_camera_option_name),
             key="camera_angle_radio_key_widget",
             on_change=lambda: setattr(st.session_state,
                                       "selected_camera_option_name",
                                       st.session_state.camera_angle_radio_key_widget))

    selected_func_info = default_funcs_info[st.session_state.selected_func_type]
    st.markdown(f"**선택된 함수 정보:**<br>{selected_func_info['desc']}", unsafe_allow_html=True)

    # 4-2 사용자 정의 함수 입력
    if st.session_state.selected_func_type == "사용자 정의 함수 입력":
        st.text_input("f(x, y) 입력", st.session_state.user_func_input,
                      key="user_func",
                      on_change=lambda: setattr(st.session_state,
                                                "user_func_input",
                                                st.session_state.user_func))

    # 4-3 범위·시작점·학습률·스텝
    st.slider("x 범위", -20.0, 20.0, st.session_state.x_min_max_slider,
              step=0.1, key="x_range",
              on_change=lambda: setattr(st.session_state,
                                        "x_min_max_slider",
                                        st.session_state.x_range))
    st.slider("y 범위", -20.0, 20.0, st.session_state.y_min_max_slider,
              step=0.1, key="y_range",
              on_change=lambda: setattr(st.session_state,
                                        "y_min_max_slider",
                                        st.session_state.y_range))

    st.slider("시작 x", *st.session_state.x_min_max_slider,
              value=st.session_state.start_x_slider,
              step=0.01, key="start_x",
              on_change=lambda: setattr(st.session_state,
                                        "start_x_slider",
                                        st.session_state.start_x))
    st.slider("시작 y", *st.session_state.y_min_max_slider,
              value=st.session_state.start_y_slider,
              step=0.01, key="start_y",
              on_change=lambda: setattr(st.session_state,
                                        "start_y_slider",
                                        st.session_state.start_y))

    st.number_input("학습률 (α)", 0.00001, 5.0,
                    value=st.session_state.learning_rate_input,
                    step=0.0001, format="%.5f",
                    key="lr",
                    on_change=lambda: setattr(st.session_state,
                                              "learning_rate_input",
                                              st.session_state.lr))
    st.slider("최대 반복", 1, 200, st.session_state.steps_slider,
              key="steps",
              on_change=lambda: setattr(st.session_state,
                                        "steps_slider",
                                        st.session_state.steps))

# ------------------------------------------------------------------------------
# 5. 함수·기울기 람다 생성
# ------------------------------------------------------------------------------
x_sym, y_sym = symbols('x y')
func_str = (st.session_state.user_func_input if
            st.session_state.selected_func_type == "사용자 정의 함수 입력"
            else selected_func_info["func"])
try:
    f_sym = sympify(func_str)
except Exception:
    st.error("수식 파싱 오류, 기본 함수 x**2 + y**2 로 대체합니다.")
    f_sym = x_sym**2 + y_sym**2

f_np  = lambdify((x_sym, y_sym), f_sym,
                 modules=['numpy', {'cos': np.cos, 'sin': np.sin,
                                    'exp': np.exp, 'sqrt': np.sqrt, 'pi': np.pi}])
dx_np = lambdify((x_sym, y_sym), diff(f_sym, x_sym),
                 modules=['numpy', {'cos': np.cos, 'sin': np.sin,
                                    'exp': np.exp, 'sqrt': np.sqrt, 'pi': np.pi}])
dy_np = lambdify((x_sym, y_sym), diff(f_sym, y_sym),
                 modules=['numpy', {'cos': np.cos, 'sin': np.sin,
                                    'exp': np.exp, 'sqrt': np.sqrt, 'pi': np.pi}])

# ------------------------------------------------------------------------------
# 6. 메인 영역 – 버튼 + 그래프 + 현재 스텝 정보
# ------------------------------------------------------------------------------
with st.container():
    # 6-1 Button Row ------------------------------------------------------------
    col_btn1, col_btn2, col_btn3 = st.columns([1.2, 1.8, 1])
    with col_btn1:
        step_btn = st.button("🚶 한 스텝 이동", use_container_width=True,
                             disabled=st.session_state.is_calculating_all_steps)
    with col_btn2:
        run_all_btn = st.button("🚀 전체 경로 계산", use_container_width=True,
                                disabled=st.session_state.is_calculating_all_steps)
    with col_btn3:
        reset_btn = st.button("🔄 초기화", use_container_width=True,
                              disabled=st.session_state.is_calculating_all_steps)

    # 6-2 Graph Placeholders ----------------------------------------------------
    graph_placeholder_3d = st.empty()
    graph_placeholder_2d = st.empty()

    # 6-3 Step-info Placeholder -------------------------------------------------
    step_info_placeholder = st.empty()

# ------------------------------------------------------------------------------
# 7. 경사 하강법 유틸리티 함수
# ------------------------------------------------------------------------------
def perform_one_step():
    """gd_path 갱신 & history 기록"""
    if not st.session_state.gd_path:
        st.session_state.gd_path = [(st.session_state.start_x_slider,
                                     st.session_state.start_y_slider)]
        z0 = f_np(*st.session_state.gd_path[0])
        st.session_state.function_values_history = [float(z0)]

    if st.session_state.gd_step >= st.session_state.steps_slider:
        return False

    x, y = st.session_state.gd_path[-1]
    grad_x, grad_y = dx_np(x, y), dy_np(x, y)
    lr = st.session_state.learning_rate_input
    next_x, next_y = x - lr*grad_x, y - lr*grad_y

    st.session_state.gd_path.append((next_x, next_y))
    st.session_state.gd_step += 1
    st.session_state.function_values_history.append(float(f_np(next_x, next_y)))

    st.session_state.current_step_info = {
        "curr_x": x, "curr_y": y, "f_val": f_np(x, y),
        "grad_x": grad_x, "grad_y": grad_y,
        "next_x": next_x, "next_y": next_y
    }
    return True

# ------------------------------------------------------------------------------
# 8. 버튼 핸들러
# ------------------------------------------------------------------------------
if reset_btn:
    preset = default_funcs_info[default_func_type]["preset"]
    st.session_state.selected_func_type = default_func_type
    st.session_state.user_func_input = "x**2 + y**2"
    st.session_state.x_min_max_slider = preset["x_range"]
    st.session_state.y_min_max_slider = preset["y_range"]
    st.session_state.start_x_slider = preset["start_x"]
    st.session_state.start_y_slider = preset["start_y"]
    st.session_state.learning_rate_input = preset["lr"]
    st.session_state.steps_slider = preset["steps"]
    st.session_state.selected_camera_option_name = preset["camera"]
    st.session_state.gd_path = []
    st.session_state.function_values_history = []
    st.session_state.gd_step = 0
    st.session_state.current_step_info = {}
    st.rerun()

if step_btn and not st.session_state.is_calculating_all_steps:
    perform_one_step()
    st.rerun()

if run_all_btn and not st.session_state.is_calculating_all_steps:
    st.session_state.is_calculating_all_steps = True
    for _ in range(st.session_state.steps_slider):
        if not perform_one_step():
            break
    st.session_state.is_calculating_all_steps = False
    st.rerun()

# ------------------------------------------------------------------------------
# 9. 그래프 그리기
# ------------------------------------------------------------------------------
def draw_graphs():
    x_min, x_max = st.session_state.x_min_max_slider
    y_min, y_max = st.session_state.y_min_max_slider
    camera_eye = angle_options[st.session_state.selected_camera_option_name]

    # 3D surface
    X = np.linspace(x_min, x_max, 80)
    Y = np.linspace(y_min, y_max, 80)
    Xg, Yg = np.meshgrid(X, Y)
    Z = f_np(Xg, Yg)
    fig3d = go.Figure(data=[go.Surface(x=X, y=Y, z=Z,
                                       colorscale="Viridis", opacity=0.75,
                                       showscale=False,
                                       contours_z=dict(show=True,
                                                       usecolormap=True))])
    # GD path
    if st.session_state.gd_path:
        px, py = zip(*st.session_state.gd_path)
        pz = [f_np(a, b) for a, b in st.session_state.gd_path]
        fig3d.add_trace(go.Scatter3d(x=px, y=py, z=pz, mode='lines+markers',
                                     marker=dict(size=4, color='red'),
                                     line=dict(color='red', width=4),
                                     name="GD Path"))

    fig3d.update_layout(scene=dict(camera=dict(eye=camera_eye),
                                   aspectmode='cube'),
                        height=550, margin=dict(l=0, r=0, t=40, b=0),
                        title_text="3D 함수 표면 및 경사 하강 경로",
                        title_x=0.5)

    # 2D loss history
    fig2d = go.Figure()
    if st.session_state.function_values_history:
        fig2d.add_trace(go.Scatter(
            y=st.session_state.function_values_history,
            mode='lines+markers', marker=dict(color='green'),
            name="f(x,y)"))
    fig2d.update_layout(height=250, title_text="반복에 따른 함숫값 변화",
                        title_x=0.5, xaxis_title="Step", yaxis_title="f(x,y)",
                        margin=dict(l=20, r=20, t=50, b=20))

    # 현재 스텝 정보 markdown
    info_md = "#### 📌 현재 스텝 정보\n"
    if st.session_state.current_step_info:
        c = st.session_state.current_step_info
        info_md += (f"- 현재 스텝: {st.session_state.gd_step}/{st.session_state.steps_slider}\n"
                    f"- 현재 위치 (x, y): `({c['curr_x']:.3f}, {c['curr_y']:.3f})`\n"
                    f"- f(x,y): `{c['f_val']:.4f}`\n"
                    f"- grad: `({c['grad_x']:.3f}, {c['grad_y']:.3f})`\n"
                    f"- 다음 위치 → `({c['next_x']:.3f}, {c['next_y']:.3f})`")
    else:
        info_md += "경사 하강을 시작해 보세요!"

    return fig3d, fig2d, info_md

fig3d, fig2d, info_md = draw_graphs()
graph_placeholder_3d.plotly_chart(fig3d, use_container_width=True)
graph_placeholder_2d.plotly_chart(fig2d, use_container_width=True)
step_info_placeholder.markdown(info_md, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 10. 학습용 질문
# ------------------------------------------------------------------------------
st.markdown("---")
st.subheader("🤔 더 생각해 볼까요?")
questions = [
    "1. 학습률(α)을 크게/작게 바꾸면 경로가 어떻게 달라지나요?",
    "2. 시작점을 바꾸면 최저점이 항상 같을까요?",
    "3. 안장점 함수에서 경사 하강법은 왜 안장점 근처에서 정체될까요?",
    "4. 지역 최저점이 많은 함수에서 전역 최저점을 어떻게 찾을 수 있을까요?",
    "5. 3D 그래프의 기울기 화살표와 수치로 본 grad 값의 관계는?"
]
for q in questions:
    st.markdown(q)

st.markdown("<p class='custom-caption'>이 도구를 통해 경사 하강법의 원리를 직접 탐구해 보세요!</p>",
            unsafe_allow_html=True)
