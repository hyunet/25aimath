import streamlit as st
from sympy import symbols, diff, sympify, lambdify
import numpy as np
import plotly.graph_objects as go

st.title("딥러닝 경사하강법 체험 - 다양한 함수와 시점 선택")

# 카메라 각도 라디오 버튼
angle_options = {
    "사선(전체 보기)": dict(x=1.7, y=1.7, z=1.2),
    "정면(x+방향)": dict(x=2.0, y=0.0, z=0.5),
    "정면(y+방향)": dict(x=0.0, y=2.0, z=0.5),
    "위에서 내려다보기": dict(x=0.0, y=0.0, z=3.0),
    "뒤쪽(x-방향)": dict(x=-2.0, y=0.0, z=0.5),
    "옆(y-방향)": dict(x=0.0, y=-2.0, z=0.5)
}
angle_radio = st.radio(
    "그래프 시점(카메라 각도) 선택",
    list(angle_options.keys()),
    index=0,
    horizontal=True
)
camera_eye = angle_options[angle_radio]

# 함수 선택
default_funcs = {
    "볼록 함수 (최적화 쉬움, 예: x²+y²)": "x**2 + y**2",
    "안장점 함수 (최적화 어려움, 예: x²-y²)": "x**2 - y**2",
    "사용자 정의 함수 입력": ""
}
func_options = list(default_funcs.keys())
func_radio = st.radio(
    "함수 유형을 선택하세요.",
    func_options,
    horizontal=True,
    index=0
)

if func_radio == "사용자 정의 함수 입력":
    func_input = st.text_input("함수 f(x, y)를 입력하세요 (예: x**2 + y**2)", value="x**2 + y**2")
else:
    func_input = default_funcs[func_radio]
    st.text_input("함수 f(x, y)", value=func_input, disabled=True)

x_min, x_max = st.slider("x 범위", -10, 10, (-5, 5))
y_min, y_max = st.slider("y 범위", -10, 10, (-5, 5))

start_x = st.slider("시작 x 위치", x_min, x_max, 4)
start_y = st.slider("시작 y 위치", y_min, y_max, 4)
learning_rate = st.number_input("학습률(learning rate)", min_value=0.001, max_value=1.0, value=0.2, step=0.01, format="%.3f")
steps = st.slider("최대 반복 횟수", 1, 50, 15)

x, y = symbols('x y')

# --- 상태 ---
if "gd_path" not in st.session_state or st.session_state.get("last_func", "") != func_input:
    st.session_state.gd_path = [(float(start_x), float(start_y))]
    st.session_state.gd_step = 0
    st.session_state.play = False
    st.session_state.last_func = func_input

def plot_gd(f_np, dx_np, dy_np, x_min, x_max, y_min, y_max, gd_path, min_point, camera_eye):
    X = np.linspace(x_min, x_max, 80)
    Y = np.linspace(y_min, y_max, 80)
    Xs, Ys = np.meshgrid(X, Y)
    Zs = f_np(Xs, Ys)

    fig = go.Figure()
    fig.add_trace(go.Surface(x=X, y=Y, z=Zs, opacity=0.6, colorscale='Viridis', showscale=False))

    px, py = zip(*gd_path)
    pz = [f_np(x, y) for x, y in gd_path]
    fig.add_trace(go.Scatter3d(
        x=px, y=py, z=pz,
        mode='lines+markers+text',
        marker=dict(size=6, color='red'),
        line=dict(color='red', width=4),
        name="경로",
        text=[f"({x:.2f}, {y:.2f})" for x, y in gd_path],
        textposition="top center"
    ))

    arrow_scale = 0.45
    for i in range(-1, -min(11, len(gd_path)), -1):
        gx, gy = gd_path[i]
        gz = f_np(gx, gy)
        grad_x = dx_np(gx, gy)
        grad_y = dy_np(gx, gy)
        fig.add_trace(go.Cone(
            x=[gx], y=[gy], z=[gz],
            u=[-grad_x * arrow_scale],
            v=[-grad_y * arrow_scale],
            w=[0],
            sizemode="absolute", sizeref=0.6,
            colorscale="Blues", showscale=False,
            anchor="tail", name="기울기"
        ))

    min_x, min_y, min_z = min_point
    fig.add_trace(go.Scatter3d(
        x=[min_x], y=[min_y], z=[min_z],
        mode='markers+text',
        marker=dict(size=10, color='limegreen', symbol='diamond'),
        text=["최적점"],
        textposition="bottom center",
        name="최적점"
    ))

    last_x, last_y = gd_path[-1]
    last_z = f_np(last_x, last_y)
    fig.add_trace(go.Scatter3d(
        x=[last_x], y=[last_y], z=[last_z],
        mode='markers+text',
        marker=dict(size=10, color='blue'),
        text=["경사하강법 결과"],
        textposition="top right",
        name="최종점"
    ))

    # 카메라 eye = 항상 사용자가 고른 값!
    fig.update_layout(
        scene=dict(
            xaxis_title='x', yaxis_title='y', zaxis_title='f(x, y)',
            camera=dict(eye=camera_eye)
        ),
        width=800, height=600, margin=dict(l=10, r=10, t=30, b=10),
        title="경사하강법 경로 vs 최적점"
    )
    return fig

col1, col2, col3 = st.columns([1,1,2])
with col1:
    step_btn = st.button("한 스텝 이동")
with col2:
    play_btn = st.button("▶ 전체 실행 (애니메이션)", key="playbtn")
with col3:
    reset_btn = st.button("🔄 초기화", key="resetbtn")

try:
    f = sympify(func_input)
    f_np = lambdify((x, y), f, modules='numpy')
    dx_f = diff(f, x)
    dy_f = diff(f, y)
    dx_np = lambdify((x, y), dx_f, modules='numpy')
    dy_np = lambdify((x, y), dy_f, modules='numpy')

    from scipy.optimize import minimize
    def min_func(vars):
        return f_np(vars[0], vars[1])
    res = minimize(min_func, [start_x, start_y])
    min_x, min_y = res.x
    min_z = f_np(min_x, min_y)

    if reset_btn:
        st.session_state.gd_path = [(float(start_x), float(start_y))]
        st.session_state.gd_step = 0
        st.session_state.play = False

    # 한 스텝 이동
    if step_btn and st.session_state.gd_step < steps:
        curr_x, curr_y = st.session_state.gd_path[-1]
        grad_x = dx_np(curr_x, curr_y)
        grad_y = dy_np(curr_x, curr_y)
        next_x = curr_x - learning_rate * grad_x
        next_y = curr_y - learning_rate * grad_y
        st.session_state.gd_path.append((next_x, next_y))
        st.session_state.gd_step += 1

    # 전체 실행 애니메이션 (항상 camera_eye 고정)
    import time
    if play_btn:
        st.session_state.play = True

    if st.session_state.play and st.session_state.gd_step < steps:
        fig_placeholder = st.empty()
        for i in range(st.session_state.gd_step, steps):
            curr_x, curr_y = st.session_state.gd_path[-1]
            grad_x = dx_np(curr_x, curr_y)
            grad_y = dy_np(curr_x, curr_y)
            next_x = curr_x - learning_rate * grad_x
            next_y = curr_y - learning_rate * grad_y
            st.session_state.gd_path.append((next_x, next_y))
            st.session_state.gd_step += 1
            fig = plot_gd(
                f_np, dx_np, dy_np, x_min, x_max, y_min, y_max,
                st.session_state.gd_path, (min_x, min_y, min_z), camera_eye)
            fig_placeholder.plotly_chart(fig, use_container_width=True, key="animation_chart")
            time.sleep(0.14)
        st.session_state.play = False

    # Step/일반 출력 (항상 camera_eye 고정)
    fig = plot_gd(
        f_np, dx_np, dy_np, x_min, x_max, y_min, y_max,
        st.session_state.gd_path, (min_x, min_y, min_z), camera_eye)
    st.plotly_chart(fig, use_container_width=True, key="main_chart")

    last_x, last_y = st.session_state.gd_path[-1]
    last_z = f_np(last_x, last_y)
    grad_x = dx_np(last_x, last_y)
    grad_y = dy_np(last_x, last_y)
    st.success(
        f"""
        **현재 위치:** ({last_x:.3f}, {last_y:.3f})  
        **현재 함수값:** {last_z:.3f}  
        **현재 기울기:** (∂f/∂x = {grad_x:.3f}, ∂f/∂y = {grad_y:.3f})  
        """
    )
except Exception as e:
    st.error(f"수식 오류 또는 지원 불가: {e}")

st.caption("제작: 서울고 송석리 선생님")
