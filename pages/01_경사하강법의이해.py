import streamlit as st
from sympy import symbols, diff, sympify, lambdify
import numpy as np
import plotly.graph_objects as go

st.title("🧠 딥러닝의 핵심: 경사하강법(Gradient Descent) 시각화")

st.markdown("""
**경사하강법**은 인공지능이 "오차"를 줄여가며 정답을 찾아가는 수학적 방법입니다.  
함수를 직접 입력하고, 시작점과 학습률을 바꿔가며 최적점을 찾아가는 과정을 시각적으로 체험해보세요!
""")

# 함수 입력 및 범위, 시작 위치, 학습률, 반복 횟수 조절
func_input = st.text_input("함수 f(x, y)를 입력하세요 (예: x**2 + y**2)", value="x**2 + y**2")
x_min, x_max = st.slider("x 범위", -10, 10, (-5, 5))
y_min, y_max = st.slider("y 범위", -10, 10, (-5, 5))

start_x = st.slider("시작 x 위치", x_min, x_max, 4)
start_y = st.slider("시작 y 위치", y_min, y_max, 4)
learning_rate = st.number_input("학습률(learning rate)", min_value=0.001, max_value=1.0, value=0.2, step=0.01, format="%.3f")
steps = st.slider("경사하강법 반복 횟수", 1, 50, 15)

x, y = symbols('x y')

try:
    # 수식 변환 및 미분
    f = sympify(func_input)
    f_np = lambdify((x, y), f, modules='numpy')
    dx_f = diff(f, x)
    dy_f = diff(f, y)
    dx_np = lambdify((x, y), dx_f, modules='numpy')
    dy_np = lambdify((x, y), dy_f, modules='numpy')

    # 곡면
    X = np.linspace(x_min, x_max, 80)
    Y = np.linspace(y_min, y_max, 80)
    Xs, Ys = np.meshgrid(X, Y)
    Zs = f_np(Xs, Ys)

    # 경사하강법 경로 계산
    path_x = [start_x]
    path_y = [start_y]
    path_z = [f_np(start_x, start_y)]
    curr_x, curr_y = start_x, start_y

    for i in range(steps):
        grad_x = dx_np(curr_x, curr_y)
        grad_y = dy_np(curr_x, curr_y)
        # 경사하강법 업데이트
        next_x = curr_x - learning_rate * grad_x
        next_y = curr_y - learning_rate * grad_y
        next_z = f_np(next_x, next_y)
        path_x.append(next_x)
        path_y.append(next_y)
        path_z.append(next_z)
        curr_x, curr_y = next_x, next_y

    fig = go.Figure()

    # 곡면
    fig.add_trace(go.Surface(x=X, y=Y, z=Zs, opacity=0.6, colorscale='Viridis', showscale=False, name="곡면"))

    # 경로
    fig.add_trace(go.Scatter3d(
        x=path_x, y=path_y, z=path_z,
        mode='lines+markers',
        marker=dict(size=6, color='red'),
        line=dict(color='red', width=4),
        name="경사하강법 경로"
    ))

    # 화살표로 기울기 방향 (각 단계별, 가장 최근 5개만)
    for i in range(1, min(6, len(path_x))):
        fig.add_trace(go.Cone(
            x=[path_x[-i]],
            y=[path_y[-i]],
            z=[path_z[-i]],
            u=[-dx_np(path_x[-i], path_y[-i])*0.4],
            v=[-dy_np(path_x[-i], path_y[-i])*0.4],
            w=[0],
            sizemode="absolute",
            sizeref=0.5,
            colorscale="Reds",
            showscale=False,
            anchor="tail",
            name="Gradient"
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title='x',
            yaxis_title='y',
            zaxis_title='f(x, y)'
        ),
        width=850, height=650,
        margin=dict(l=10, r=10, t=30, b=10),
        title="경사하강법(Gradient Descent) 이동 경로"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        f"""
        **설명:**  
        - 빨간 경로가 인공지능이 오차를 줄이며 최적점(최솟값)으로 이동하는 과정입니다.  
        - **학습률**을 너무 크게 하면 튕기고, 너무 작으면 천천히 접근합니다.  
        - 실제 딥러닝에서 이 원리가 반복적으로 쓰입니다!
        """
    )

except Exception as e:
    st.error(f"수식 오류 또는 지원 불가: {e}")

st.caption("제작: 서울고 송석리 선생님")
