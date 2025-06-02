import streamlit as st
from sympy import symbols, diff, sympify, lambdify
import numpy as np
import plotly.graph_objects as go

st.title("경사하강법 이해를 위한  - 3D 곡면, 절단선, 교점 시각화")

func_input = st.text_input("함수 f(x, y)를 입력하세요 (예: 2*x**3 + 3*y**3)", value="2*x**3 + 3*y**3")
x_min, x_max = st.slider("x 범위", -10, 10, (-5, 5))
y_min, y_max = st.slider("y 범위", -10, 10, (-5, 5))

gx = st.slider("분석할 x 위치", x_min, x_max, 1)
gy = st.slider("분석할 y 위치", y_min, y_max, 1)

x, y = symbols('x y')

try:
    f = sympify(func_input)
    f_np = lambdify((x, y), f, modules='numpy')
    dx_f = diff(f, x)
    dy_f = diff(f, y)
    dx_np = lambdify((x, y), dx_f, modules='numpy')
    dy_np = lambdify((x, y), dy_f, modules='numpy')

    # 전체 곡면
    X = np.linspace(x_min, x_max, 80)
    Y = np.linspace(y_min, y_max, 80)
    Xs, Ys = np.meshgrid(X, Y)
    Zs = f_np(Xs, Ys)

    # y=gy에서 x 방향 단면 (즉, 곡면과 y=b 평면의 교선)
    Z_x = f_np(X, np.full_like(X, gy))
    # x=gx에서 y 방향 단면 (즉, 곡면과 x=a 평면의 교선)
    Z_y = f_np(np.full_like(Y, gx), Y)

    # 기울기 벡터
    gz = float(f_np(gx, gy))
    gdx = float(dx_np(gx, gy))
    gdy = float(dy_np(gx, gy))
    arrow_scale = 0.7

    fig = go.Figure()

    # 1. 곡면
    fig.add_trace(go.Surface(x=X, y=Y, z=Zs, opacity=0.6, colorscale='Viridis', showscale=False, name="곡면"))

    # 2. y=gy 단면선 (x축 평면)
    fig.add_trace(go.Scatter3d(
        x=X, y=[gy]*len(X), z=Z_x,
        mode='lines', line=dict(color='blue', width=7), name="y=b 단면"
    ))
    # 3. x=gx 단면선 (y축 평면)
    fig.add_trace(go.Scatter3d(
        x=[gx]*len(Y), y=Y, z=Z_y,
        mode='lines', line=dict(color='orange', width=7), name="x=a 단면"
    ))

    # 4. 두 곡선의 교차점 (a, b, f(a, b))
    fig.add_trace(go.Scatter3d(
        x=[gx], y=[gy], z=[gz],
        mode='markers+text',
        marker=dict(size=8, color='red'),
        text=["교차점"],
        textposition="top right",
        name="교차점"
    ))

    # 5. 두 단면선의 교선 (즉, x=a, y=b, z의 직선) - 실제론 곡면위 교차점 하나뿐
    # 대신, 기울기 방향 직선(접선) 표시 (dx/dy 기준)
    t = np.linspace(-2, 2, 20)
    tangent_x = gx + t
    tangent_y = gy + t * (gdy / gdx) if gdx != 0 else gy + t
    tangent_z = gz + gdx * t + gdy * t  # 대략적인 tangent 방향
    fig.add_trace(go.Scatter3d(
        x=tangent_x, y=tangent_y, z=tangent_z,
        mode='lines', line=dict(color='red', width=4, dash='dash'), name="기울기(접선) 방향"
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title='x',
            yaxis_title='y',
            zaxis_title='f(x, y)'
        ),
        width=800, height=600,
        margin=dict(l=10, r=10, t=30, b=10),
        title="3D 곡면, 단면선, 교차점, 기울기 방향"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f"""- **파란색 선:** y={gy} 평면의 단면  
        - **주황색 선:** x={gx} 평면의 단면  
        - **빨간 점:** 두 단면선의 교점 (즉, 선택한 점)  
        - **빨간 점선:** 선택점에서의 기울기(gradient) 방향 접선  
        """
    )

except Exception as e:
    st.error(f"수식 오류: {e}")

st.caption("제작: 서울고 송석리 선생님")
