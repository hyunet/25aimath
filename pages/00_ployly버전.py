import streamlit as st
from sympy import symbols, diff, sympify, lambdify
import numpy as np
import plotly.graph_objects as go
import koreanize_matplotlib

st.title("🎯 AI 미적분: 3D 그래프와 기울기(Gradient) 체험")

st.write("함수를 입력하고, x와 y 위치를 골라 해당 점에서의 기울기를 3D 그래프 위에 시각화해보세요!")

# 함수 입력 및 슬라이더
func_input = st.text_input("함수 f(x, y)를 입력하세요 (예: 2*x**3 + 3*y**3)", value="2*x**3 + 3*y**3")
x_min, x_max = st.slider("x 범위", -10, 10, (-5, 5))
y_min, y_max = st.slider("y 범위", -10, 10, (-5, 5))

x_slider = st.slider("기울기를 볼 x 위치", x_min, x_max, 1)
y_slider = st.slider("기울기를 볼 y 위치", y_min, y_max, 1)

x, y = symbols('x y')
try:
    f = sympify(func_input)
    dx_f = diff(f, x)
    dy_f = diff(f, y)

    st.latex(f"f(x, y) = {f}")
    st.write("**x에 대한 편미분**:")
    st.latex(f"\\frac{{\\partial f}}{{\\partial x}} = {dx_f}")
    st.write("**y에 대한 편미분**:")
    st.latex(f"\\frac{{\\partial f}}{{\\partial y}} = {dy_f}")

    # 함수, 편미분 함수 넘파이 변환
    f_np = lambdify((x, y), f, modules='numpy')
    dx_np = lambdify((x, y), dx_f, modules='numpy')
    dy_np = lambdify((x, y), dy_f, modules='numpy')

    # 3D surface 생성
    X = np.linspace(x_min, x_max, 80)
    Y = np.linspace(y_min, y_max, 80)
    X, Y = np.meshgrid(X, Y)
    Z = f_np(X, Y)

    fig = go.Figure(data=[
        go.Surface(x=X, y=Y, z=Z, colorscale='Viridis', opacity=0.8, name="f(x, y)")
    ])

    # 학생이 선택한 점의 위치와 기울기
    gx = x_slider
    gy = y_slider
    gz = float(f_np(gx, gy))
    gdx = float(dx_np(gx, gy))
    gdy = float(dy_np(gx, gy))

    # 기울기 벡터(화살표) 시각화 (dx, dy, dz)
    # dz: 위로 올라가는 크기, 즉 방향 벡터를 정규화해서 보정(너무 크면 축소)
    arrow_scale = 0.7  # 길이 조정
    dz = gdx * arrow_scale + gdy * arrow_scale  # 대략적인 방향
    dz = dz if abs(dz) > 0.01 else 0.1

    # 화살표(기울기 벡터)
    fig.add_trace(
        go.Cone(
            x=[gx],
            y=[gy],
            z=[gz],
            u=[gdx*arrow_scale],
            v=[gdy*arrow_scale],
            w=[0],  # z축 방향 기울기는 여기선 단순화
            sizemode="absolute",
            sizeref=0.5,
            anchor="tail",
            colorscale="Reds",
            showscale=False,
            name="Gradient"
        )
    )
    # 선택한 점을 빨간 점으로 표시
    fig.add_trace(go.Scatter3d(
        x=[gx], y=[gy], z=[gz],
        mode='markers+text',
        marker=dict(size=6, color='red'),
        text=["기울기 벡터 시작점"],
        textposition="top center"
    ))

    fig.update_layout(
        scene = dict(
            xaxis_title='x',
            yaxis_title='y',
            zaxis_title='f(x, y)'
        ),
        width=700, height=600,
        margin=dict(l=10, r=10, t=30, b=10),
        title="3D 곡면과 기울기(Gradient) 벡터"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        f"""선택한 점 ({gx}, {gy})에서의 gradient(기울기)는  
        $\\nabla f = \\left( \\frac{{\\partial f}}{{\\partial x}} = {gdx:.2f}, \\ \\frac{{\\partial f}}{{\\partial y}} = {gdy:.2f} \\right)$ 입니다.
        """
    )

except Exception as e:
    st.error(f"수식 오류 또는 지원 불가: {e}")

st.caption("제작: 서울고 송석리 선생님 (더 발전된 예시 가능!)")
