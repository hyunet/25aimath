import streamlit as st
from sympy import symbols, diff, sympify, latex
from sympy.utilities.lambdify import lambdify
import matplotlib.pyplot as plt
import numpy as np
import koreanize_matplotlib

st.title("🎲 인터랙티브 AI 미적분 실습")

st.write("함수를 직접 입력하고, x, y의 범위도 조절해보세요!")

func_input = st.text_input("함수 f(x, y) 를 입력하세요 (예: 2*x**3 + 3*y**3)", value="2*x**3 + 3*y**3")
x_min, x_max = st.slider("x 범위", -10, 10, (-5, 5))
y_min, y_max = st.slider("y 범위", -10, 10, (-5, 5))

x, y = symbols('x y')
try:
    f = sympify(func_input)
    dx_f = diff(f, x)
    dy_f = diff(f, y)
    
    st.latex(f"f(x, y) = {latex(f)}")
    st.write("**x에 대한 편미분**:")
    st.latex(f"\\frac{{\\partial f}}{{\\partial x}} = {latex(dx_f)}")
    st.write("**y에 대한 편미분**:")
    st.latex(f"\\frac{{\\partial f}}{{\\partial y}} = {latex(dy_f)}")

    X = np.linspace(x_min, x_max, 100)
    Y = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(X, Y)
    F_func = lambdify((x, y), f, modules='numpy')
    Z = F_func(X, Y)

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, alpha=0.7)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')
    ax.set_title(f'f(x, y) = {func_input}')

    st.pyplot(fig)

except Exception as e:
    st.error(f"수식에 오류가 있습니다: {e}")

st.caption("제작: 서울고 송석리 선생님")
