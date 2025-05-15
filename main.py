import streamlit as st
from sympy import symbols, sympify, diff, latex
from sympy.utilities.lambdify import lambdify
import matplotlib.pyplot as plt
import numpy as np

# --- 페이지 기본 설정 ---
st.set_page_config(
    page_title="AI 수학 탐험기 🧠",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 앱 제목 및 소개 ---
st.title("🎓 고등학생을 위한 AI 수학 탐험기")
st.markdown("""
안녕하세요! AI의 핵심 원리 중 하나인 **다변수 함수와 편미분**을 직접 체험해보는 공간입니다.
수학 함수를 직접 입력하고, 그 함수의 편미분 결과를 확인하고, 함수의 3D 모양도 관찰해보세요!
이 모든 과정이 AI가 세상을 이해하고 학습하는 방식과 깊은 관련이 있답니다.
""")

# --- 사이드바: 사용자 입력 ---
st.sidebar.header("🛠️ 함수 및 그래프 설정")
default_function = "2*x**3 + 3*y**3"
user_function_str = st.sidebar.text_input("🧮 함수 f(x, y)를 입력하세요:", value=default_function, help="예: x**2 + y**2, sin(x)*cos(y), exp(-x**2-y**2)")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 3D 그래프 범위 설정")
col_x_min, col_x_max = st.sidebar.columns(2)
x_min = col_x_min.number_input("x 최소값", -10.0, 10.0, -5.0, 0.5)
x_max = col_x_max.number_input("x 최대값", -10.0, 10.0, 5.0, 0.5)

col_y_min, col_y_max = st.sidebar.columns(2)
y_min = col_y_min.number_input("y 최소값", -10.0, 10.0, -5.0, 0.5)
y_max = col_y_max.number_input("y 최대값", -10.0, 10.0, 5.0, 0.5)

if x_min >= x_max:
    st.sidebar.error("x 최소값은 x 최대값보다 작아야 합니다.")
    st.stop()
if y_min >= y_max:
    st.sidebar.error("y 최소값은 y 최대값보다 작아야 합니다.")
    st.stop()

# --- 메인 화면 구성 ---
st.header("🔍 함수 분석 및 시각화")

# 입력된 함수 처리 및 편미분 계산
try:
    x, y = symbols('x y')
    # 사용자 입력 문자열을 SymPy 표현식으로 변환
    # (보안 참고: 실제 서비스에서는 eval() 이나 sympify() 사용 시 매우 주의해야 합니다.)
    f_expr = sympify(user_function_str, locals={'x': x, 'y': y, 'sin': sympy.sin, 'cos': sympy.cos, 'exp': sympy.exp, 'sqrt': sympy.sqrt, 'log': sympy.log, 'pi': sympy.pi})

    # 편미분 계산
    dx_f_expr = diff(f_expr, x)
    dy_f_expr = diff(f_expr, y)

    # 결과 표시를 위한 LaTeX 변환
    st.subheader("1. 입력된 함수")
    st.latex(f"f(x, y) = {latex(f_expr)}")

    st.subheader("2. 편미분 결과")
    st.latex(r"\frac{\partial f}{\partial x} = " + latex(dx_f_expr))
    st.latex(r"\frac{\partial f}{\partial y} = " + latex(dy_f_expr))

except Exception as e:
    st.error(f"함수 입력 오류: {e}")
    st.markdown("""
    올바른 형식으로 함수를 입력해주세요. 사용할 수 있는 변수는 `x`, `y` 이며, 함수는 `sin`, `cos`, `exp`, `sqrt`, `log` 등을 포함할 수 있습니다.
    예시: `x**2 + y**2`, `sin(x) * cos(y)`, `2*x**3 + 3*y**3`, `exp(-x**2 - y**2)`
    """)
    st.stop() # 함수 오류 시 중단

# 3D 그래프 생성
st.subheader("3. 함수의 3D 그래프")
try:
    # SymPy 표현식을 NumPy 계산이 가능한 함수로 변환
    f_lambdify = lambdify((x, y), f_expr, modules=['numpy'])

    # 그래프를 그릴 x, y 값 범위 생성
    x_vals = np.linspace(x_min, x_max, 70) # 점의 개수를 늘리면 더 부드러워집니다.
    y_vals = np.linspace(y_min, y_max, 70)
    X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
    Z_grid = f_lambdify(X_grid, Y_grid)

    # Matplotlib을 사용해 3D 그래프 생성
    fig = plt.figure(figsize=(10, 8)) # 그래프 크기 조절
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X_grid, Y_grid, Z_grid, cmap='viridis', edgecolor='none') # 'viridis', 'plasma', 'magma' 등 다양한 컬러맵 사용 가능

    ax.set_xlabel('X 축', fontsize=10)
    ax.set_ylabel('Y 축', fontsize=10)
    ax.set_zlabel('f(X, Y) 값 (높이)', fontsize=10)
    ax.set_title(f"f(x,y) = {latex(f_expr)} 의 3D 그래프", fontsize=12)
    fig.colorbar(surf, shrink=0.5, aspect=10, label='f(x,y) 값') # 컬러바 추가

    st.pyplot(fig)
    plt.close(fig) # 메모리 관리를 위해 명시적으로 닫기

except Exception as e:
    st.error(f"3D 그래프 생성 중 오류 발생: {e}")
    st.markdown("함수가 3D로 표현하기에 너무 복잡하거나, 지정된 범위에서 정의되지 않았을 수 있습니다 (예: log(0)). 범위를 조절하거나 함수를 확인해주세요.")


# --- AI와의 연관성 설명 ---
st.markdown("---")
st.header("💡 왜 이 개념들이 AI에 중요할까요?")
st.markdown("""
지금까지 살펴본 함수와 미분은 인공지능, 특히 **머신러닝 모델을 학습시키는 데 핵심적인 역할**을 합니다.

1.  **손실 함수 (Loss Function)**:
    * 우리가 입력한 $f(x, y)$와 같은 함수를 AI에서는 '손실 함수' 또는 '비용 함수'라고 부릅니다.
    * 이 함수는 AI 모델의 예측이 실제 정답과 얼마나 차이가 나는지(즉, 얼마나 '틀렸는지')를 나타내는 척도입니다.
    * AI의 목표는 이 손실 함수의 값을 **최소화**하는 것입니다. 마치 우리가 3D 그래프에서 가장 낮은 지점을 찾는 것과 같아요!

2.  **편미분과 경사 (Gradient)**:
    * 우리가 계산한 $\frac{\partial f}{\partial x}$와 $\frac{\partial f}{\partial y}$는 각 변수 방향으로 함수가 얼마나 빠르게 변하는지, 즉 '경사'를 알려줍니다.
    * AI 모델은 이 경사 정보를 사용하여 손실 함수 값이 가장 가파르게 줄어드는 방향으로 자신의 내부 파라미터(변수)들을 조금씩 조정합니다. 이 과정을 **경사 하강법 (Gradient Descent)**이라고 부릅니다.
    * 마치 안개 속에서 산을 내려올 때, 발밑의 경사를 느껴 가장 낮은 곳으로 한 걸음씩 나아가는 것과 비슷합니다.

3.  **모델 학습**:
    * AI는 수많은 데이터를 보면서 이 '경사 하강법' 과정을 반복합니다.
    * 반복할수록 손실 함수의 값은 점점 작아지고, AI 모델은 점점 더 정확한 예측을 할 수 있게 됩니다. 즉, '학습'이 이루어지는 것이죠!

이처럼 우리가 직접 다뤄본 수학적 개념들이 AI가 스스로 학습하고 똑똑해지는 근본적인 원리가 된답니다. 신기하죠? 😊
""")

st.sidebar.markdown("---")
st.sidebar.info("""
**만든이**: AI 도우미 (Google Gemini)
**기반**: 사용자가 제공한 SymPy 스크립트
**기술 스택**: Python, Streamlit, SymPy, Matplotlib, NumPy
""")
