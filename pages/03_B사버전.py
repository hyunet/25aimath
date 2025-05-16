# ----------------- 0. 기본 설정 ----------------- o3
import streamlit as st
from sympy import symbols, diff, lambdify, sympify
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize
import time, uuid                                # <— uuid 로 고유 키 생성

st.set_page_config(layout="wide", page_title="경사 하강법 체험 2.0")

# 세션 초기화 ------------------------------------
if "run_uuid" not in st.session_state:
    st.session_state.run_uuid = str(uuid.uuid4())     # 그래프 키 중복 방지용
if "camera_eye" not in st.session_state:
    st.session_state.camera_eye = dict(x=2.0, y=0.0, z=0.5)

# ------------- 1. 교육 모드 선택 ------------------
with st.sidebar:
    st.title("⚙️ 체험 모드")
    mode = st.radio("모드를 고르세요",
                    ("① 탐구 단계(가이드 포함)", "② 자유 실험(전체 UI)"))
    st.markdown("---")

# ------------- 2. 공통 파라미터 -------------------
func_dict = {
    "볼록 (x²+y²)"          : "x**2 + y**2",
    "안장 (0.3x²-0.3y²)"    : "0.3*x**2 - 0.3*y**2",
    "Himmelblau"           : "(x**2 + y - 11)**2 + (x + y**2 - 7)**2",
    "Rastrigin 유사"       : "20 + (x**2 - 10*cos(2*pi*x)) + (y**2 - 10*cos(2*pi*y))",
    "직접 입력"             : ""
}
func_names = list(func_dict.keys())
default_func = func_names[0]

# ---------- 3. 단계별 UI – 탐구 / 실험 ------------
if mode.startswith("①"):
    st.header("👣 1단계 : 함수 선택")
    sel_func = st.selectbox("연습할 함수를 골라 보세요", func_names, index=0)
    if sel_func == "직접 입력":
        user_expr = st.text_input("f(x,y) = ", "x**2 + y**2")
    else:
        user_expr = func_dict[sel_func]

    st.markdown("**Tip 🧑‍🏫** : 볼록 함수는 전역 최소점이 하나라서 학습이 쉽습니다.")
    if st.button("다음 단계 ➡️"): st.session_state["page"] = "step2"
    st.stop()

# 자유 실험(또는 탐구 2단계 이후)
expr = st.session_state.get("user_expr", func_dict[default_func]) \
       if mode.startswith("②") else user_expr
# ---------------- 4. 파라미터 UI -----------------
with st.sidebar:
    st.subheader("📊 그래프·파라미터")
    xrng = st.slider("x 범위", -6.0, 6.0, (-4.0, 4.0), 0.1)
    yrng = st.slider("y 범위", -6.0, 6.0, (-4.0, 4.0), 0.1)
    start_x = st.slider("시작 x", *xrng, 2.0, 0.1)
    start_y = st.slider("시작 y", *yrng, 1.0, 0.1)
    lr      = st.number_input("학습률 α", 0.0001, 1.0, 0.1, 0.001, format="%.4f")
    steps   = st.slider("반복 횟수", 1, 100, 40)
    st.markdown("---")
    if st.button("🔄 매개변수 초기화"):
        st.experimental_rerun()

# ---------------- 5. 수식 준비 --------------------
x_sym, y_sym = symbols("x y")
try:
    f_sym = sympify(expr)
except Exception as e:
    st.error(f"수식 오류: {e}")
    st.stop()
f_np  = lambdify((x_sym, y_sym), f_sym, modules=['numpy'])
dx_np = lambdify((x_sym, y_sym), diff(f_sym, x_sym), modules=['numpy'])
dy_np = lambdify((x_sym, y_sym), diff(f_sym, y_sym), modules=['numpy'])

# ---------------- 6. 경사 하강 실행 ---------------
path, losses = [(start_x, start_y)], [f_np(start_x, start_y)]
for _ in range(steps):
    gx, gy = dx_np(*path[-1]), dy_np(*path[-1])
    nx, ny = path[-1][0] - lr*gx, path[-1][1] - lr*gy
    path.append((nx, ny)); losses.append(f_np(nx, ny))

# ---------------- 7. 3D 그래프 --------------------
px, py = zip(*path)
pz = [f_np(x, y) for x, y in path]

X = np.linspace(*xrng, 80)
Y = np.linspace(*yrng, 80)
Xs, Ys = np.meshgrid(X, Y)
Zs = f_np(Xs, Ys)

fig = go.Figure()
fig.add_trace(go.Surface(x=X, y=Y, z=Zs, colorscale="Viridis",
                         opacity=0.7, showscale=False))
fig.add_trace(go.Scatter3d(x=px, y=py, z=pz,
                           mode="lines+markers",
                           marker=dict(size=5, color="red"),
                           line=dict(color="red", width=3),
                           name="GD 경로"))
fig.update_layout(scene=dict(camera=dict(eye=st.session_state.camera_eye),
                             aspectmode="cube"),
                  margin=dict(l=0, r=0, b=0, t=30),
                  height=600,
                  title="경사 하강법 경로")

chart_key = f"surf_{st.session_state.run_uuid}"
st.plotly_chart(fig, use_container_width=True, key=chart_key)

# --------------- 8. 손실 곡선 ---------------------
st.subheader("📉 손실 값 변화")
st.line_chart(losses)

# --------------- 9. 학습 리플렉션 ------------------
st.markdown("### ✍️ 오늘 배운 점을 한 줄로 기록해 보세요")
reflection = st.text_area("", placeholder="예) 학습률을 너무 크게 하면 발산할 수 있다는 걸 알았다!")
if st.button("저장"):
    st.session_state.reflection = reflection
    st.success("훌륭해요! 기록이 저장되었습니다 🎯")
