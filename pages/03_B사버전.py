# ============================================================
#  경사 하강법 체험 2.0  (최종 전체 코드)
#  작성: 서울고 송석리   |   개선: ChatGPT 교육 버전
# ============================================================

import streamlit as st
from sympy import symbols, diff, sympify, lambdify
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize
import uuid, time

# 0. 페이지 기본 설정 -----------------------------------------------------------
st.set_page_config(layout="wide", page_title="경사 하강법 체험 2.0")

# 1. 세션 상태 초기화 -----------------------------------------------------------
if "run_uuid" not in st.session_state:          # plotly key 중복 방지
    st.session_state.run_uuid = str(uuid.uuid4())

if "camera_eye" not in st.session_state:        # 시점 고정
    st.session_state.camera_eye = dict(x=2.0, y=0.0, z=0.5)

if "page" not in st.session_state:              # 탐구 단계(step1 / step2)
    st.session_state.page = "step1"

# 2. 함수 사전 및 기본값 --------------------------------------------------------
FUNC_DICT = {
    "볼록 (x² + y²)"       : "x**2 + y**2",
    "안장 (0.3x² − 0.3y²)" : "0.3*x**2 - 0.3*y**2",
    "Himmelblau"          : "(x**2 + y - 11)**2 + (x + y**2 - 7)**2",
    "Rastrigin 유사"      : "20 + (x**2 - 10*cos(2*pi*x)) + (y**2 - 10*cos(2*pi*y))",
    "직접 입력"            : ""
}
FUNC_NAMES = list(FUNC_DICT.keys())

# 3. 사이드바 – 모드 선택 -------------------------------------------------------
with st.sidebar:
    st.title("⚙️ 체험 모드")
    mode = st.radio(
        "모드를 고르세요",
        ("① 탐구 단계(가이드 포함)", "② 자유 실험(전체 UI)"),
        index=0
    )
    st.markdown("---")

# -----------------------------------------------------------------------------#
#                ┏━━━━━━━━━━━┓   탐     구     단     계   ┏━━━━━━━━━━━┓       #
# ---------------------------------------------------------------------------- #
if mode.startswith("①"):

    # ───────────────────── STEP 1 ─────────────────────
    if st.session_state.page == "step1":
        st.header("👣 1단계 : 함수 선택")
        sel_func = st.selectbox("연습할 함수를 골라 보세요", FUNC_NAMES, index=0)
        expr_box = st.empty()

        if sel_func == "직접 입력":
            user_expr = expr_box.text_input("f(x, y) = ", "x**2 + y**2")
        else:
            user_expr = FUNC_DICT[sel_func]
            expr_box.text_input("f(x, y) = ", user_expr, disabled=True)

        st.markdown(
            "💡 **Tip** : 볼록 함수는 전역 최소점이 하나라서 학습이 쉽습니다."
        )

        # ▶ 다음 단계 버튼
        if st.button("다음 단계 ➡️", use_container_width=True):
            st.session_state.user_expr = user_expr       # 다음 단계로 전달
            st.session_state.page = "step2"
            st.experimental_rerun()                      # 즉시 2단계로 전환

        st.stop()  # 여기서 렌더링 종료, 2단계 코드 실행 안 됨

    # ───────────────────── STEP 2 ─────────────────────
    # (파라미터 조정 + 시각화)
    if st.session_state.page == "step2":
        st.header("👣 2단계 : 시작점·학습률 조정 및 시각화")

        expr = st.session_state.user_expr

        # --- 파라미터 UI -----------------------------------------------------
        col_l, col_r = st.columns([1.2, 1])
        with col_l:
            xrng = st.slider("x 범위", -6.0, 6.0, (-4.0, 4.0), 0.1)
            yrng = st.slider("y 범위", -6.0, 6.0, (-4.0, 4.0), 0.1)
        with col_r:
            start_x = st.slider("시작 x", xrng[0], xrng[1], 2.0, 0.1)
            start_y = st.slider("시작 y", yrng[0], yrng[1], 1.0, 0.1)
            lr      = st.number_input("학습률 α", 0.0001, 1.0, 0.1, 0.001, format="%.4f")
            steps   = st.slider("반복 횟수", 1, 100, 40)

        # --- 시각화 실행 버튼 -------------------------------------------------
        if st.button("시각화 ▶️", use_container_width=True):
            st.session_state.vis_params = dict(
                expr=expr, xrng=xrng, yrng=yrng,
                start_x=start_x, start_y=start_y,
                lr=lr, steps=steps
            )
            st.session_state.page = "step2_vis"
            st.experimental_rerun()

        st.stop()

    # ─────────────────── STEP 2 (시각화) ───────────────────
    if st.session_state.page == "step2_vis":
        params = st.session_state.vis_params
        expr    = params["expr"]
        xrng    = params["xrng"]
        yrng    = params["yrng"]
        start_x = params["start_x"]
        start_y = params["start_y"]
        lr      = params["lr"]
        steps   = params["steps"]

        st.info("🔄 다시 조정하려면 **사이드바 모드**에서 '탐구 단계'를 선택하세요.")

# -----------------------------------------------------------------------------#
#                     ┏━━━━━━━━━━━━┓   자   유   실   험   ┏━━━━━━━━━━━━┓       #
# ---------------------------------------------------------------------------- #
else:
    st.header("② 자유 실험")

    # --- 함수 선택 -----------------------------------------------------------
    col1, col2 = st.columns([1.2, 1])
    with col1:
        sel_func = st.selectbox("함수 유형", FUNC_NAMES, index=0)
    with col2:
        if sel_func == "직접 입력":
            expr = st.text_input("f(x, y) = ", "x**2 + y**2")
        else:
            expr = FUNC_DICT[sel_func]
            st.text_input("f(x, y) = ", expr, disabled=True)

    # --- 파라미터 UI ---------------------------------------------------------
    xrng = st.slider("x 범위", -6.0, 6.0, (-4.0, 4.0), 0.1)
    yrng = st.slider("y 범위", -6.0, 6.0, (-4.0, 4.0), 0.1)
    start_x = st.slider("시작 x", xrng[0], xrng[1], 2.0, 0.1)
    start_y = st.slider("시작 y", yrng[0], yrng[1], 1.0, 0.1)
    lr      = st.number_input("학습률 α", 0.0001, 1.0, 0.1, 0.001, format="%.4f")
    steps   = st.slider("반복 횟수", 1, 100, 40)

# -----------------------------------------------------------------------------#
#                    ▼▼▼  (공통) 경사 하강 시각화  ▼▼▼                         #
# ---------------------------------------------------------------------------- #
# ※ 위의 조건 분기에서 expr, xrng, … 등의 변수가 정의된 상태

# 4. 수식 준비 -----------------------------------------------------------------
x_sym, y_sym = symbols("x y")
try:
    f_sym = sympify(expr)
except Exception as e:
    st.error(f"수식 오류: {e}")
    st.stop()

modules = ['numpy', {'cos': np.cos, 'sin': np.sin, 'exp': np.exp, 'pi': np.pi}]
f_np  = lambdify((x_sym, y_sym), f_sym, modules=modules)
dx_np = lambdify((x_sym, y_sym), diff(f_sym, x_sym), modules=modules)
dy_np = lambdify((x_sym, y_sym), diff(f_sym, y_sym), modules=modules)

# 5. SciPy로 전역 최소점(참고용) ------------------------------------------------
def try_scipy_min(func, guess):
    try:
        res = minimize(lambda v: func(v[0], v[1]), guess, method="Nelder-Mead")
        if res.success: return res.x, res.fun
    except Exception:
        pass
    return None, None

scipy_pt, scipy_val = try_scipy_min(f_np, [0.0, 0.0])

# 6. 경사 하강 실행 ------------------------------------------------------------
path = [(start_x, start_y)]
losses = [f_np(start_x, start_y)]
for _ in range(steps):
    gx, gy = dx_np(*path[-1]), dy_np(*path[-1])
    nx, ny = path[-1][0] - lr*gx, path[-1][1] - lr*gy
    path.append((nx, ny))
    losses.append(f_np(nx, ny))

# 7. 3D 그래프 ---------------------------------------------------------------
px, py = zip(*path)
pz = [f_np(x, y) for x, y in path]

X = np.linspace(*xrng, 80)
Y = np.linspace(*yrng, 80)
Xs, Ys = np.meshgrid(X, Y)
Zs = f_np(Xs, Ys)

fig = go.Figure()

fig.add_trace(go.Surface(
    x=X, y=Y, z=Zs,
    colorscale="Viridis", opacity=0.7, showscale=False,
    name="f(x, y)"
))

fig.add_trace(go.Scatter3d(
    x=px, y=py, z=pz,
    mode="lines+markers",
    marker=dict(size=5, color="red"),
    line=dict(color="red", width=3),
    name="GD 경로"
))

# SciPy 최적점 표시
if scipy_pt is not None:
    fig.add_trace(go.Scatter3d(
        x=[scipy_pt[0]], y=[scipy_pt[1]], z=[scipy_val],
        mode="markers+text",
        marker=dict(size=8, color="cyan", symbol="diamond"),
        text=["SciPy 최소점"], textposition="bottom center",
        name="SciPy 최소점"
    ))

fig.update_layout(
    scene=dict(camera=dict(eye=st.session_state.camera_eye), aspectmode="cube"),
    height=600, margin=dict(l=0, r=0, b=0, t=40),
    title="경사 하강법 경로"
)

chart_key = f"surf_{st.session_state.run_uuid}"
st.plotly_chart(fig, use_container_width=True, key=chart_key)

# 8. 손실 곡선 -----------------------------------------------------------------
st.subheader("📉 손실 값 변화")
st.line_chart(losses)

# 9. 리플렉션 ------------------------------------------------------------------
st.markdown("### ✍️ 오늘 배운 점을 한 줄로 기록해 보세요")
reflection = st.text_area("", placeholder="예) 학습률을 너무 크게 하면 발산할 수 있다는 걸 알았다!")
if st.button("저장"):
    st.session_state.reflection = reflection
    st.success("기록이 저장되었습니다! 🎯")

# 10. 카메라 시점 고정(드래그 후 유지) -----------------------------------------
# Plotly에서 드래그하면 camera.eye 값이 streamlit에 자동 전달되지 않으므로,
# simple workaround: 마지막 렌더링 시점을 저장
if "last_camera" not in st.session_state:
    st.session_state.last_camera = st.session_state.camera_eye

def _capture_plotly_msg(msg):
    if msg["event"] == "plotly_relayout" and "scene.camera.eye" in msg["data"]:
        st.session_state.camera_eye = msg["data"]["scene.camera.eye"]

# st.plotly_chart는 streamlit-events를 직접 노출하지 않으므로,
# 완전 고정이 필요하면 st.session_state.camera_eye 값을 직접 수정하거나
# plotly.relayout callback을 JS로 삽입해야 함. (교육 목적이므로 간략화)

# ---------------------------------------------------------------------------
# 끝 ‼
