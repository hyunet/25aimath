import streamlit as st
from sympy import symbols, diff, sympify, lambdify
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize
import time

st.set_page_config(layout="wide", page_title="경사 하강법 체험")

st.title("🎢 딥러닝 경사하강법 체험 - 다양한 함수와 시점 선택")
st.caption("제작: 서울고 송석리 선생님 | 개선: Gemini AI")

# --- 교육적 설명 섹션 ---
with st.expander("💡 경사 하강법(Gradient Descent)이란?", expanded=False):
    st.markdown("""
    경사 하강법은 함수의 최솟값을 찾기 위한 기본적인 1차 최적화 알고리즘입니다.
    마치 안개가 자욱한 산을 내려오는 등산객처럼, 현재 위치에서 가장 가파른 경사(기울기)를 따라 한 걸음씩 내려가는 과정을 반복합니다.

    - **기울기 (Gradient, $\\nabla f$)**: 각 지점에서 함수 값이 가장 빠르게 증가하는 방향과 그 정도를 나타내는 벡터입니다. 경사 하강법에서는 이 기울기의 **반대 방향**으로 이동하여 함수 값을 줄여나갑니다.
    - **학습률 (Learning Rate, $\\alpha$)**: 한 번에 얼마나 크게 이동할지(보폭)를 결정하는 값입니다.
        - 너무 크면: 최적점을 지나쳐 멀어지거나(발산), 주변에서 크게 진동할 수 있습니다.
        - 너무 작으면: 최적점까지 수렴하는 데 시간이 매우 오래 걸립니다.
    - **목표**: 반복적인 이동을 통해 기울기가 거의 0인 지점, 즉 더 이상 내려갈 곳이 없는 지점(지역 또는 전역 최적점, 때로는 안장점)에 도달하는 것입니다.
    """)

with st.expander("⚙️ 주요 파라미터 설정 가이드", expanded=False):
    st.markdown(f"""
    - **함수 $f(x, y)$ 선택**: 최적화하려는 대상 함수입니다. 이 앱에서는 두 개의 변수 $x, y$를 갖는 함수를 사용합니다.
        - **볼록 함수 (예: $x^2+y^2$)**: 하나의 전역 최적점(Global Minimum)만을 가집니다. 경사 하강법이 안정적으로 최적점을 찾기 쉬운 이상적인 경우입니다.
        - **안장점 함수 (예: $x^2-y^2$)**: 안장점(Saddle Point)은 특정 방향으로는 극소값처럼 보이지만 다른 방향으로는 극대값처럼 보이는 지점입니다 (말의 안장 모양과 유사). 이 지점에서는 기울기가 0이므로, 경사 하강법이 안장점에 도달하면 학습이 매우 느려지거나 멈춘 것처럼 보일 수 있습니다. 고차원 문제에서 자주 등장합니다.
        - **사용자 정의 함수**: Python 문법에 맞는 수식을 직접 입력하여 실험해볼 수 있습니다. (예: `(x-1)**2 + (y+2)**2 + x*y`)
    - **시작 (x, y) 위치**: 경사 하강법 탐색을 시작하는 초기 지점입니다. 특히 볼록하지 않은 함수에서는 시작 위치에 따라 다른 지역 최적점에 도달할 수 있습니다.
    - **학습률 (Learning Rate, $\\alpha$)**: 매 스텝에서 기울기에 곱해져 이동 거리를 조절합니다. 수식: $x_{{new}} = x_{{old}} - \\alpha \cdot \frac{{\partial f}}{{\partial x}}$
    - **최대 반복 횟수**: 경사 하강법을 몇 번이나 반복할지 최대 한계를 정합니다. 이 횟수 내에 최적점에 도달하지 못할 수도 있습니다.
    - **x, y 범위**: 그래프에 표시될 함수의 범위를 지정합니다.
    """)

# --- UI 컨트롤 섹션 ---
col_params1, col_params2 = st.columns(2)

with col_params1:
    st.subheader("📊 함수 및 그래프 설정")
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
        "복잡한 함수 (다중 지역 최적점 가능성, 예: Rastrigin 유사)": "20 + (x**2 - 10*cos(2*3.14159*x)) + (y**2 - 10*cos(2*3.14159*y))",
        "사용자 정의 함수 입력": ""
    }
    func_options = list(default_funcs.keys())
    func_radio = st.radio(
        "함수 유형을 선택하세요.",
        func_options,
        horizontal=False, # 세로로 변경하여 가독성 향상
        index=0
    )

    if func_radio == "사용자 정의 함수 입력":
        func_input_user = st.text_input("함수 f(x, y)를 입력하세요 (예: x**2 + y**2)", value="x**2 + y**2")
        func_input = func_input_user
    else:
        func_input_default = default_funcs[func_radio]
        st.text_input("선택된 함수 f(x, y)", value=func_input_default, disabled=True)
        func_input = func_input_default

    x_min_max = st.slider("x 범위", -10.0, 10.0, (-5.0, 5.0), step=0.1)
    y_min_max = st.slider("y 범위", -10.0, 10.0, (-5.0, 5.0), step=0.1)
    x_min, x_max = x_min_max
    y_min, y_max = y_min_max

with col_params2:
    st.subheader("⚙️ 경사 하강법 파라미터")
    start_x = st.slider("시작 x 위치", float(x_min), float(x_max), 4.0, step=0.1)
    start_y = st.slider("시작 y 위치", float(y_min), float(y_max), 4.0, step=0.1)
    learning_rate = st.number_input("학습률 (Learning Rate, α)", min_value=0.0001, max_value=1.0, value=0.1, step=0.001, format="%.4f")
    steps = st.slider("최대 반복 횟수", 1, 100, 15, help="경사 하강법을 몇 번 반복할지 설정합니다.")


# 기호 변수 정의
x, y = symbols('x y')

# --- 세션 상태 관리 ---
if "gd_path" not in st.session_state or \
   st.session_state.get("last_func", "") != func_input or \
   st.session_state.get("last_start_x") != start_x or \
   st.session_state.get("last_start_y") != start_y:

    st.session_state.gd_path = [(float(start_x), float(start_y))]
    st.session_state.gd_step = 0
    st.session_state.play = False
    st.session_state.last_func = func_input
    st.session_state.last_start_x = start_x
    st.session_state.last_start_y = start_y
    # 애니메이션 중 카메라 시점 고정을 위한 변수 (현재 UI 선택 값으로 초기화)
    st.session_state.animation_camera_eye = camera_eye
    st.session_state.messages = [] # 메시지 초기화

# --- 그래프 그리기 함수 ---
def plot_gd(f_np, dx_np, dy_np, x_min_plot, x_max_plot, y_min_plot, y_max_plot, gd_path_plot, min_point_scipy, current_camera_eye):
    X = np.linspace(x_min_plot, x_max_plot, 80)
    Y = np.linspace(y_min_plot, y_max_plot, 80)
    Xs, Ys = np.meshgrid(X, Y)
    
    try:
        Zs = f_np(Xs, Ys)
    except Exception as e: # numpy 연산 중 에러 방지 (예: log(0))
        st.error(f"함수 값 계산 중 오류 (그래프 표면): {e}. 함수나 범위를 확인해주세요.")
        # 빈 Zs로 대체하거나 에러 처리
        Zs = np.zeros_like(Xs)


    fig = go.Figure()
    fig.add_trace(go.Surface(x=X, y=Y, z=Zs, opacity=0.7, colorscale='Viridis',
                             contours_z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True),
                             name="함수 표면 f(x,y)", showscale=False))

    px, py = zip(*gd_path_plot)
    try:
        pz = [f_np(pt_x, pt_y) for pt_x, pt_y in gd_path_plot]
    except Exception: # 경로상 점 계산 오류 시 (발산 등)
        pz = [np.nan_to_num(f_np(pt_x, pt_y)) for pt_x, pt_y in gd_path_plot] # NaN을 숫자로 대체 (0 또는 큰 값)


    path_texts = []
    for idx, (pt_x, pt_y) in enumerate(gd_path_plot):
        path_texts.append(f"S{idx}<br>({pt_x:.2f}, {pt_y:.2f})")

    fig.add_trace(go.Scatter3d(
        x=px, y=py, z=pz,
        mode='lines+markers+text',
        marker=dict(size=5, color='red', symbol='o'),
        line=dict(color='red', width=3),
        name="경사 하강 경로",
        text=path_texts,
        textposition="top right",
        textfont=dict(size=10, color='black')
    ))

    # 기울기 화살표 (최근 5개 스텝, 단 첫 스텝은 기울기 없음)
    arrow_scale_factor = 0.3  # 화살표 기본 크기 조절 인자
    num_arrows_to_show = min(5, len(gd_path_plot) - 1)
    if num_arrows_to_show > 0:
        for i in range(num_arrows_to_show):
            # 화살표는 이전 점에서 다음 점으로의 방향이 아니라, 각 점에서의 기울기를 표시
            # gd_path_plot[-(i+2)] 가 (i+1)번째 전 점, gd_path_plot[-(i+1)]이 i번째 전 점
            arrow_start_idx = len(gd_path_plot) - 1 - i -1 # 화살표 시작점의 인덱스 (경로에서 뒤에서 i+2번째 점)
            if arrow_start_idx < 0: continue # 경로가 짧으면 스킵

            gx, gy = gd_path_plot[arrow_start_idx]
            
            try:
                gz = f_np(gx, gy)
                grad_x_arrow = dx_np(gx, gy)
                grad_y_arrow = dy_np(gx, gy)
            except Exception: # 기울기 계산 중 오류 (발산 등)
                continue # 해당 화살표는 그리지 않음

            if not (np.isnan(grad_x_arrow) or np.isnan(grad_y_arrow) or np.isnan(gz)):
                fig.add_trace(go.Cone(
                    x=[gx], y=[gy], z=[gz + 0.02 * np.abs(gz) if gz != 0 else 0.02], # 표면과 겹치지 않게 살짝 띄움
                    u=[-grad_x_arrow * arrow_scale_factor],
                    v=[-grad_y_arrow * arrow_scale_factor],
                    w=[0], # 2D 평면상의 기울기
                    sizemode="absolute", sizeref=0.25, # 화살표 두께
                    colorscale=[[0, 'magenta'], [1, 'magenta']], showscale=False, # 단색 Magenta
                    anchor="tail",
                    name=f"기울기 S{arrow_start_idx}" if i == 0 else "", # 최근 기울기만 범례 표시
                    hoverinfo='skip'
                ))
    
    # SciPy로 찾은 최적점 표시
    if min_point_scipy:
        min_x_sp, min_y_sp, min_z_sp = min_point_scipy
        fig.add_trace(go.Scatter3d(
            x=[min_x_sp], y=[min_y_sp], z=[min_z_sp],
            mode='markers+text',
            marker=dict(size=10, color='cyan', symbol='diamond'),
            text=["SciPy 최적점"], textposition="bottom center", name="SciPy 최적점"
        ))

    # 경사하강법 최종점 표시
    last_x_gd, last_y_gd = gd_path_plot[-1]
    try:
        last_z_gd = f_np(last_x_gd, last_y_gd)
    except Exception:
        last_z_gd = np.nan # 계산 불가 시

    fig.add_trace(go.Scatter3d(
        x=[last_x_gd], y=[last_y_gd], z=[last_z_gd if not np.isnan(last_z_gd) else Zs.min()], # NaN이면 그래프 최소값에 표시
        mode='markers+text',
        marker=dict(size=10, color='blue', symbol='x'),
        text=["GD 최종점"], textposition="top left", name="GD 최종점"
    ))


    fig.update_layout(
        scene=dict(
            xaxis_title='x', yaxis_title='y', zaxis_title='f(x, y)',
            camera=dict(eye=current_camera_eye),
            aspectmode='cube' # 비율 고정하여 왜곡 방지
        ),
        width=None, height=700, margin=dict(l=10, r=10, t=30, b=10),
        title_text="경사 하강법 경로 및 함수 표면", title_x=0.5,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# --- 제어 버튼 ---
st.markdown("---")
col_btn1, col_btn2, col_btn3 = st.columns([1.5, 2, 1]) # 버튼 크기 조절
with col_btn1:
    step_btn = st.button("🚶 한 스텝 이동", use_container_width=True)
with col_btn2:
    play_btn = st.button("🎥 전체 실행 (애니메이션)", key="playbtn", use_container_width=True)
with col_btn3:
    reset_btn = st.button("🔄 초기화", key="resetbtn", use_container_width=True)

# --- 메인 로직 ---
graph_placeholder = st.empty() # 그래프를 표시할 영역
info_cols = st.columns(3) # 정보 표시 컬럼

try:
    f_sym = sympify(func_input)
    f_np = lambdify((x, y), f_sym, modules=['numpy', {'cos': np.cos, 'sin': np.sin, 'exp': np.exp, 'sqrt': np.sqrt}]) # numpy 모듈 및 추가 함수 지원
    dx_f_sym = diff(f_sym, x)
    dy_f_sym = diff(f_sym, y)
    dx_np = lambdify((x, y), dx_f_sym, modules=['numpy', {'cos': np.cos, 'sin': np.sin, 'exp': np.exp, 'sqrt': np.sqrt}])
    dy_np = lambdify((x, y), dy_f_sym, modules=['numpy', {'cos': np.cos, 'sin': np.sin, 'exp': np.exp, 'sqrt': np.sqrt}])

    # SciPy를 사용한 최적점 계산 (참고용)
    min_point_scipy_coords = None
    try:
        def min_func_scipy(vars_list):
            return f_np(vars_list[0], vars_list[1])
        
        # 최적화 시작점을 다양하게 시도 (예: (0,0), 현재 시작점)
        # 이는 복잡한 함수에서 더 나은 전역 최적점을 찾는데 도움을 줄 수 있음
        potential_starts = [[0,0], [start_x, start_y]]
        best_res = None
        for p_start in potential_starts:
            res_temp = minimize(min_func_scipy, p_start, method='Nelder-Mead', tol=1e-6)
            if best_res is None or res_temp.fun < best_res.fun:
                best_res = res_temp
        
        if best_res and best_res.success:
            min_x_sp, min_y_sp = best_res.x
            min_z_sp = f_np(min_x_sp, min_y_sp)
            min_point_scipy_coords = (min_x_sp, min_y_sp, min_z_sp)
    except Exception as e:
        st.sidebar.warning(f"SciPy 최적점 계산 중 오류: {e}")


    if reset_btn:
        st.session_state.gd_path = [(float(start_x), float(start_y))]
        st.session_state.gd_step = 0
        st.session_state.play = False
        st.session_state.animation_camera_eye = camera_eye # 초기화 시 카메라 각도 업데이트
        st.session_state.messages = []
        st.rerun() # 상태 초기화 후 즉시 UI 반영

    # 한 스텝 이동 로직
    if step_btn and st.session_state.gd_step < steps:
        curr_x, curr_y = st.session_state.gd_path[-1]
        try:
            grad_x_val = dx_np(curr_x, curr_y)
            grad_y_val = dy_np(curr_x, curr_y)

            if np.isnan(grad_x_val) or np.isnan(grad_y_val):
                st.session_state.messages.append(("error", "기울기 계산 결과가 NaN입니다. 발산 가능성이 있습니다."))
            else:
                next_x = curr_x - learning_rate * grad_x_val
                next_y = curr_y - learning_rate * grad_y_val
                st.session_state.gd_path.append((next_x, next_y))
                st.session_state.gd_step += 1
        except Exception as e:
            st.session_state.messages.append(("error", f"스텝 진행 중 오류: {e}"))

    # 전체 실행 애니메이션 로직
    if play_btn:
        st.session_state.play = True
        st.session_state.animation_camera_eye = camera_eye # 애니메이션 시작 시 카메라 각도 고정
        st.session_state.messages = [] # 애니메이션 시작 시 이전 메시지 클리어

    if st.session_state.play and st.session_state.gd_step < steps:
        # 애니메이션 루프에서는 고정된 카메라 각도 사용
        # st.session_state에 animation_camera_eye가 없을 경우 현재 UI 카메라 사용
        current_animation_cam = st.session_state.get("animation_camera_eye", camera_eye)

        curr_x_anim, curr_y_anim = st.session_state.gd_path[-1]
        try:
            grad_x_anim = dx_np(curr_x_anim, curr_y_anim)
            grad_y_anim = dy_np(curr_x_anim, curr_y_anim)

            if np.isnan(grad_x_anim) or np.isnan(grad_y_anim):
                st.session_state.messages.append(("error", "애니메이션 중 기울기 NaN. 중단합니다."))
                st.session_state.play = False
            else:
                next_x_anim = curr_x_anim - learning_rate * grad_x_anim
                next_y_anim = curr_y_anim - learning_rate * grad_y_anim
                st.session_state.gd_path.append((next_x_anim, next_y_anim))
                st.session_state.gd_step += 1
                
                fig_anim = plot_gd(f_np, dx_np, dy_np, x_min, x_max, y_min, y_max,
                                   st.session_state.gd_path, min_point_scipy_coords, current_animation_cam)
                graph_placeholder.plotly_chart(fig_anim, use_container_width=True) # key 제거
                time.sleep(0.15) # 애니메이션 속도
                if st.session_state.gd_step < steps: # 마지막 스텝이 아니면 다시 실행
                     st.rerun() 
                else: # 마지막 스텝이면 play 상태 해제
                     st.session_state.play = False
        except Exception as e:
            st.session_state.messages.append(("error", f"애니메이션 중 오류: {e}"))
            st.session_state.play = False
            
    else: # 애니메이션 중이 아닐 때 (일반 업데이트 또는 애니메이션 종료 후)
        current_display_cam = camera_eye # 일반 표시는 현재 UI 카메라 설정 따름
        if st.session_state.get("play_just_finished", False): # 애니메이션이 방금 끝났다면
            current_display_cam = st.session_state.get("animation_camera_eye", camera_eye) # 애니메이션 마지막 카메라 유지
            st.session_state.play_just_finished = False

        fig_static = plot_gd(f_np, dx_np, dy_np, x_min, x_max, y_min, y_max,
                             st.session_state.gd_path, min_point_scipy_coords, current_display_cam)
        graph_placeholder.plotly_chart(fig_static, use_container_width=True, key="main_chart_static")


    # --- 현재 상태 정보 표시 ---
    last_x_info, last_y_info = st.session_state.gd_path[-1]
    try:
        last_z_info = f_np(last_x_info, last_y_info)
        current_grad_x = dx_np(last_x_info, last_y_info)
        current_grad_y = dy_np(last_x_info, last_y_info)
        grad_norm = np.sqrt(current_grad_x**2 + current_grad_y**2)
    except Exception: # 계산 오류 시 (예: func_input이 비어있거나 잘못된 초기 상태)
        last_z_info = np.nan
        current_grad_x = np.nan
        current_grad_y = np.nan
        grad_norm = np.nan


    prev_z_info = np.nan
    if len(st.session_state.gd_path) > 1:
        prev_x_path, prev_y_path = st.session_state.gd_path[-2]
        try:
            prev_z_info = f_np(prev_x_path, prev_y_path)
        except Exception:
            prev_z_info = np.nan

    delta_val = last_z_info - prev_z_info if not (np.isnan(last_z_info) or np.isnan(prev_z_info)) else np.nan

    with info_cols[0]:
        st.metric(label="현재 스텝", value=f"{st.session_state.gd_step} / {steps}")
        st.markdown(f"**위치 (x, y)**: `({last_x_info:.3f}, {last_y_info:.3f})`")
    with info_cols[1]:
        st.metric(label="현재 함수 값 f(x,y)", value=f"{last_z_info:.4f}" if not np.isnan(last_z_info) else "N/A",
                  delta=f"{delta_val:.4f}" if not np.isnan(delta_val) else None,
                  delta_color="inverse" if not np.isnan(delta_val) and delta_val < 0 else ("normal" if not np.isnan(delta_val) and delta_val > 0 else "off"))
    with info_cols[2]:
        st.metric(label="기울기 크기 ||∇f||", value=f"{grad_norm:.4f}" if not np.isnan(grad_norm) else "N/A")
        st.markdown(f"**∂f/∂x**: `{current_grad_x:.3f}`\n**∂f/∂y**: `{current_grad_y:.3f}`")

    # 메시지 출력
    for msg_type, msg_content in st.session_state.get("messages", []):
        if msg_type == "error":
            st.error(msg_content)
        elif msg_type == "warning":
            st.warning(msg_content)
        elif msg_type == "success":
            st.success(msg_content)
    st.session_state.messages = [] # 메시지 표시 후 초기화

    # 최종 상태 판단 메시지 (애니메이션 중이 아닐 때만)
    if not st.session_state.play:
        if np.isnan(last_z_info) or np.isinf(last_z_info):
            st.error("🚨 함수 값이 발산했습니다! (NaN 또는 무한대) 학습률을 줄이거나 시작점, 함수를 변경해보세요.")
        elif st.session_state.gd_step >= steps and grad_norm > 1e-2: # 임계값은 조절 가능
            st.warning(f"⚠️ 최대 반복 횟수({steps})에 도달했지만, 기울기 크기({grad_norm:.4f})가 아직 충분히 작지 않습니다. 최적점에 더 가까워지려면 반복 횟수를 늘리거나 학습률/함수를 조절해보세요.")
        elif grad_norm < 1e-2 and not (np.isnan(grad_norm) or np.isinf(grad_norm)):
            st.success(f"🎉 기울기 크기({grad_norm:.4f})가 매우 작아져 최적점 또는 안장점에 근접한 것으로 보입니다!")
        
except SyntaxError:
    st.error(f"🚨 함수 수식에 문법 오류가 있습니다: '{func_input}'. Python 수학 표현식을 확인해주세요 (예: x**2 + sin(y)).")
except Exception as e:
    st.error(f"🚨 처리 중 오류 발생: {e}. 함수 수식이나 입력값을 확인해주세요.")

# --- SciPy 최적점 정보 사이드바 표시 ---
if min_point_scipy_coords:
    st.sidebar.subheader("🔬 SciPy 최적화 결과 (참고용)")
    st.sidebar.markdown(f"""
    `scipy.optimize.minimize` (Nelder-Mead)를 사용해 찾은 (지역) 최적점 후보:
    - **위치 (x, y)**: `({min_point_scipy_coords[0]:.3f}, {min_point_scipy_coords[1]:.3f})`
    - **함수 값 f(x,y)**: `{min_point_scipy_coords[2]:.4f}`
    ---
    *경사 하강법의 목표는 이와 같은 최적점에 도달하는 것입니다. SciPy는 다른 최적화 기법을 사용하며, 찾은 점이 전역 최적점이 아닐 수도 있습니다.*
    """)
else:
    st.sidebar.info("SciPy 최적점 정보를 계산할 수 없었습니다 (선택된 함수에 따라 다름).")
