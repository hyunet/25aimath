import streamlit as st
from sympy import symbols, diff, sympify, lambdify
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize
import time

st.set_page_config(layout="wide", page_title="경사 하강법 체험")

st.title("🎢 딥러닝 경사하강법 체험")
st.caption("제작: 서울고 송석리 선생님 | 개선: Gemini AI")

# --- 0. 정적 옵션 정의 ---
angle_options = {
    "사선(전체 보기)": dict(x=1.7, y=1.7, z=1.2),
    "정면(x+방향)": dict(x=2.0, y=0.0, z=0.5), 
    "정면(y+방향)": dict(x=0.0, y=2.0, z=0.5), # 안장점 함수 기본 시점으로 사용
    "위에서 내려다보기": dict(x=0.0, y=0.0, z=3.0),
    "뒤쪽(x-방향)": dict(x=-2.0, y=0.0, z=0.5),
    "옆(y-방향)": dict(x=0.0, y=-2.0, z=0.5)
}
default_angle_option_name = "정면(x+방향)"

default_funcs = {
    "볼록 함수 (최적화 쉬움, 예: x²+y²)": "x**2 + y**2",
    # 요청사항 2: 안장점 함수 변경 및 이름 수정
    "안장점 함수 (예: 0.3x²-0.3y²)": "0.3*x**2 - 0.3*y**2", 
    "Himmelblau 함수 (다중 최적점)": "(x**2 + y - 11)**2 + (x + y**2 - 7)**2",
    "복잡한 함수 (Rastrigin 유사)": "20 + (x**2 - 10*cos(2*3.14159*x)) + (y**2 - 10*cos(2*3.14159*y))",
    "사용자 정의 함수 입력": ""
}
func_options = list(default_funcs.keys())
default_func_type = func_options[0] 

default_x_range_convex = (-6.0, 6.0)
default_y_range_convex = (-6.0, 6.0)
default_start_x_convex = 5.0
default_start_y_convex = -4.0
default_lr_convex = 0.1
default_steps_convex = 25

# --- 1. 모든 UI 제어용 세션 상태 변수 최상단 초기화 ---
if "selected_camera_option_name" not in st.session_state:
    st.session_state.selected_camera_option_name = default_angle_option_name
if "selected_func_type" not in st.session_state:
    st.session_state.selected_func_type = default_func_type
if "user_func_input" not in st.session_state:
    st.session_state.user_func_input = "x**2 + y**2" 

def apply_preset_for_func_type(func_type_name, is_initial_load=False):
    current_x_range = st.session_state.get("x_min_max_slider", default_x_range_convex)
    current_y_range = st.session_state.get("y_min_max_slider", default_y_range_convex)

    if func_type_name == "안장점 함수 (예: 0.3x²-0.3y²)": # 수정된 함수 이름
        st.session_state.x_min_max_slider = (-4.0, 4.0) 
        st.session_state.y_min_max_slider = (-4.0, 4.0) 
        st.session_state.start_x_slider = 2.0 # 안장점 추천 시작 x
        st.session_state.start_y_slider = 1.0 # 안장점 추천 시작 y
        st.session_state.selected_camera_option_name = "정면(y+방향)" # 요청사항 1: 카메라 각도 변경
        st.session_state.steps_slider = 40 # 스텝 수 조정
        st.session_state.learning_rate_input = 0.1 # 학습률 조정
    elif func_type_name == "Himmelblau 함수 (다중 최적점)":
        st.session_state.x_min_max_slider = (-6.0, 6.0) 
        st.session_state.y_min_max_slider = (-6.0, 6.0) 
        st.session_state.start_x_slider = 1.0
        st.session_state.start_y_slider = 1.0
        st.session_state.selected_camera_option_name = "사선(전체 보기)"
        st.session_state.steps_slider = 60
        st.session_state.learning_rate_input = 0.01
    elif func_type_name == "복잡한 함수 (Rastrigin 유사)":
        st.session_state.x_min_max_slider = (-5.0, 5.0) 
        st.session_state.y_min_max_slider = (-5.0, 5.0) 
        st.session_state.start_x_slider = 3.5
        st.session_state.start_y_slider = -2.5
        st.session_state.selected_camera_option_name = "사선(전체 보기)"
        st.session_state.steps_slider = 70
        st.session_state.learning_rate_input = 0.02
    elif func_type_name == "볼록 함수 (최적화 쉬움, 예: x²+y²)":
        st.session_state.x_min_max_slider = default_x_range_convex 
        st.session_state.y_min_max_slider = default_y_range_convex 
        st.session_state.start_x_slider = default_start_x_convex
        st.session_state.start_y_slider = default_start_y_convex
        st.session_state.selected_camera_option_name = default_angle_option_name
        st.session_state.steps_slider = default_steps_convex
        st.session_state.learning_rate_input = default_lr_convex
    
    # "사용자 정의 함수 입력"의 경우, is_initial_load가 True일 때만 기본값으로 설정하고,
    # 사용자가 함수 유형을 변경하여 넘어온 경우에는 이전 사용자 정의 값을 유지하도록 할 수 있음.
    # 여기서는 일단 다른 파라미터는 볼록 함수 기준으로 설정.
    elif func_type_name == "사용자 정의 함수 입력" and is_initial_load:
        st.session_state.x_min_max_slider = default_x_range_convex
        st.session_state.y_min_max_slider = default_y_range_convex
        st.session_state.start_x_slider = default_start_x_convex
        st.session_state.start_y_slider = default_start_y_convex
        st.session_state.selected_camera_option_name = default_angle_option_name
        st.session_state.steps_slider = default_steps_convex
        st.session_state.learning_rate_input = default_lr_convex


    new_x_min, new_x_max = st.session_state.x_min_max_slider
    new_y_min, new_y_max = st.session_state.y_min_max_slider
    st.session_state.start_x_slider = max(new_x_min, min(new_x_max, st.session_state.start_x_slider))
    st.session_state.start_y_slider = max(new_y_min, min(new_y_max, st.session_state.start_y_slider))

param_keys_to_init = ["x_min_max_slider", "y_min_max_slider", "start_x_slider", "start_y_slider", "learning_rate_input", "steps_slider"]
is_first_load = not all(key in st.session_state for key in param_keys_to_init)
if is_first_load:
    apply_preset_for_func_type(st.session_state.selected_func_type, is_initial_load=True)

# --- 2. 현재 설정값 결정 (세션 상태 기반) ---
camera_eye = angle_options[st.session_state.selected_camera_option_name]
if st.session_state.selected_func_type == "사용자 정의 함수 입력":
    func_input = st.session_state.user_func_input
else:
    func_input = default_funcs.get(st.session_state.selected_func_type, "x**2+y**2")

x_min, x_max = st.session_state.x_min_max_slider
y_min, y_max = st.session_state.y_min_max_slider
start_x = st.session_state.start_x_slider
start_y = st.session_state.start_y_slider
learning_rate = st.session_state.learning_rate_input
steps = st.session_state.steps_slider

x_sym, y_sym = symbols('x y') 

# --- 3. 경로 관련 세션 상태 초기화 ---
if "gd_path" not in st.session_state or \
   st.session_state.get("last_func_eval", "") != func_input or \
   st.session_state.get("last_start_x_eval", 0.0) != start_x or \
   st.session_state.get("last_start_y_eval", 0.0) != start_y or \
   st.session_state.get("last_lr_eval", 0.0) != learning_rate:

    st.session_state.gd_path = [(float(start_x), float(start_y))]
    st.session_state.gd_step = 0
    st.session_state.play = False 
    st.session_state.last_func_eval = func_input
    st.session_state.last_start_x_eval = start_x
    st.session_state.last_start_y_eval = start_y
    st.session_state.last_lr_eval = learning_rate
    st.session_state.animation_camera_eye = camera_eye 
    st.session_state.messages = []


# --- 사이드바 UI 구성 ---
with st.sidebar:
    st.header("⚙️ 설정 및 파라미터")

    with st.expander("💡 경사 하강법이란?", expanded=False):
        st.markdown("""(설명 내용 생략)""") # 이전과 동일
    with st.expander("📖 주요 파라미터 가이드", expanded=False):
        st.markdown(f"""(설명 내용 생략)""") # 이전과 동일

    st.subheader("📊 함수 및 그래프 설정")
    
    def handle_func_type_change():
        new_func_type = st.session_state.func_radio_key_widget # 라디오 버튼 자체의 현재 값
        st.session_state.selected_func_type = new_func_type # 우리 로직용 세션 상태 업데이트
        apply_preset_for_func_type(new_func_type, is_initial_load=True) # 함수 변경 시 프리셋 강제 적용

    st.radio( 
        "그래프 시점(카메라 각도)",
        options=list(angle_options.keys()),
        index=list(angle_options.keys()).index(st.session_state.selected_camera_option_name),
        key="camera_angle_radio_key_widget", 
        on_change=lambda: setattr(st.session_state, "selected_camera_option_name", st.session_state.camera_angle_radio_key_widget)
    )
    st.radio(
        "함수 유형",
        func_options, # 수정된 default_funcs에 따라 업데이트된 func_options 사용
        index = func_options.index(st.session_state.selected_func_type),
        key="func_radio_key_widget", 
        on_change=handle_func_type_change 
    )
    if st.session_state.selected_func_type == "사용자 정의 함수 입력":
        st.text_input("함수 f(x, y) 입력", 
                      value=st.session_state.user_func_input,
                      key="user_func_text_input_key_widget", 
                      on_change=lambda: setattr(st.session_state, "user_func_input", st.session_state.user_func_text_input_key_widget)
                      )
    else:
        st.text_input("선택된 함수 f(x, y)", value=func_input, disabled=True)
    
    st.slider("x 범위", -10.0, 10.0, st.session_state.x_min_max_slider, step=0.1, 
              key="x_slider_key_widget", 
              on_change=lambda: setattr(st.session_state, "x_min_max_slider", st.session_state.x_slider_key_widget))
    st.slider("y 범위", -10.0, 10.0, st.session_state.y_min_max_slider, step=0.1, 
              key="y_slider_key_widget", 
              on_change=lambda: setattr(st.session_state, "y_min_max_slider", st.session_state.y_slider_key_widget))

    st.subheader("🔩 경사 하강법 파라미터")
    current_x_min_ui, current_x_max_ui = st.session_state.x_min_max_slider # 슬라이더 범위는 항상 최신 세션 상태 따름
    current_y_min_ui, current_y_max_ui = st.session_state.y_min_max_slider
    st.slider("시작 x 위치", float(current_x_min_ui), float(current_x_max_ui), st.session_state.start_x_slider, step=0.1, 
              key="start_x_key_widget", 
              on_change=lambda: setattr(st.session_state, "start_x_slider", st.session_state.start_x_key_widget))
    st.slider("시작 y 위치", float(current_y_min_ui), float(current_y_max_ui), st.session_state.start_y_slider, step=0.1, 
              key="start_y_key_widget", 
              on_change=lambda: setattr(st.session_state, "start_y_slider", st.session_state.start_y_key_widget))
    st.number_input("학습률 (Learning Rate, α)", min_value=0.0001, max_value=1.0, value=st.session_state.learning_rate_input, step=0.001, format="%.4f", 
                    key="lr_key_widget", 
                    on_change=lambda: setattr(st.session_state, "learning_rate_input", st.session_state.lr_key_widget))
    st.slider("최대 반복 횟수", 1, 100, st.session_state.steps_slider, help="경사 하강법을 몇 번 반복할지 설정합니다.", 
              key="steps_key_widget", 
              on_change=lambda: setattr(st.session_state, "steps_slider", st.session_state.steps_key_widget))

    st.sidebar.subheader("🔬 SciPy 최적화 결과 (참고용)")
    scipy_result_placeholder = st.sidebar.empty() 


# --- 그래프 그리기 함수 (plot_gd) ---
# (이전 코드와 동일하며, GD 최종점 마커만 수정되었음)
def plot_gd(f_np_func, dx_np_func, dy_np_func, x_min_curr, x_max_curr, y_min_curr, y_max_curr, gd_path_curr, min_point_scipy_curr, current_camera_eye_func):
    X_plot = np.linspace(x_min_curr, x_max_curr, 80) 
    Y_plot = np.linspace(y_min_curr, y_max_curr, 80)
    Xs_plot, Ys_plot = np.meshgrid(X_plot, Y_plot)
    
    try: Zs_plot = f_np_func(Xs_plot, Ys_plot)
    except Exception: Zs_plot = np.zeros_like(Xs_plot)

    fig = go.Figure()
    fig.add_trace(go.Surface(x=X_plot, y=Y_plot, z=Zs_plot, opacity=0.7, colorscale='Viridis',
                             contours_z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True),
                             name="함수 표면 f(x,y)", showscale=False))

    px, py = zip(*gd_path_curr)
    try: pz = [f_np_func(pt_x, pt_y) for pt_x, pt_y in gd_path_curr]
    except Exception: pz = [np.nan_to_num(f_np_func(pt_x, pt_y)) for pt_x, pt_y in gd_path_curr]

    path_texts = [f"S{idx}<br>({pt_x:.2f}, {pt_y:.2f})" for idx, (pt_x, pt_y) in enumerate(gd_path_curr)]

    fig.add_trace(go.Scatter3d(
        x=px, y=py, z=pz, mode='lines+markers+text',
        marker=dict(size=5, color='red', symbol='circle'), line=dict(color='red', width=3),
        name="경사 하강 경로", text=path_texts, textposition="top right", textfont=dict(size=10, color='black')
    ))

    arrow_scale_factor = 0.3
    num_arrows_to_show = min(5, len(gd_path_curr) - 1)
    if num_arrows_to_show > 0:
        for i in range(num_arrows_to_show):
            arrow_start_idx = len(gd_path_curr) - 1 - i -1 
            if arrow_start_idx < 0: continue
            gx, gy = gd_path_curr[arrow_start_idx]
            try:
                gz = f_np_func(gx, gy)
                grad_x_arrow = dx_np_func(gx, gy)
                grad_y_arrow = dy_np_func(gx, gy)
                if not (np.isnan(grad_x_arrow) or np.isnan(grad_y_arrow) or np.isnan(gz)):
                    fig.add_trace(go.Cone(
                        x=[gx], y=[gy], z=[gz + 0.02 * np.abs(gz) if gz != 0 else 0.02],
                        u=[-grad_x_arrow * arrow_scale_factor], v=[-grad_y_arrow * arrow_scale_factor], w=[0], 
                        sizemode="absolute", sizeref=0.25, colorscale=[[0, 'magenta'], [1, 'magenta']], showscale=False, 
                        anchor="tail", name=f"기울기 S{arrow_start_idx}" if i == 0 else "", hoverinfo='skip'
                    ))
            except Exception: continue 
    
    if min_point_scipy_curr:
        min_x_sp, min_y_sp, min_z_sp = min_point_scipy_curr
        fig.add_trace(go.Scatter3d(
            x=[min_x_sp], y=[min_y_sp], z=[min_z_sp], mode='markers+text',
            marker=dict(size=10, color='cyan', symbol='diamond'),
            text=["SciPy 최적점"], textposition="bottom center", name="SciPy 최적점"
        ))

    last_x_gd, last_y_gd = gd_path_curr[-1]
    try: last_z_gd = f_np_func(last_x_gd, last_y_gd)
    except Exception: last_z_gd = np.nan 

    fig.add_trace(go.Scatter3d(
        x=[last_x_gd], y=[last_y_gd], z=[last_z_gd if not np.isnan(last_z_gd) else Zs_plot.min()], mode='markers+text',
        marker=dict(size=7, color='orange', symbol='circle', line=dict(color='black', width=1)), # ***** 마커 수정됨 *****
        text=["GD 최종점"], textposition="top left", name="GD 최종점"
    ))

    fig.update_layout(
        scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='f(x, y)', camera=dict(eye=current_camera_eye_func), aspectmode='cube'),
        height=600, margin=dict(l=0, r=0, t=30, b=0),
        title_text="경사 하강법 경로 및 함수 표면", title_x=0.5,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# --- 메인 페이지 레이아웃 및 나머지 로직 (이전과 거의 동일) ---
if st.session_state.get("play", False):
    st.info("🎥 애니메이션 실행 중...")

st.markdown("---") 
col_btn1, col_btn2, col_btn3 = st.columns([1.5, 2, 1])
with col_btn1: step_btn = st.button("🚶 한 스텝 이동", use_container_width=True)
with col_btn2: play_btn = st.button("▶️ 전체 실행", key="playbtn_widget_key", use_container_width=True)
with col_btn3: reset_btn = st.button("🔄 초기화", key="resetbtn_widget_key", use_container_width=True)

graph_placeholder = st.empty() 
min_point_scipy_coords = None 

try:
    f_sym_parsed = sympify(func_input) 
    f_np_parsed = lambdify((x_sym, y_sym), f_sym_parsed, modules=['numpy', {'cos': np.cos, 'sin': np.sin, 'exp': np.exp, 'sqrt': np.sqrt, 'pi': np.pi}])
    try: 
        def min_func_scipy(vars_list): return f_np_parsed(vars_list[0], vars_list[1])
        potential_starts = [[0.0,0.0], [float(start_x), float(start_y)]] 
        if "Himmelblau" in st.session_state.selected_func_type: 
            potential_starts.extend([[3,2], [-2.805, 3.131], [-3.779, -3.283], [3.584, -1.848]])
        best_res = None
        for p_start in potential_starts:
            res_temp = minimize(min_func_scipy, p_start, method='Nelder-Mead', tol=1e-6, options={'maxiter': 200, 'adaptive': True})
            if best_res is None or (res_temp.success and res_temp.fun < best_res.fun) or (res_temp.success and not best_res.success):
                best_res = res_temp
        if best_res and best_res.success:
            min_x_sp, min_y_sp = best_res.x
            min_z_sp = f_np_parsed(min_x_sp, min_y_sp) 
            min_point_scipy_coords = (min_x_sp, min_y_sp, min_z_sp)
            scipy_result_placeholder.markdown(f"""
            - **위치 (x, y)**: `({min_x_sp:.3f}, {min_y_sp:.3f})`
            - **함수 값 f(x,y)**: `{min_z_sp:.4f}`""")
        else: scipy_result_placeholder.info("SciPy 최적점을 찾지 못했습니다.")
    except Exception as e_scipy: scipy_result_placeholder.warning(f"SciPy 오류: {str(e_scipy)[:100]}...")
except Exception as e: 
    st.error(f"🚨 함수 정의 오류: {e}. 함수 수식을 확인해주세요."); st.stop()
if not callable(f_np_parsed): st.error("함수 변환 실패."); st.stop()

dx_f_sym_parsed = diff(f_sym_parsed, x_sym)
dy_f_sym_parsed = diff(f_sym_parsed, y_sym)
dx_np_parsed = lambdify((x_sym, y_sym), dx_f_sym_parsed, modules=['numpy', {'cos': np.cos, 'sin': np.sin, 'exp': np.exp, 'sqrt': np.sqrt, 'pi': np.pi}])
dy_np_parsed = lambdify((x_sym, y_sym), dy_f_sym_parsed, modules=['numpy', {'cos': np.cos, 'sin': np.sin, 'exp': np.exp, 'sqrt': np.sqrt, 'pi': np.pi}])

if reset_btn:
    st.session_state.selected_func_type = default_func_type 
    apply_preset_for_func_type(st.session_state.selected_func_type, is_initial_load=True) 
    st.session_state.user_func_input = "x**2 + y**2" 
    
    current_start_x_on_reset = st.session_state.start_x_slider 
    current_start_y_on_reset = st.session_state.start_y_slider
    current_func_input_on_reset = default_funcs.get(st.session_state.selected_func_type, "x**2+y**2") if st.session_state.selected_func_type != "사용자 정의 함수 입력" else st.session_state.user_func_input
    
    st.session_state.gd_path = [(float(current_start_x_on_reset), float(current_start_y_on_reset))]
    st.session_state.gd_step = 0
    st.session_state.play = False 
    st.session_state.animation_camera_eye = angle_options[st.session_state.selected_camera_option_name]
    st.session_state.messages = []
    st.session_state.last_func_eval = current_func_input_on_reset
    st.session_state.last_start_x_eval = current_start_x_on_reset
    st.session_state.last_start_y_eval = current_start_y_on_reset
    st.session_state.last_lr_eval = st.session_state.learning_rate_input
    st.rerun() 

if step_btn and st.session_state.gd_step < steps:
    st.session_state.play = False 
    curr_x, curr_y = st.session_state.gd_path[-1]
    try:
        grad_x_val = dx_np_parsed(curr_x, curr_y); grad_y_val = dy_np_parsed(curr_x, curr_y) 
        if np.isnan(grad_x_val) or np.isnan(grad_y_val): st.session_state.messages.append(("error", "기울기 계산 결과가 NaN입니다."))
        else:
            next_x = curr_x - learning_rate * grad_x_val; next_y = curr_y - learning_rate * grad_y_val
            st.session_state.gd_path.append((next_x, next_y)); st.session_state.gd_step += 1
    except Exception as e: st.session_state.messages.append(("error", f"스텝 진행 중 오류: {e}"))
    st.rerun() 

if play_btn: 
    if not st.session_state.get("play", False): 
        st.session_state.play = True; st.session_state.animation_camera_eye = camera_eye 
        st.session_state.messages = []; st.rerun() 

if st.session_state.get("play", False) and st.session_state.gd_step < steps:
    current_animation_cam = st.session_state.get("animation_camera_eye", camera_eye) 
    curr_x_anim, curr_y_anim = st.session_state.gd_path[-1]
    try:
        grad_x_anim = dx_np_parsed(curr_x_anim, curr_y_anim); grad_y_anim = dy_np_parsed(curr_x_anim, curr_y_anim) 
        if np.isnan(grad_x_anim) or np.isnan(grad_y_anim):
            st.session_state.messages.append(("error", "애니메이션 중 기울기 NaN. 중단합니다.")); st.session_state.play = False; st.rerun()
        else:
            next_x_anim = curr_x_anim - learning_rate * grad_x_anim; next_y_anim = curr_y_anim - learning_rate * grad_y_anim
            st.session_state.gd_path.append((next_x_anim, next_y_anim)); st.session_state.gd_step += 1
            fig_anim = plot_gd(f_np_parsed, dx_np_parsed, dy_np_parsed, x_min, x_max, y_min, y_max,
                            st.session_state.gd_path, min_point_scipy_coords, current_animation_cam)
            graph_placeholder.plotly_chart(fig_anim, use_container_width=True) 
            time.sleep(0.12) 
            if st.session_state.gd_step < steps: st.rerun() 
            else: st.session_state.play = False; st.session_state.play_just_finished = True; st.rerun()
    except Exception as e:
        st.session_state.messages.append(("error", f"애니메이션 중 오류: {e}")); st.session_state.play = False; st.rerun()
else: 
    current_display_cam = camera_eye 
    if st.session_state.get("play_just_finished", False): 
        current_display_cam = st.session_state.get("animation_camera_eye", camera_eye) 
        st.session_state.play_just_finished = False
    fig_static = plot_gd(f_np_parsed, dx_np_parsed, dy_np_parsed, x_min, x_max, y_min, y_max,
                        st.session_state.gd_path, min_point_scipy_coords, current_display_cam)
    graph_placeholder.plotly_chart(fig_static, use_container_width=True, key="main_chart_static")

temp_messages = st.session_state.get("messages", []) 
for msg_type, msg_content in temp_messages:
    if msg_type == "error": st.error(msg_content)
    elif msg_type == "warning": st.warning(msg_content)
    elif msg_type == "success": st.success(msg_content)
if not st.session_state.get("play", False) : 
    st.session_state.messages = [] 
    last_x_final, last_y_final = st.session_state.gd_path[-1]
    try:
        last_z_final = f_np_parsed(last_x_final, last_y_final); grad_x_final = dx_np_parsed(last_x_final, last_y_final)
        grad_y_final = dy_np_parsed(last_x_final, last_y_final); grad_norm_final = np.sqrt(grad_x_final**2 + grad_y_final**2)
        if np.isnan(last_z_final) or np.isinf(last_z_final): st.error("🚨 함수 값이 발산했습니다! (NaN 또는 무한대)")
        elif st.session_state.gd_step >= steps and grad_norm_final > 1e-2: st.warning(f"⚠️ 최대 반복({steps}) 도달, 기울기({grad_norm_final:.4f})가 아직 충분히 작지 않음.")
        elif grad_norm_final < 1e-2 and not (np.isnan(grad_norm_final) or np.isinf(grad_norm_final)): st.success(f"🎉 기울기({grad_norm_final:.4f})가 매우 작아 최적점/안장점에 근접한 듯 합니다!")
    except Exception: pass
