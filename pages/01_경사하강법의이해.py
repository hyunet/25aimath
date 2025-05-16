import streamlit as st
from sympy import symbols, diff, sympify, lambdify
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize
import time

st.set_page_config(layout="wide", page_title="경사 하강법 체험")

st.title("🎢 딥러닝 경사하강법 체험 - 다양한 함수와 시점 선택")
st.caption("제작: 서울고 송석리 선생님 | 개선: Gemini AI")

# --- 정적 옵션 정의 ---
angle_options = {
    "사선(전체 보기)": dict(x=1.7, y=1.7, z=1.2),
    "정면(x+방향)": dict(x=2.0, y=0.0, z=0.5),
    "정면(y+방향)": dict(x=0.0, y=2.0, z=0.5),
    "위에서 내려다보기": dict(x=0.0, y=0.0, z=3.0),
    "뒤쪽(x-방향)": dict(x=-2.0, y=0.0, z=0.5),
    "옆(y-방향)": dict(x=0.0, y=-2.0, z=0.5)
}
default_angle_option_name = list(angle_options.keys())[0]

default_funcs = {
    "볼록 함수 (최적화 쉬움, 예: x²+y²)": "x**2 + y**2",
    "안장점 함수 (최적화 어려움, 예: x²-y²)": "x**2 - y**2",
    "복잡한 함수 (다중 지역 최적점 가능성, 예: Rastrigin 유사)": "20 + (x**2 - 10*cos(2*3.14159*x)) + (y**2 - 10*cos(2*3.14159*y))",
    "사용자 정의 함수 입력": ""
}
func_options = list(default_funcs.keys())

# --- 1. 모든 UI 제어용 세션 상태 변수 최상단 초기화 ---
if "selected_camera_option_name" not in st.session_state:
    st.session_state.selected_camera_option_name = default_angle_option_name
if "selected_func_type" not in st.session_state:
    st.session_state.selected_func_type = func_options[0]
if "user_func_input" not in st.session_state:
    st.session_state.user_func_input = "x**2 + y**2" # 사용자 정의 함수 기본값
if "x_min_max_slider" not in st.session_state:
    st.session_state.x_min_max_slider = (-5.0, 5.0)
if "y_min_max_slider" not in st.session_state:
    st.session_state.y_min_max_slider = (-5.0, 5.0)
if "start_x_slider" not in st.session_state: 
    st.session_state.start_x_slider = 4.0
if "start_y_slider" not in st.session_state: 
    st.session_state.start_y_slider = 4.0
if "learning_rate_input" not in st.session_state: 
    st.session_state.learning_rate_input = 0.1
if "steps_slider" not in st.session_state: 
    st.session_state.steps_slider = 15

# --- 2. 현재 설정값 결정 (세션 상태 기반) ---
camera_eye = angle_options[st.session_state.selected_camera_option_name]

if st.session_state.selected_func_type == "사용자 정의 함수 입력":
    func_input = st.session_state.user_func_input
else:
    func_input = default_funcs[st.session_state.selected_func_type]

x_min, x_max = st.session_state.x_min_max_slider
y_min, y_max = st.session_state.y_min_max_slider
start_x = st.session_state.start_x_slider
start_y = st.session_state.start_y_slider
learning_rate = st.session_state.learning_rate_input
steps = st.session_state.steps_slider

# 기호 변수 정의
x_sym, y_sym = symbols('x y') 

# --- 3. 경로 관련 세션 상태 초기화 (func_input 등 사용) ---
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
    st.session_state.animation_camera_eye = camera_eye # 현재 camera_eye 사용
    st.session_state.messages = []

# --- 교육적 설명 섹션 (애니메이션 중에도 보이도록 유지) ---
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


# --- 4. UI 컨트롤 섹션 (애니메이션 중에는 숨김 처리) ---
if not st.session_state.get("play", False):
    col_params1, col_params2 = st.columns(2)
    with col_params1:
        st.subheader("📊 함수 및 그래프 설정")
        # 카메라 각도 라디오 버튼
        st.radio( # 위젯 반환값은 사용하지 않으므로 _ 처리 가능
            "그래프 시점(카메라 각도) 선택",
            options=list(angle_options.keys()),
            index=list(angle_options.keys()).index(st.session_state.selected_camera_option_name),
            horizontal=True,
            key="camera_angle_radio_key_widget", # 키 이름 변경 (세션 상태 변수와 구분)
            on_change=lambda: setattr(st.session_state, "selected_camera_option_name", st.session_state.camera_angle_radio_key_widget)
        )
        # camera_eye는 이미 상단에서 st.session_state.selected_camera_option_name 기반으로 계산됨

        # 함수 선택 라디오 버튼
        st.radio(
            "함수 유형을 선택하세요.",
            func_options,
            horizontal=False,
            index = func_options.index(st.session_state.selected_func_type),
            key="func_radio_key_widget", # 키 이름 변경
            on_change=lambda: setattr(st.session_state, "selected_func_type", st.session_state.func_radio_key_widget)
        )
        # func_input은 이미 상단에서 st.session_state.selected_func_type 기반으로 계산됨

        if st.session_state.selected_func_type == "사용자 정의 함수 입력":
            st.text_input("함수 f(x, y)를 입력하세요 (예: x**2 + y**2)", 
                          value=st.session_state.user_func_input,
                          key="user_func_text_input_key_widget", # 키 이름 변경
                          on_change=lambda: setattr(st.session_state, "user_func_input", st.session_state.user_func_text_input_key_widget)
                          )
        else:
            st.text_input("선택된 함수 f(x, y)", value=default_funcs[st.session_state.selected_func_type], disabled=True)
        
        # x, y 범위 슬라이더
        st.slider("x 범위", -10.0, 10.0, st.session_state.x_min_max_slider, step=0.1, 
                  key="x_slider_key_widget", # 키 이름 변경
                  on_change=lambda: setattr(st.session_state, "x_min_max_slider", st.session_state.x_slider_key_widget))
        st.slider("y 범위", -10.0, 10.0, st.session_state.y_min_max_slider, step=0.1, 
                  key="y_slider_key_widget", # 키 이름 변경
                  on_change=lambda: setattr(st.session_state, "y_min_max_slider", st.session_state.y_slider_key_widget))

    with col_params2:
        st.subheader("⚙️ 경사 하강법 파라미터")
        # 시작 위치, 학습률, 스텝 수 위젯
        st.slider("시작 x 위치", float(x_min), float(x_max), st.session_state.start_x_slider, step=0.1, 
                  key="start_x_key_widget", # 키 이름 변경
                  on_change=lambda: setattr(st.session_state, "start_x_slider", st.session_state.start_x_key_widget))
        st.slider("시작 y 위치", float(y_min), float(y_max), st.session_state.start_y_slider, step=0.1, 
                  key="start_y_key_widget", # 키 이름 변경
                  on_change=lambda: setattr(st.session_state, "start_y_slider", st.session_state.start_y_key_widget))
        st.number_input("학습률 (Learning Rate, α)", min_value=0.0001, max_value=1.0, value=st.session_state.learning_rate_input, step=0.001, format="%.4f", 
                        key="lr_key_widget", # 키 이름 변경
                        on_change=lambda: setattr(st.session_state, "learning_rate_input", st.session_state.lr_key_widget))
        st.slider("최대 반복 횟수", 1, 100, st.session_state.steps_slider, help="경사 하강법을 몇 번 반복할지 설정합니다.", 
                  key="steps_key_widget", # 키 이름 변경
                  on_change=lambda: setattr(st.session_state, "steps_slider", st.session_state.steps_key_widget))
else: 
    st.info("🎥 애니메이션 실행 중... 완료 후 파라미터 설정이 다시 나타납니다.")
    # 애니메이션 중에는 이미 상단에서 결정된 camera_eye, func_input, 범위 등을 사용


# --- 5. 그래프 그리기 함수 (이전과 거의 동일, 변수명 정리) ---
def plot_gd(f_np_func, dx_np_func, dy_np_func, x_min_curr, x_max_curr, y_min_curr, y_max_curr, gd_path_curr, min_point_scipy_curr, current_camera_eye_func):
    X_plot = np.linspace(x_min_curr, x_max_curr, 80) 
    Y_plot = np.linspace(y_min_curr, y_max_curr, 80)
    Xs_plot, Ys_plot = np.meshgrid(X_plot, Y_plot)
    
    try:
        Zs_plot = f_np_func(Xs_plot, Ys_plot)
    except Exception:
        Zs_plot = np.zeros_like(Xs_plot)

    fig = go.Figure()
    fig.add_trace(go.Surface(x=X_plot, y=Y_plot, z=Zs_plot, opacity=0.7, colorscale='Viridis',
                             contours_z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True),
                             name="함수 표면 f(x,y)", showscale=False))

    px, py = zip(*gd_path_curr)
    try:
        pz = [f_np_func(pt_x, pt_y) for pt_x, pt_y in gd_path_curr]
    except Exception: 
        pz = [np.nan_to_num(f_np_func(pt_x, pt_y)) for pt_x, pt_y in gd_path_curr]

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
        marker=dict(size=10, color='blue', symbol='x'),
        text=["GD 최종점"], textposition="top left", name="GD 최종점"
    ))

    fig.update_layout(
        scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='f(x, y)', camera=dict(eye=current_camera_eye_func), aspectmode='cube'),
        width=None, height=650, margin=dict(l=0, r=0, t=30, b=0),
        title_text="경사 하강법 경로 및 함수 표면", title_x=0.5,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# --- 6. 제어 버튼 (항상 표시) ---
st.markdown("---")
col_btn1, col_btn2, col_btn3 = st.columns([1.5, 2, 1])
with col_btn1:
    step_btn = st.button("🚶 한 스텝 이동", use_container_width=True)
with col_btn2:
    play_btn = st.button("🎥 전체 실행 (애니메이션)", key="playbtn_widget_key", use_container_width=True) # 위젯 키 추가
with col_btn3:
    reset_btn = st.button("🔄 초기화", key="resetbtn_widget_key", use_container_width=True) # 위젯 키 추가

# --- 7. 메인 로직 ---
graph_placeholder = st.empty() 
min_point_scipy_coords = None 

try:
    f_sym_parsed = sympify(func_input) 
    f_np_parsed = lambdify((x_sym, y_sym), f_sym_parsed, modules=['numpy', {'cos': np.cos, 'sin': np.sin, 'exp': np.exp, 'sqrt': np.sqrt}])

    try: # SciPy 최적화
        def min_func_scipy(vars_list): return f_np_parsed(vars_list[0], vars_list[1])
        potential_starts = [[0.0,0.0], [float(start_x), float(start_y)]] 
        best_res = None
        for p_start in potential_starts:
            res_temp = minimize(min_func_scipy, p_start, method='Nelder-Mead', tol=1e-6, options={'maxiter': 200, 'adaptive': True})
            if best_res is None or (res_temp.success and res_temp.fun < best_res.fun) or (res_temp.success and not best_res.success):
                best_res = res_temp
        if best_res and best_res.success:
            min_x_sp, min_y_sp = best_res.x
            min_z_sp = f_np_parsed(min_x_sp, min_y_sp) 
            min_point_scipy_coords = (min_x_sp, min_y_sp, min_z_sp)
    except Exception: pass

except Exception as e: 
    st.error(f"🚨 함수 정의 오류: {e}. 함수 수식을 확인해주세요.")
    st.stop()

if not callable(f_np_parsed): # lambdify 실패 시
    st.error("함수 변환에 실패하여 메인 로직을 실행할 수 없습니다.")
    st.stop()

dx_f_sym_parsed = diff(f_sym_parsed, x_sym)
dy_f_sym_parsed = diff(f_sym_parsed, y_sym)
dx_np_parsed = lambdify((x_sym, y_sym), dx_f_sym_parsed, modules=['numpy', {'cos': np.cos, 'sin': np.sin, 'exp': np.exp, 'sqrt': np.sqrt}])
dy_np_parsed = lambdify((x_sym, y_sym), dy_f_sym_parsed, modules=['numpy', {'cos': np.cos, 'sin': np.sin, 'exp': np.exp, 'sqrt': np.sqrt}])


if reset_btn:
    st.session_state.gd_path = [(float(start_x), float(start_y))] 
    st.session_state.gd_step = 0
    st.session_state.play = False 
    st.session_state.selected_camera_option_name = default_angle_option_name
    st.session_state.animation_camera_eye = angle_options[st.session_state.selected_camera_option_name]
    
    # 다른 파라미터들도 기본값으로 초기화
    st.session_state.selected_func_type = func_options[0]
    st.session_state.user_func_input = "x**2 + y**2"
    st.session_state.x_min_max_slider = (-5.0, 5.0)
    st.session_state.y_min_max_slider = (-5.0, 5.0)
    st.session_state.start_x_slider = 4.0
    st.session_state.start_y_slider = 4.0
    st.session_state.learning_rate_input = 0.1
    st.session_state.steps_slider = 15
    st.session_state.messages = []
    st.rerun() 

if step_btn and st.session_state.gd_step < steps:
    st.session_state.play = False 
    curr_x, curr_y = st.session_state.gd_path[-1]
    try:
        grad_x_val = dx_np_parsed(curr_x, curr_y) 
        grad_y_val = dy_np_parsed(curr_x, curr_y) 
        if np.isnan(grad_x_val) or np.isnan(grad_y_val):
            st.session_state.messages.append(("error", "기울기 계산 결과가 NaN입니다."))
        else:
            next_x = curr_x - learning_rate * grad_x_val
            next_y = curr_y - learning_rate * grad_y_val
            st.session_state.gd_path.append((next_x, next_y))
            st.session_state.gd_step += 1
    except Exception as e: st.session_state.messages.append(("error", f"스텝 진행 중 오류: {e}"))
    st.rerun() 

if play_btn: # play_btn은 st.button의 반환값 (True if clicked)
    if not st.session_state.get("play", False): 
        st.session_state.play = True
        st.session_state.animation_camera_eye = camera_eye # 상단에서 이미 계산된 camera_eye 사용
        st.session_state.messages = [] 
        st.rerun() 

if st.session_state.get("play", False) and st.session_state.gd_step < steps:
    current_animation_cam = st.session_state.get("animation_camera_eye", camera_eye) 
    curr_x_anim, curr_y_anim = st.session_state.gd_path[-1]
    try:
        grad_x_anim = dx_np_parsed(curr_x_anim, curr_y_anim) 
        grad_y_anim = dy_np_parsed(curr_x_anim, curr_y_anim) 
        if np.isnan(grad_x_anim) or np.isnan(grad_y_anim):
            st.session_state.messages.append(("error", "애니메이션 중 기울기 NaN. 중단합니다."))
            st.session_state.play = False; st.rerun()
        else:
            next_x_anim = curr_x_anim - learning_rate * grad_x_anim
            next_y_anim = curr_y_anim - learning_rate * grad_y_anim
            st.session_state.gd_path.append((next_x_anim, next_y_anim))
            st.session_state.gd_step += 1
            
            fig_anim = plot_gd(f_np_parsed, dx_np_parsed, dy_np_parsed, x_min, x_max, y_min, y_max,
                            st.session_state.gd_path, min_point_scipy_coords, current_animation_cam)
            graph_placeholder.plotly_chart(fig_anim, use_container_width=True) 
            time.sleep(0.15) 
            if st.session_state.gd_step < steps: st.rerun() 
            else: st.session_state.play = False; st.session_state.play_just_finished = True; st.rerun()
    except Exception as e:
        st.session_state.messages.append(("error", f"애니메이션 중 오류: {e}"))
        st.session_state.play = False; st.rerun()
        
else: 
    current_display_cam = camera_eye 
    if st.session_state.get("play_just_finished", False): 
        current_display_cam = st.session_state.get("animation_camera_eye", camera_eye) 
        st.session_state.play_just_finished = False

    fig_static = plot_gd(f_np_parsed, dx_np_parsed, dy_np_parsed, x_min, x_max, y_min, y_max,
                        st.session_state.gd_path, min_point_scipy_coords, current_display_cam)
    graph_placeholder.plotly_chart(fig_static, use_container_width=True, key="main_chart_static")


# --- 정보 블록 삭제됨 ---

# 메시지 출력
temp_messages = st.session_state.get("messages", []) 
for msg_type, msg_content in temp_messages:
    if msg_type == "error": st.error(msg_content)
    elif msg_type == "warning": st.warning(msg_content)
    elif msg_type == "success": st.success(msg_content)

if not st.session_state.get("play", False) : 
    st.session_state.messages = [] 
    last_x_final, last_y_final = st.session_state.gd_path[-1]
    try:
        last_z_final = f_np_parsed(last_x_final, last_y_final)
        grad_x_final = dx_np_parsed(last_x_final, last_y_final)
        grad_y_final = dy_np_parsed(last_x_final, last_y_final)
        grad_norm_final = np.sqrt(grad_x_final**2 + grad_y_final**2)
        if np.isnan(last_z_final) or np.isinf(last_z_final):
            st.error("🚨 함수 값이 발산했습니다! (NaN 또는 무한대)")
        elif st.session_state.gd_step >= steps and grad_norm_final > 1e-2: 
            st.warning(f"⚠️ 최대 반복({steps}) 도달, 기울기 크기({grad_norm_final:.4f})가 아직 충분히 작지 않음.")
        elif grad_norm_final < 1e-2 and not (np.isnan(grad_norm_final) or np.isinf(grad_norm_final)):
            st.success(f"🎉 기울기 크기({grad_norm_final:.4f})가 매우 작아 최적점/안장점에 근접한 듯 합니다!")
    except Exception: pass # 오류 발생 시 메시지 생략
            

# --- SciPy 최적점 정보 사이드바 표시 ---
if min_point_scipy_coords:
    st.sidebar.subheader("🔬 SciPy 최적화 결과 (참고용)")
    st.sidebar.markdown(f"""
    `scipy.optimize.minimize` (Nelder-Mead)를 사용해 찾은 (지역) 최적점 후보:
    - **위치 (x, y)**: `({min_point_scipy_coords[0]:.3f}, {min_point_scipy_coords[1]:.3f})`
    - **함수 값 f(x,y)**: `{min_point_scipy_coords[2]:.4f}`
    ---
    *경사 하강법의 목표는 이와 같은 최적점에 도달하는 것입니다.*
    """)
else:
    st.sidebar.info("SciPy 최적점 정보를 계산할 수 없었거나, 함수 정의에 오류가 있어 생략되었습니다.")
