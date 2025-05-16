import streamlit as st
from sympy import symbols, diff, sympify, lambdify
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize
import time
import math

# 페이지 설정 및 제목
st.set_page_config(page_title="딥러닝 경사하강법 체험", layout="wide")
st.title("딥러닝 경사하강법 체험 - 최적화 알고리즘의 이해")
st.markdown("""
이 애플리케이션은 딥러닝의 핵심 알고리즘인 **경사하강법(Gradient Descent)**을 
시각적으로 체험할 수 있도록 설계되었습니다. 다양한 함수와 파라미터를 조정하며
최적화 알고리즘의 작동 원리를 직접 확인해보세요!
""")

# 사이드바에 학습 가이드 추가
with st.sidebar:
    st.header("학습 가이드 📚")
    st.markdown("""
    ### 경사하강법이란?
    경사하강법은 함수의 **최소값**을 찾기 위한 최적화 알고리즘입니다.
    함수의 기울기(그래디언트)를 계산하고, 기울기가 감소하는 방향으로 
    조금씩 이동하면서 최솟값을 찾아갑니다.
    
    ### 주요 개념:
    - **그래디언트(∇f)**: 함수의 기울기 벡터, 가장 가파르게 증가하는 방향을 가리킴
    - **학습률(η)**: 한 번에 이동하는 거리를 결정하는 파라미터
    - **반복 횟수**: 알고리즘의 실행 스텝 수
    
    ### 탐구해볼 활동:
    1. 학습률을 다양하게 변경하며 수렴 속도 비교하기
    2. 볼록 함수와 안장점 함수에서의 동작 차이 관찰하기
    3. 다양한 시작점에서 경로 비교하기
    4. 사용자 정의 함수 만들어 실험하기
    
    ### 중요 용어:
    - **지역 최소값**: 주변 영역에서는 가장 작은 값이지만 전체에서 최소는 아님
    - **전역 최소값**: 함수 전체에서 가장 작은 값
    - **안장점**: 한 방향으로는 최소값, 다른 방향으로는 최대값인 지점
    """)

# 카메라 각도 라디오 버튼 (col 구성 추가)
st.subheader("1️⃣ 그래프 시점 설정")
col_camera1, col_camera2 = st.columns([3, 1])
with col_camera1:
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
with col_camera2:
    # 애니메이션 속도 조절 추가
    animation_speed = st.slider(
        "애니메이션 속도",
        min_value=0.05,
        max_value=1.0,
        value=0.2,
        step=0.05,
        help="애니메이션의 속도를 조절합니다. 값이 작을수록 빠릅니다."
    )

# 함수 선택 섹션
st.subheader("2️⃣ 함수 설정")
default_funcs = {
    "볼록 함수 (최적화 쉬움, 예: x²+y²)": "x**2 + y**2",
    "안장점 함수 (최적화 어려움, 예: x²-y²)": "x**2 - y**2",
    "로젠브록 함수 (바나나 함수, 최적화 어려움)": "100*(y - x**2)**2 + (1 - x)**2",
    "사인 함수 (여러 지역 최소값)": "np.sin(x) + np.sin(y)",
    "사용자 정의 함수 입력": ""
}
func_options = list(default_funcs.keys())

col_func1, col_func2 = st.columns([1, 1])
with col_func1:
    func_radio = st.radio(
        "함수 유형을 선택하세요.",
        func_options,
        horizontal=False,
        index=0
    )

    # 함수 도움말 추가
    func_help = {
        "볼록 함수 (최적화 쉬움, 예: x²+y²)": "이상적인 볼록 함수로, 하나의 최소값을 가집니다. 경사하강법이 항상 전역 최소값으로 수렴합니다.",
        "안장점 함수 (최적화 어려움, 예: x²-y²)": "안장점 함수로, (0,0)에서 x방향은 증가하고 y방향은 감소합니다. 경사하강법이 수렴하기 어려울 수 있습니다.",
        "로젠브록 함수 (바나나 함수, 최적화 어려움)": "최적화 알고리즘 테스트에 자주 사용되는 함수입니다. 좁은 곡률의 계곡 형태를 가져 최적화가 어렵습니다.",
        "사인 함수 (여러 지역 최소값)": "여러 개의 지역 최소값을 가진 함수입니다. 시작점에 따라 다른 최소값으로 수렴할 수 있습니다.",
        "사용자 정의 함수 입력": "원하는 함수를 직접 입력할 수 있습니다. 변수는 x, y를 사용하세요. numpy 함수는 np.로 시작합니다."
    }
    
    if func_radio in func_help:
        st.info(func_help[func_radio])

with col_func2:
    if func_radio == "사용자 정의 함수 입력":
        func_input = st.text_input(
            "함수 f(x, y)를 입력하세요 (예: x**2 + y**2)", 
            value="x**2 + y**2",
            help="Python 구문으로 함수를 입력하세요. NumPy 함수는 np.를 앞에 붙입니다."
        )
    else:
        func_input = default_funcs[func_radio]
        st.text_input("함수 f(x, y)", value=func_input, disabled=True)

# 함수 영역 및 파라미터 섹션 
st.subheader("3️⃣ 경사하강법 파라미터 설정")
col1, col2 = st.columns(2)

with col1:
    x_min, x_max = st.slider("x 범위", -10, 10, (-5, 5))
    y_min, y_max = st.slider("y 범위", -10, 10, (-5, 5))

with col2:
    # 시작점 계산 개선: 슬라이더 범위 내에서 중간값으로 초기화
    default_x = min(max(0, x_min), x_max) if x_min <= 0 <= x_max else (x_min + x_max) / 2
    default_y = min(max(0, y_min), y_max) if y_min <= 0 <= y_max else (y_min + y_max) / 2
    
    start_x = st.slider("시작 x 위치", x_min, x_max, float(default_x))
    start_y = st.slider("시작 y 위치", y_min, y_max, float(default_y))

# 학습 파라미터
col3, col4, col5 = st.columns(3)
with col3:
    learning_rate = st.number_input(
        "학습률(η, Learning Rate)", 
        min_value=0.001, 
        max_value=1.0, 
        value=0.2, 
        step=0.01, 
        format="%.3f",
        help="각 스텝에서 이동하는 거리를 결정합니다. 값이 크면 빠르게 이동하지만 발산할 수 있고, 작으면 안정적이지만 수렴이 느립니다."
    )
with col4:
    steps = st.slider(
        "최대 반복 횟수", 
        1, 50, 15,
        help="경사하강법을 실행할 최대 스텝 수입니다."
    )
with col5:
    # 수렴 임계값 추가
    convergence_threshold = st.number_input(
        "수렴 임계값", 
        min_value=0.0001, 
        max_value=0.1, 
        value=0.01, 
        step=0.001, 
        format="%.4f",
        help="그래디언트 크기가 이 값보다 작아지면 알고리즘이 수렴한 것으로 간주합니다."
    )

# 수식 변수 정의
x, y = symbols('x y')

# --- 상태 관리 개선 ---
# 중요 파라미터 변경 시 상태 재설정
params_key = f"{func_input}_{start_x}_{start_y}_{learning_rate}_{x_min}_{x_max}_{y_min}_{y_max}"
if "params_key" not in st.session_state or st.session_state.get("params_key", "") != params_key:
    st.session_state.gd_path = [(float(start_x), float(start_y))]
    st.session_state.gd_step = 0
    st.session_state.play = False
    st.session_state.params_key = params_key
    st.session_state.converged = False
    st.session_state.converged_step = -1

# 함수 및 그래디언트 계산 함수 (오류 처리 추가)
def setup_function(func_str):
    try:
        # 함수 파싱 및 변환
        f = sympify(func_str)
        f_np = lambdify((x, y), f, modules=['numpy', 'scipy'])
        
        # 미분 계산
        dx_f = diff(f, x)
        dy_f = diff(f, y)
        dx_np = lambdify((x, y), dx_f, modules=['numpy', 'scipy'])
        dy_np = lambdify((x, y), dy_f, modules=['numpy', 'scipy'])
        
        return f_np, dx_np, dy_np, None
    except Exception as e:
        error_msg = f"함수 처리 오류: {str(e)}"
        return None, None, None, error_msg

# 경사하강법 단계 실행 함수 (범위 체크 및 수렴 확인 추가)
def gradient_descent_step(curr_x, curr_y, dx_np, dy_np, learning_rate, 
                          x_min, x_max, y_min, y_max, threshold):
    # 그래디언트 계산
    try:
        grad_x = dx_np(curr_x, curr_y)
        grad_y = dy_np(curr_x, curr_y)
    except Exception:
        # 그래디언트 계산 오류 시 작은 랜덤 값으로 대체
        grad_x = np.random.uniform(-0.1, 0.1)
        grad_y = np.random.uniform(-0.1, 0.1)
    
    # 그래디언트 크기 계산
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # 수렴 확인
    converged = grad_magnitude < threshold
    
    # 다음 위치 계산
    next_x = curr_x - learning_rate * grad_x
    next_y = curr_y - learning_rate * grad_y
    
    # 범위 체크 및 클리핑
    next_x = np.clip(next_x, x_min, x_max)
    next_y = np.clip(next_y, y_min, y_max)
    
    return next_x, next_y, grad_x, grad_y, grad_magnitude, converged

# 최적점 찾기 함수 (안전한 최적화)
def find_optimal_point(f_np, x_min, x_max, y_min, y_max):
    def min_func(vars):
        return float(f_np(vars[0], vars[1]))
    
    # 여러 시작점에서 최적화 시도
    best_result = None
    best_value = float('inf')
    
    # 그리드 시작점으로 여러 번 최적화 시도
    start_points = [
        [0, 0],  # 원점
        [start_x, start_y],  # 사용자 시작점
        [(x_min + x_max) / 2, (y_min + y_max) / 2],  # 중앙
        [x_min, y_min],  # 좌하단
        [x_max, y_max]   # 우상단
    ]
    
    for start_point in start_points:
        try:
            res = minimize(min_func, start_point, bounds=[(x_min, x_max), (y_min, y_max)])
            if res.success and res.fun < best_value:
                best_result = res
                best_value = res.fun
        except Exception:
            continue
    
    # 최적화 성공했으면 결과 반환, 실패하면 기본값
    if best_result is not None and best_result.success:
        min_x, min_y = best_result.x
        min_z = f_np(min_x, min_y)
    else:
        # 최적화 실패 시 기본 최적점 (원점)
        min_x, min_y = 0, 0
        try:
            min_z = f_np(min_x, min_y)
        except Exception:
            min_z = 0
    
    return min_x, min_y, min_z

# 경사하강법 경로 시각화 함수
def plot_gd(f_np, dx_np, dy_np, x_min, x_max, y_min, y_max, 
            gd_path, min_point, camera_eye, converged=False, converged_step=-1):
    # 그래프용 데이터 생성
    X = np.linspace(x_min, x_max, 80)
    Y = np.linspace(y_min, y_max, 80)
    Xs, Ys = np.meshgrid(X, Y)
    
    # 함수값 계산 (안전하게)
    try:
        Zs = f_np(Xs, Ys)
        # NaN이나 무한대 값 처리
        Zs = np.nan_to_num(Zs, nan=0, posinf=1e3, neginf=-1e3)
        # 극단적 값 클리핑
        z_range = max(100, np.percentile(Zs[~np.isinf(Zs) & ~np.isnan(Zs)], 99) - 
                     np.percentile(Zs[~np.isinf(Zs) & ~np.isnan(Zs)], 1))
        z_mid = np.median(Zs[~np.isinf(Zs) & ~np.isnan(Zs)])
        Zs = np.clip(Zs, z_mid - z_range, z_mid + z_range)
    except Exception:
        # 오류 발생 시 기본 Z값
        Zs = Xs**2 + Ys**2
    
    # 그래프 생성
    fig = go.Figure()
    
    # 3D 표면 추가
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Zs, 
        opacity=0.8, 
        colorscale='Viridis', 
        showscale=False,
        contours=dict(
            z=dict(
                show=True,
                usecolormap=True,
                highlightcolor="limegreen",
                project=dict(z=True)
            )
        )
    ))
    
    # 경사하강법 경로 계산 및 시각화
    try:
        px, py = zip(*gd_path)
        pz = [float(f_np(x, y)) for x, y in gd_path]
    except Exception:
        # 오류 발생 시 기본 경로
        px, py = zip(*gd_path)
        pz = [x**2 + y**2 for x, y in gd_path]
    
    # 경로 색상 설정 (수렴 시 초록색으로 변경)
    path_color = 'red'
    if converged and converged_step >= 0 and converged_step < len(gd_path) - 1:
        # 수렴 전은 빨간색, 수렴 후는 초록색
        fig.add_trace(go.Scatter3d(
            x=px[:converged_step+1], 
            y=py[:converged_step+1], 
            z=pz[:converged_step+1],
            mode='lines+markers',
            marker=dict(size=6, color='red'),
            line=dict(color='red', width=4),
            name="경로 (수렴 전)"
        ))
        fig.add_trace(go.Scatter3d(
            x=px[converged_step:], 
            y=py[converged_step:], 
            z=pz[converged_step:],
            mode='lines+markers',
            marker=dict(size=6, color='green'),
            line=dict(color='green', width=4),
            name="경로 (수렴 후)"
        ))
    else:
        # 일반 경로 시각화
        fig.add_trace(go.Scatter3d(
            x=px, y=py, z=pz,
            mode='lines+markers+text',
            marker=dict(size=6, color=path_color),
            line=dict(color=path_color, width=4),
            name="경로",
            text=[f"({x:.2f}, {y:.2f})" for x, y in gd_path],
            textposition="top center"
        ))
    
    # 경로 라벨 (첫 위치, 현재 위치)
    fig.add_trace(go.Scatter3d(
        x=[px[0]], y=[py[0]], z=[pz[0]],
        mode='markers+text',
        marker=dict(size=8, color='blue'),
        text=["시작점"],
        textposition="bottom center",
        name="시작점"
    ))
    
    # 그래디언트 화살표 (최대 15개, 스케일 자동 조정)
    arrow_points = min(15, len(gd_path) - 1)
    if len(gd_path) > 1:
        # 화살표 스케일 자동 계산 (그래디언트 크기에 따라 조정)
        gradients = []
        for i in range(-1, -min(arrow_points+1, len(gd_path)), -1):
            gx, gy = gd_path[i]
            try:
                grad_x = dx_np(gx, gy)
                grad_y = dy_np(gx, gy)
                grad_mag = np.sqrt(grad_x**2 + grad_y**2)
                gradients.append(grad_mag)
            except Exception:
                gradients.append(0.1)
        
        if gradients:
            median_grad = np.median(gradients)
            # 그래디언트 크기에 따라 화살표 스케일 조정
            arrow_scale = 0.3 / max(0.0001, median_grad)
            arrow_scale = min(arrow_scale, 1.0)  # 최대 스케일 제한
            
            for i in range(-1, -min(arrow_points+1, len(gd_path)), -1):
                gx, gy = gd_path[i]
                try:
                    gz = f_np(gx, gy)
                    grad_x = dx_np(gx, gy)
                    grad_y = dy_np(gx, gy)
                    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
                    
                    # 그래디언트 크기에 따른 화살표 색상 (빨간색→노란색→초록색)
                    arrow_color = 'red'
                    if grad_mag < convergence_threshold * 2:
                        arrow_color = 'green'
                    elif grad_mag < convergence_threshold * 5:
                        arrow_color = 'yellow'
                    
                    fig.add_trace(go.Cone(
                        x=[gx], y=[gy], z=[gz],
                        u=[-grad_x * arrow_scale],
                        v=[-grad_y * arrow_scale],
                        w=[0],
                        sizemode="absolute", 
                        sizeref=0.5,
                        colorscale=[[0, arrow_color], [1, arrow_color]], 
                        showscale=False,
                        anchor="tail", 
                        name="기울기"
                    ))
                except Exception:
                    continue
    
    # 최적점 표시
    min_x, min_y, min_z = min_point
    fig.add_trace(go.Scatter3d(
        x=[min_x], y=[min_y], z=[min_z],
        mode='markers+text',
        marker=dict(size=10, color='limegreen', symbol='diamond'),
        text=["최적점"],
        textposition="bottom center",
        name="최적점"
    ))
    
    # 현재 위치 표시
    if gd_path:
        last_x, last_y = gd_path[-1]
        try:
            last_z = f_np(last_x, last_y)
        except Exception:
            last_z = last_x**2 + last_y**2
            
        current_label = "현재 위치"
        if converged:
            current_label = "수렴점"
            
        fig.add_trace(go.Scatter3d(
            x=[last_x], y=[last_y], z=[last_z],
            mode='markers+text',
            marker=dict(
                size=10, 
                color='green' if converged else 'blue',
                symbol='circle'
            ),
            text=[current_label],
            textposition="top right",
            name=current_label
        ))
    
    # 레이아웃 설정
    fig.update_layout(
        scene=dict(
            xaxis_title='x', 
            yaxis_title='y', 
            zaxis_title='f(x, y)',
            camera=dict(eye=camera_eye),
            aspectratio=dict(x=1, y=1, z=0.8)
        ),
        width=800, 
        height=600, 
        margin=dict(l=10, r=10, t=30, b=10),
        title="경사하강법 경로 시각화",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)"
        ),
    )
    
    # 등고선 바닥에 추가
    fig.update_layout(
        scene=dict(
            xaxis=dict(showbackground=True, backgroundcolor='rgb(230, 230, 230)'),
            yaxis=dict(showbackground=True, backgroundcolor='rgb(230, 230, 230)'),
            zaxis=dict(showbackground=True, backgroundcolor='rgb(230, 230, 230)')
        )
    )
    
    return fig

# 컨트롤 버튼
st.subheader("4️⃣ 경사하강법 실행")
col_btn1, col_btn2, col_btn3, col_btn4 = st.columns([1, 1, 1, 1])
with col_btn1:
    step_btn = st.button("한 스텝 이동", help="경사하강법을 한 단계 진행합니다.")
with col_btn2:
    play_btn = st.button("▶ 전체 실행 (애니메이션)", key="playbtn", help="경사하강법을 애니메이션으로 끝까지 실행합니다.")
with col_btn3:
    stop_btn = st.button("⏹ 중지", key="stopbtn", help="실행 중인 애니메이션을 중지합니다.")
with col_btn4:
    reset_btn = st.button("🔄 초기화", key="resetbtn", help="모든 상태를 초기 상태로 되돌립니다.")

# 함수 및 그래디언트 설정
f_np, dx_np, dy_np, func_error = setup_function(func_input)

# 오류 발생 시 처리
if func_error:
    st.error(func_error)
    st.stop()

# 유효한 함수로 진행
try:
    # 최적점 찾기
    min_x, min_y, min_z = find_optimal_point(f_np, x_min, x_max, y_min, y_max)
    
    # 리셋 버튼 처리
    if reset_btn:
        st.session_state.gd_path = [(float(start_x), float(start_y))]
        st.session_state.gd_step = 0
        st.session_state.play = False
        st.session_state.converged = False
        st.session_state.converged_step = -1
    
    # 중지 버튼 처리
    if stop_btn:
        st.session_state.play = False
    
    # 한 스텝 이동
    if step_btn and st.session_state.gd_step < steps and not st.session_state.converged:
        curr_x, curr_y = st.session_state.gd_path[-1]
        next_x, next_y, grad_x, grad_y, grad_mag, converged = gradient_descent_step(
            curr_x, curr_y, dx_np, dy_np, learning_rate,
            x_min, x_max, y_min, y_max, convergence_threshold
        )
        
        # 수렴 여부 확인 및 저장
        if converged and not st.session_state.converged:
            st.session_state.converged = True
            st.session_state.converged_step = st.session_state.gd_step
        
        st.session_state.gd_path.append((next_x, next_y))
        st.session_state.gd_step += 1
    
    # 전체 실행 애니메이션
    if play_btn:
        st.session_state.play = True
    
    if st.session_state.play and st.session_state.gd_step < steps and not st.session_state.converged:
        # 애니메이션 실행을 위한 빈 컨테이너
        fig_placeholder = st.empty()
        status_placeholder = st.empty()
        
        for i in range(st.session_state.gd_step, steps):
            if not st.session_state.play:  # 중지 버튼 확인
                break
                
            curr_x, curr_y = st.session_state.gd_path[-1]
            next_x, next_y, grad_x, grad_y, grad_mag, converged = gradient_descent_step(
                curr_x, curr_y, dx_np, dy_np, learning_rate,
                x_min, x_max, y_min, y_max, convergence_threshold
            )
            
            # 수렴 여부 확인 및 저장
            if converged and not st.session_state.converged:
                st.session_state.converged = True
                st.session_state.converged_step = st.session_state.gd_step
                
            st.session_state.gd_path.append((next_x, next_y))
            st.session_state.gd_step += 1
            
            # 애니메이션 업데이트
            fig = plot_gd(
                f_np, dx_np, dy_np, x_min, x_max, y_min, y_max,
                st.session_state.gd_path, (min_x, min_y, min_z), 
                camera_eye, st.session_state.converged, st.session_state.converged_step
            )
            fig_placeholder.plotly_chart(fig, use_container_width=True, key=f"animation_chart_{i}")
            
            # 상태 정보 업데이트
            last_x, last_y = st.session_state.gd_path[-1]
            try:
                last_z = f_np(last_x, last_y)
                grad_x = dx_np(last_x, last_y)
                grad_y = dy_np(last_x, last_y)
                grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            except Exception:
                last_z = last_x**2 + last_y**2
                grad_x = last_x * 2
                grad_y = last_y * 2
                grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            
            # 상태 메시지 업데이트    
            status_msg = f"""
            **스텝 {st.session_state.gd_step}/{steps}** | 
            **위치:** ({last_x:.3f}, {last_y:.3f}) | 
            **함수값:** {last_z:.3f} | 
            **기울기 크기:** {grad_mag:.4f}
            """
            if converged:
                status_placeholder.success(f"{status_msg} ✅ 수렴!")
                # 수렴 시 애니메이션 중단
                st.session_state.play = False
                break
            else:
                status_placeholder.info(status_msg)
                
            # 애니메이션 속도 조절
            time.sleep(animation_speed)
            
        st.session_state.play = False
    
    # 그래프 및 상태 표시
    fig = plot_gd(
        f_np, dx_np, dy_np, x_min, x_max, y_min, y_max,
        st.session_state.gd_path, (min_x, min_y, min_z), 
        camera_eye, st.session_state.converged, st.session_state.converged_step
    )
    st.plotly_chart(fig, use_container_width=True, key="main_chart")
    
    # 현재 상태 정보 표시
    if st.session_state.gd_path:
        last_x, last_y = st.session_state.gd_path[-1]
        try:
            last_z = f_np(last_x, last_y)
            grad_x = dx_np(last_x, last_y)
            grad_y = dy_np(last_x, last_y)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        except Exception:
            last_z = last_x**2 + last_y**2
            grad_x = last_x * 2
            grad_y = last_y * 2
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # 최적점과의 비교
        try:
            min_z = f_np(min_x, min_y)
            distance_to_min = np.sqrt((last_x - min_x)**2 + (last_y - min_y)**2)
            value_diff = last_z - min_z
        except Exception:
            min_z = 0
            distance_to_min = np.sqrt((last_x - min_x)**2 + (last_y - min_y)**2)
            value_diff = 0
            
        # 상태 표시 - 수렴 여부에 따라 다른 스타일 적용
        if st.session_state.converged:
            st.success(
                f"""
                ✅ **수렴 완료!** (스텝 {st.session_state.converged_step+1}에서 수렴)
                
                **현재 위치:** ({last_x:.4f}, {last_y:.4f})  
                **현재 함수값:** {last_z:.4f}  
                **현재 기울기:** (∂f/∂x = {grad_x:.4f}, ∂f/∂y = {grad_y:.4f})  
                **기울기 크기:** {grad_mag:.6f} < {convergence_threshold} (임계값)
                
                **최적점까지 거리:** {distance_to_min:.4f}  
                **최적값과 차이:** {value_diff:.6f}
                """
            )
        else:
            if grad_mag < convergence_threshold * 5:
                st.info(
                    f"""
                    **현재 위치:** ({last_x:.4f}, {last_y:.4f})  
                    **현재 함수값:** {last_z:.4f}  
                    **현재 기울기:** (∂f/∂x = {grad_x:.4f}, ∂f/∂y = {grad_y:.4f})  
                    **기울기 크기:** {grad_mag:.6f} (수렴 임계값: {convergence_threshold})
                    
                    ℹ️ 기울기가 작아지고 있습니다! 알고리즘이 곧 수렴할 것으로 예상됩니다.
                    """
                )
            else:
                st.info(
                    f"""
                    **현재 위치:** ({last_x:.4f}, {last_y:.4f})  
                    **현재 함수값:** {last_z:.4f}  
                    **현재 기울기:** (∂f/∂x = {grad_x:.4f}, ∂f/∂y = {grad_y:.4f})  
                    **기울기 크기:** {grad_mag:.6f} (수렴 임계값: {convergence_threshold})
                    
                    **최적점까지 거리:** {distance_to_min:.4f}
                    """
                )
        
        # 성능 평가 섹션
        if st.session_state.gd_step > 0:
            st.subheader("5️⃣ 경사하강법 평가")
            
            col_eval1, col_eval2 = st.columns(2)
            
            with col_eval1:
                # 초기값과 현재값의 차이
                initial_x, initial_y = st.session_state.gd_path[0]
                try:
                    initial_z = f_np(initial_x, initial_y)
                    improvement = initial_z - last_z
                    improvement_percent = (improvement / abs(initial_z) * 100) if initial_z != 0 else 0
                except Exception:
                    initial_z = initial_x**2 + initial_y**2
                    improvement = initial_z - last_z
                    improvement_percent = (improvement / abs(initial_z) * 100) if initial_z != 0 else 0
                
                st.metric(
                    "함수값 개선", 
                    f"{last_z:.4f}", 
                    f"{improvement:.4f} ({improvement_percent:.1f}%)",
                    delta_color="inverse"
                )
                
                # 수렴 속도 측정
                if st.session_state.converged:
                    convergence_speed = f"{st.session_state.converged_step+1}스텝만에 수렴"
                    st.metric("수렴 속도", convergence_speed)
                else:
                    remaining_steps = "아직 수렴하지 않음"
                    st.metric("수렴 상태", remaining_steps)
            
            with col_eval2:
                # 최적점과의 비교
                try:
                    opt_z = f_np(min_x, min_y)
                    opt_diff = last_z - opt_z
                    opt_percent = (opt_diff / abs(opt_z) * 100) if opt_z != 0 else 0
                except Exception:
                    opt_z = 0
                    opt_diff = last_z
                    opt_percent = 100
                
                st.metric(
                    "최적값과의 차이", 
                    f"{last_z:.4f} vs {opt_z:.4f}", 
                    f"{opt_diff:.4f} ({opt_percent:.1f}%)",
                    delta_color="inverse"
                )
                
                # 그래디언트 크기
                grad_threshold_ratio = grad_mag / convergence_threshold
                grad_status = "수렴" if grad_threshold_ratio < 1 else f"임계값의 {grad_threshold_ratio:.1f}배"
                
                st.metric(
                    "기울기 크기", 
                    f"{grad_mag:.6f}", 
                    grad_status,
                    delta_color="inverse" if grad_threshold_ratio >= 1 else "normal"
                )

            # 학습 포인트 표시
            st.subheader("💡 학습 포인트")
            
            # 함수 유형별 메시지
            learning_messages = {
                "볼록 함수 (최적화 쉬움, 예: x²+y²)": 
                    "볼록 함수는 하나의 전역 최소값만 가지므로 경사하강법이 항상 최적해로 수렴합니다. 학습률 조정을 통해 수렴 속도를 제어할 수 있습니다.",
                "안장점 함수 (최적화 어려움, 예: x²-y²)": 
                    "안장점 함수는 (0,0)에서 한 방향으로는 증가하고 다른 방향으로는 감소합니다. 시작점에 따라 경로가 크게 달라지며, 실제 딥러닝에서 이런 지형은 학습을 어렵게 만듭니다.",
                "로젠브록 함수 (바나나 함수, 최적화 어려움)": 
                    "로젠브록 함수는 좁은 계곡 형태로, 일반적인 경사하강법이 최적화하기 어려운 지형입니다. 학습률이 너무 크면 발산하고, 너무 작으면 수렴이 매우 느립니다. 이는 실제 딥러닝에서 발생하는 어려운 최적화 문제와 유사합니다.",
                "사인 함수 (여러 지역 최소값)": 
                    "사인 함수는 여러 개의 지역 최소값을 가지고 있어 시작점에 따라 다른 최소값으로 수렴합니다. 딥러닝에서도 이러한 여러 지역 최소값 문제가 발생하며, 이를 해결하기 위해 모멘텀이나 학습률 조정과 같은 기법이 사용됩니다."
            }
            
            if func_radio in learning_messages:
                st.info(learning_messages[func_radio])
            else:
                st.info("사용자 정의 함수를 탐색해보세요. 함수의 특성에 따라 경사하강법의 성능이 크게 달라질 수 있습니다.")
            
            # 경사하강법 성능 분석
            if st.session_state.converged:
                if st.session_state.converged_step < 5:
                    st.success("🚀 **빠른 수렴**: 경사하강법이 매우 효율적으로 최적점에 도달했습니다. 이는 함수가 최적화하기 쉬운 형태이거나 학습률이 적절하게 설정되었음을 의미합니다.")
                elif distance_to_min < 0.1 and opt_diff < 0.1:
                    st.success("✅ **성공적인 최적화**: 경사하강법이 최적점에 매우 가깝게 수렴했습니다.")
                else:
                    st.warning("⚠️ **부분적 수렴**: 알고리즘이 수렴했지만 전역 최적점에 도달하지 못했을 수 있습니다. 다른 시작점이나 학습률을 시도해보세요.")
            elif st.session_state.gd_step >= steps:
                if grad_mag < convergence_threshold * 2:
                    st.warning("⚠️ **느린 수렴**: 기울기가 작아지고 있지만 최대 반복 횟수에 도달했습니다. 더 많은 스텝을 실행하거나 수렴 임계값을 높이세요.")
                else:
                    st.error("❌ **수렴 실패**: 최대 반복 횟수에 도달했지만 알고리즘이 수렴하지 않았습니다. 학습률을 조정하거나 다른 시작점을 시도해보세요.")
            elif grad_mag > 100:
                st.error("❌ **발산 가능성**: 그래디언트 크기가 매우 큽니다. 학습률이 너무 크거나 함수가 불안정한 영역에 있을 수 있습니다.")
            else:
                st.info("🔄 **최적화 진행 중**: 계속해서 '한 스텝 이동' 또는 '전체 실행' 버튼을 눌러 경사하강법을 진행하세요.")

except Exception as e:
    st.error(f"오류가 발생했습니다: {str(e)}")
    st.info("함수나 파라미터를 변경하고 다시 시도해보세요. 특히 사용자 정의 함수를 입력한 경우 구문이 올바른지 확인하세요.")

# 설명 섹션
with st.expander("경사하강법 알고리즘 설명"):
    st.markdown("""
    ### 경사하강법 알고리즘

    경사하강법은 다음과 같은 간단한 알고리즘으로 구현됩니다:

    1. 초기 위치 $(x_0, y_0)$를 선택합니다.
    2. $t = 0, 1, 2, ...$에 대해 다음을 반복합니다:
       * 현재 위치 $(x_t, y_t)$에서 함수의 그래디언트 $\\nabla f(x_t, y_t) = (\\frac{\\partial f}{\\partial x}, \\frac{\\partial f}{\\partial y})$를 계산합니다.
       * 그래디언트 방향의 반대 방향으로 이동: $(x_{t+1}, y_{t+1}) = (x_t, y_t) - \\eta \\nabla f(x_t, y_t)$ 
       * 여기서 $\\eta$는 학습률(learning rate)입니다.
    3. 그래디언트의 크기 $\\|\\nabla f(x_t, y_t)\\|$가 충분히 작아지거나 최대 반복 횟수에 도달하면 종료합니다.

    ### 학습률의 영향
    * 학습률이 크면: 빠르게 이동하지만 최적점을 지나치거나 발산할 위험이 있습니다.
    * 학습률이 작으면: 안정적이지만 수렴이 느립니다.

    ### 딥러닝에서의 응용
    신경망 학습에서 경사하강법은 손실 함수(loss function)를 최소화하는 모델 파라미터를 찾는 데 사용됩니다.
    이 데모에서 x, y 좌표는 실제 딥러닝에서는 수백만 개의 모델 파라미터에 해당합니다.
    """)

# 추가 정보 및 저작권
st.caption("제작: 서울고 송석리 선생님 | 교육적 개선 버전")
