# claude 3.7 sonnet
import streamlit as st
from sympy import symbols, diff, sympify, lambdify
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize
import time

# ----- 애플리케이션 설정 및 메타데이터 -----
st.set_page_config(
    layout="wide", 
    page_title="경사 하강법 학습도구",
    page_icon="🎢"
)

st.title("🎢 딥러닝 경사하강법 시각화 학습도구")

# ----- 1. 상수 및 기본 옵션 정의 -----
# 각 함수에 대한 관점 및 파라미터 프리셋 정의
PRESETS = {
    "볼록 함수 (최적화 쉬움, 예: x²+y²)": {
        "formula": "x**2 + y**2",
        "x_range": (-6.0, 6.0),
        "y_range": (-6.0, 6.0),
        "start_x": 5.0,
        "start_y": -4.0,
        "learning_rate": 0.1,
        "steps": 25,
        "camera_angle": "정면(x+방향)",
        "educational_tip": "볼록 함수는 하나의 전역 최소값을 가지며, 경사 하강법이 항상 이 최소값으로 수렴합니다."
    },
    "안장점 함수 (예: 0.3x²-0.3y²)": {
        "formula": "0.3*x**2 - 0.3*y**2",
        "x_range": (-4.0, 4.0),
        "y_range": (-4.0, 4.0),
        "start_x": 4.0,
        "start_y": 0.0,
        "learning_rate": 0.1,
        "steps": 40,
        "camera_angle": "정면(y+방향)",
        "educational_tip": "안장점 함수는 일부 방향으로는 아래로 볼록하고 다른 방향으로는 위로 볼록합니다. 안장점에서는 모든 방향의 미분이 0이지만 최소값은 아닙니다."
    },
    "Himmelblau 함수 (다중 최적점)": {
        "formula": "(x**2 + y - 11)**2 + (x + y**2 - 7)**2",
        "x_range": (-6.0, 6.0),
        "y_range": (-6.0, 6.0),
        "start_x": 1.0,
        "start_y": 1.0,
        "learning_rate": 0.01,
        "steps": 60,
        "camera_angle": "사선(전체 보기)",
        "educational_tip": "Himmelblau 함수는 최적화 테스트에 자주 사용되며, 4개의 국소 최소값이 있습니다. 시작점에 따라 다른 최소값으로 수렴합니다."
    },
    "복잡한 함수 (Rastrigin 유사)": {
        "formula": "20 + (x**2 - 10*cos(2*3.14159*x)) + (y**2 - 10*cos(2*3.14159*y))",
        "x_range": (-5.0, 5.0),
        "y_range": (-5.0, 5.0),
        "start_x": 3.5,
        "start_y": -2.5,
        "learning_rate": 0.02,
        "steps": 70,
        "camera_angle": "사선(전체 보기)", 
        "educational_tip": "Rastrigin 함수는 여러 개의 국소 최소값을 가진 복잡한 함수로, 최적화 알고리즘이 쉽게 지역 최소값에 갇힐 수 있습니다."
    },
    "사용자 정의 함수 입력": {
        "formula": "",
        "x_range": (-6.0, 6.0),
        "y_range": (-6.0, 6.0),
        "start_x": 5.0,
        "start_y": -4.0,
        "learning_rate": 0.1,
        "steps": 25,
        "camera_angle": "정면(x+방향)",
        "educational_tip": "자신만의 함수를 입력하여 경사 하강법의 동작을 탐구해보세요. 다양한 학습률과 시작점으로 실험해보는 것이 좋습니다."
    }
}

# 카메라 각도 옵션 정의
CAMERA_ANGLES = {
    "사선(전체 보기)": dict(x=1.7, y=1.7, z=1.2),
    "정면(x+방향)": dict(x=2.0, y=0.0, z=0.5), 
    "정면(y+방향)": dict(x=0.0, y=2.0, z=0.5),
    "위에서 내려다보기": dict(x=0.0, y=0.0, z=3.0),
    "뒤쪽(x-방향)": dict(x=-2.0, y=0.0, z=0.5),
    "옆(y-방향)": dict(x=0.0, y=-2.0, z=0.5)
}

# ----- 2. 세션 상태 초기화 및 관리 함수 -----
def initialize_session_state():
    """세션 상태 변수 초기화"""
    # 기본 함수 선택
    if "selected_func_type" not in st.session_state:
        st.session_state.selected_func_type = list(PRESETS.keys())[0]
    
    # 사용자 함수 입력 상태
    if "user_func_input" not in st.session_state:
        st.session_state.user_func_input = "x**2 + y**2"
    
    # 전체 파라미터 초기화 (필요한 경우)
    preset_params = PRESETS[st.session_state.selected_func_type]
    params_to_initialize = {
        "x_min_max_slider": "x_range",
        "y_min_max_slider": "y_range",
        "start_x_slider": "start_x",
        "start_y_slider": "start_y",
        "learning_rate_input": "learning_rate",
        "steps_slider": "steps",
        "selected_camera_option_name": "camera_angle"
    }
    
    for session_key, preset_key in params_to_initialize.items():
        if session_key not in st.session_state:
            st.session_state[session_key] = preset_params[preset_key]
    
    # 경사하강법 경로 초기화
    update_gd_path_if_needed()

def update_gd_path_if_needed():
    """파라미터 변경 시 경사하강법 경로 초기화"""
    current_func = get_current_function_string()
    start_x = st.session_state.start_x_slider
    start_y = st.session_state.start_y_slider
    learning_rate = st.session_state.learning_rate_input
    
    # 주요 파라미터가 변경되었는지 확인
    if ("gd_path" not in st.session_state or
        st.session_state.get("last_func_eval", "") != current_func or
        st.session_state.get("last_start_x_eval", 0.0) != start_x or
        st.session_state.get("last_start_y_eval", 0.0) != start_y or
        st.session_state.get("last_lr_eval", 0.0) != learning_rate):
        
        # 경로 초기화
        st.session_state.gd_path = [(float(start_x), float(start_y))]
        st.session_state.gd_step = 0
        st.session_state.play = False
        
        # 현재 상태 저장
        st.session_state.last_func_eval = current_func
        st.session_state.last_start_x_eval = start_x
        st.session_state.last_start_y_eval = start_y
        st.session_state.last_lr_eval = learning_rate
        st.session_state.animation_camera_eye = CAMERA_ANGLES[st.session_state.selected_camera_option_name]
        st.session_state.messages = []
        st.session_state.educational_logs = []

def apply_preset_for_func_type(func_type_name):
    """함수 유형에 맞는 프리셋 적용"""
    preset = PRESETS[func_type_name]
    
    # 프리셋 값 적용
    st.session_state.x_min_max_slider = preset["x_range"]
    st.session_state.y_min_max_slider = preset["y_range"]
    st.session_state.start_x_slider = preset["start_x"]
    st.session_state.start_y_slider = preset["start_y"]
    st.session_state.learning_rate_input = preset["learning_rate"]
    st.session_state.steps_slider = preset["steps"]
    st.session_state.selected_camera_option_name = preset["camera_angle"]
    
    # 사용자 정의 함수인 경우 입력 유지
    if func_type_name != "사용자 정의 함수 입력":
        st.session_state.user_func_input = preset["formula"]
    
    # 시작점이 새 범위 내에 있도록 조정
    new_x_min, new_x_max = st.session_state.x_min_max_slider
    new_y_min, new_y_max = st.session_state.y_min_max_slider
    st.session_state.start_x_slider = max(new_x_min, min(new_x_max, st.session_state.start_x_slider))
    st.session_state.start_y_slider = max(new_y_min, min(new_y_max, st.session_state.start_y_slider))

def get_current_function_string():
    """현재 선택된 함수식 문자열 반환"""
    if st.session_state.selected_func_type == "사용자 정의 함수 입력":
        return st.session_state.user_func_input
    else:
        return PRESETS[st.session_state.selected_func_type]["formula"]

def get_current_educational_tip():
    """현재 선택된 함수에 대한 교육적 팁 반환"""
    return PRESETS[st.session_state.selected_func_type]["educational_tip"]

# ----- 3. 수학적 함수 계산 및 시각화 함수 -----
def prepare_function_and_gradients(func_input):
    """함수 문자열로부터 함수와 기울기 함수 생성"""
    x_sym, y_sym = symbols('x y')
    
    try:
        # 함수식 파싱 및 변환
        f_sym_parsed = sympify(func_input)
        f_np_parsed = lambdify(
            (x_sym, y_sym), 
            f_sym_parsed, 
            modules=['numpy', {'cos': np.cos, 'sin': np.sin, 'exp': np.exp, 'sqrt': np.sqrt, 'pi': np.pi}]
        )
        
        # 미분 계산
        dx_f_sym_parsed = diff(f_sym_parsed, x_sym)
        dy_f_sym_parsed = diff(f_sym_parsed, y_sym)
        
        # 넘파이 함수로 변환
        dx_np_parsed = lambdify(
            (x_sym, y_sym), 
            dx_f_sym_parsed, 
            modules=['numpy', {'cos': np.cos, 'sin': np.sin, 'exp': np.exp, 'sqrt': np.sqrt, 'pi': np.pi}]
        )
        dy_np_parsed = lambdify(
            (x_sym, y_sym), 
            dy_f_sym_parsed, 
            modules=['numpy', {'cos': np.cos, 'sin': np.sin, 'exp': np.exp, 'sqrt': np.sqrt, 'pi': np.pi}]
        )
        
        return f_np_parsed, dx_np_parsed, dy_np_parsed, None
    except Exception as e:
        return None, None, None, str(e)

def find_scipy_minimum(f_np_func, start_x, start_y, func_type):
    """SciPy 최적화 함수를 사용하여 최소값 찾기"""
    try:
        def min_func_scipy(vars_list): 
            return f_np_func(vars_list[0], vars_list[1])
        
        # 여러 시작점에서 최적화 시도
        potential_starts = [[0.0, 0.0], [float(start_x), float(start_y)]]
        if "Himmelblau" in func_type:
            # Himmelblau 함수의 알려진 최소점 근처에서 시작
            potential_starts.extend([[3, 2], [-2.805, 3.131], [-3.779, -3.283], [3.584, -1.848]])
        
        best_res = None
        for p_start in potential_starts:
            res_temp = minimize(
                min_func_scipy, 
                p_start, 
                method='Nelder-Mead', 
                tol=1e-6, 
                options={'maxiter': 200, 'adaptive': True}
            )
            if best_res is None or (res_temp.success and res_temp.fun < best_res.fun) or (res_temp.success and not best_res.success):
                best_res = res_temp
        
        if best_res and best_res.success:
            min_x_sp, min_y_sp = best_res.x
            min_z_sp = f_np_func(min_x_sp, min_y_sp)
            return (min_x_sp, min_y_sp, min_z_sp), None
        else:
            return None, "SciPy 최적점을 찾지 못했습니다."
    except Exception as e:
        return None, f"SciPy 오류: {str(e)[:100]}..."

def plot_gd(f_np_func, dx_np_func, dy_np_func, x_range, y_range, gd_path, 
            min_point_scipy, current_camera_eye, educational_mode=False):
    """경사 하강법 경로 및 함수 표면 플롯팅"""
    x_min, x_max = x_range
    y_min, y_max = y_range
    
    # 그래프 데이터 준비
    X_plot = np.linspace(x_min, x_max, 80)
    Y_plot = np.linspace(y_min, y_max, 80)
    Xs_plot, Ys_plot = np.meshgrid(X_plot, Y_plot)
    
    try: 
        Zs_plot = f_np_func(Xs_plot, Ys_plot)
    except Exception: 
        Zs_plot = np.zeros_like(Xs_plot)
    
    # 그래프 객체 생성
    fig = go.Figure()
    
    # 함수 표면 추가
    fig.add_trace(go.Surface(
        x=X_plot, y=Y_plot, z=Zs_plot, 
        opacity=0.7, 
        colorscale='Viridis',
        contours_z=dict(
            show=True, 
            usecolormap=True, 
            highlightcolor="limegreen", 
            project_z=True
        ),
        name="함수 표면 f(x,y)", 
        showscale=False
    ))
    
    # 경사 하강 경로 데이터 준비
    px, py = zip(*gd_path)
    try: 
        pz = [f_np_func(pt_x, pt_y) for pt_x, pt_y in gd_path]
    except Exception: 
        pz = [np.nan_to_num(f_np_func(pt_x, pt_y)) for pt_x, pt_y in gd_path]
    
    # 경로 텍스트 준비 (교육 모드에서는 더 자세한 정보 표시)
    if educational_mode and len(gd_path) > 1:
        path_texts = []
        for idx, ((pt_x, pt_y), pt_z) in enumerate(zip(gd_path, pz)):
            if idx == 0:
                path_texts.append(f"시작점<br>({pt_x:.2f}, {pt_y:.2f})<br>f={pt_z:.2f}")
            elif idx == len(gd_path) - 1:
                path_texts.append(f"현재점<br>({pt_x:.2f}, {pt_y:.2f})<br>f={pt_z:.2f}")
            else:
                path_texts.append(f"S{idx}<br>({pt_x:.2f}, {pt_y:.2f})<br>f={pt_z:.2f}")
    else:
        path_texts = [f"S{idx}<br>({pt_x:.2f}, {pt_y:.2f})" for idx, (pt_x, pt_y) in enumerate(gd_path)]
    
    # 경로 트레이스 추가
    fig.add_trace(go.Scatter3d(
        x=px, y=py, z=pz, 
        mode='lines+markers+text',
        marker=dict(
            size=5, 
            color='red', 
            symbol='circle',
            colorscale=[[0, 'pink'], [1, 'red']],  # 시작점에서 현재점까지 색상 그라데이션
            showscale=False
        ), 
        line=dict(color='red', width=3),
        name="경사 하강 경로", 
        text=path_texts, 
        textposition="top right", 
        textfont=dict(size=10, color='black')
    ))
    
    # 기울기 화살표 추가
    arrow_scale_factor = 0.3
    num_arrows_to_show = min(5, len(gd_path) - 1)
    if num_arrows_to_show > 0:
        for i in range(num_arrows_to_show):
            arrow_start_idx = len(gd_path) - 1 - i - 1
            if arrow_start_idx < 0: 
                continue
                
            gx, gy = gd_path[arrow_start_idx]
            try:
                gz = f_np_func(gx, gy)
                grad_x_arrow = dx_np_func(gx, gy)
                grad_y_arrow = dy_np_func(gx, gy)
                
                if not (np.isnan(grad_x_arrow) or np.isnan(grad_y_arrow) or np.isnan(gz)):
                    # 기울기 벡터(경사) 화살표
                    fig.add_trace(go.Cone(
                        x=[gx], y=[gy], 
                        z=[gz + 0.02 * np.abs(gz) if gz != 0 else 0.02],
                        u=[-grad_x_arrow * arrow_scale_factor], 
                        v=[-grad_y_arrow * arrow_scale_factor], 
                        w=[0], 
                        sizemode="absolute", 
                        sizeref=0.25, 
                        colorscale=[[0, 'magenta'], [1, 'magenta']], 
                        showscale=False, 
                        anchor="tail", 
                        name=f"기울기 S{arrow_start_idx}" if i == 0 else "", 
                        hoverinfo='skip',
                        opacity = 0.15
                    ))
                    
                    # 교육 모드에서는 추가 정보 표시
                    if educational_mode and i == 0:
                        grad_mag = np.sqrt(grad_x_arrow**2 + grad_y_arrow**2)
                        fig.add_annotation(
                            x=gx, y=gy, z=gz + 0.5,
                            text=f"기울기 크기: {grad_mag:.2f}",
                            showarrow=True,
                            arrowhead=2,
                            arrowcolor="magenta",
                            arrowwidth=2,
                            ax=20, ay=-40
                        )
            except Exception: 
                continue
    
    # SciPy 최적점 추가
    if min_point_scipy:
        min_x_sp, min_y_sp, min_z_sp = min_point_scipy
        fig.add_trace(go.Scatter3d(
            x=[min_x_sp], y=[min_y_sp], z=[min_z_sp], 
            mode='markers+text',
            marker=dict(size=10, color='cyan', symbol='diamond'),
            text=["SciPy 최적점"], 
            textposition="bottom center", 
            name="SciPy 최적점"
        ))
    
    # 현재 GD 위치 강조
    last_x_gd, last_y_gd = gd_path[-1]
    try: 
        last_z_gd = f_np_func(last_x_gd, last_y_gd)
    except Exception: 
        last_z_gd = np.nan
    
    fig.add_trace(go.Scatter3d(
        x=[last_x_gd], y=[last_y_gd], 
        z=[last_z_gd if not np.isnan(last_z_gd) else Zs_plot.min()], 
        mode='markers+text',
        marker=dict(
            size=8, 
            color='orange', 
            symbol='circle', 
            line=dict(color='black', width=1)
        ),
        text=["현재 위치"], 
        textposition="top left", 
        name="GD 현재 위치"
    ))
    
    # 그래프 레이아웃 설정
    fig.update_layout(
        scene=dict(
            xaxis_title='x', 
            yaxis_title='y', 
            zaxis_title='f(x, y)',
            camera=dict(eye=current_camera_eye),
            aspectmode='cube'
        ),
        height=600, 
        margin=dict(l=0, r=0, t=30, b=0),
        title_text="경사 하강법 경로 및 함수 표면", 
        title_x=0.5,
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.02, 
            xanchor="right", 
            x=1
        )
    )
    
    # 교육 모드에서 추가적인 설명 추가
    if educational_mode and len(gd_path) > 1:
        # 최근 스텝에 대한 정보 추가
        if len(gd_path) >= 2:
            current_x, current_y = gd_path[-1]
            prev_x, prev_y = gd_path[-2]
            try:
                current_z = f_np_func(current_x, current_y)
                prev_z = f_np_func(prev_x, prev_y)
                grad_x = dx_np_func(prev_x, prev_y)
                grad_y = dy_np_func(prev_x, prev_y)
                grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                
                # 함수값 변화에 대한 주석
                change = current_z - prev_z
                change_text = f"함수값 변화: {change:.4f}"
                color = "green" if change < 0 else "red"
                
                fig.add_annotation(
                    x=(current_x + prev_x)/2, 
                    y=(current_y + prev_y)/2,
                    z=(current_z + prev_z)/2 + 0.5,
                    text=change_text,
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor=color,
                    arrowwidth=2,
                    ax=0, ay=-40
                )
            except Exception:
                pass
    
    return fig

# ----- 4. 경사 하강법 알고리즘 구현 -----
def gradient_descent_step(f_np_func, dx_np_func, dy_np_func, current_point, learning_rate):
    """경사 하강법 한 스텝 실행"""
    curr_x, curr_y = current_point
    
    try:
        # 기울기 계산
        grad_x_val = dx_np_func(curr_x, curr_y)
        grad_y_val = dy_np_func(curr_x, curr_y)
        
        # NaN 체크
        if np.isnan(grad_x_val) or np.isnan(grad_y_val):
            return None, "기울기 계산 결과가 NaN입니다."
        
        # 다음 위치 계산
        next_x = curr_x - learning_rate * grad_x_val
        next_y = curr_y - learning_rate * grad_y_val
        
        # 교육적 로그 정보
        current_value = f_np_func(curr_x, curr_y)
        next_value = f_np_func(next_x, next_y)
        grad_magnitude = np.sqrt(grad_x_val**2 + grad_y_val**2)
        
        log_info = {
            "step": st.session_state.gd_step + 1,
            "current_point": (curr_x, curr_y),
            "current_value": current_value,
            "gradient": (grad_x_val, grad_y_val),
            "gradient_magnitude": grad_magnitude,
            "next_point": (next_x, next_y),
            "next_value": next_value,
            "improvement": current_value - next_value
        }
        
        return (next_x, next_y), log_info
    except Exception as e:
        return None, f"스텝 진행 중 오류: {e}"

# ----- 5. UI 구성 함수 -----
def create_sidebar():
    """사이드바 UI 구성"""
    with st.sidebar:
        st.header("⚙️ 설정 및 파라미터")
        
        # 교육적 설명 섹션
        with st.expander("💡 경사 하강법이란?", expanded=False):
            st.markdown("""
            **경사 하강법(Gradient Descent)**은 함수의 최소값을 찾기 위한 최적화 알고리즘입니다.
            
            ### 기본 원리
            1. 현재 위치에서 함수의 기울기(경사, gradient)를 계산합니다
            2. 기울기의 반대 방향으로 일정 거리(학습률)만큼 이동합니다
            3. 새 위치에서 다시 기울기를 계산하고 반복합니다
            4. 기울기가 매우 작아지면(거의 평평한 지점) 최소값에 도달한 것으로 간주합니다
            
            ### 딥러닝에서의 역할
            신경망 학습에서 손실 함수(loss function)의 최소값을 찾는 데 사용됩니다.
            모델의 가중치(weights)를 점진적으로 조정하여 오차를 최소화합니다.
            """)
        
        with st.expander("📖 주요 파라미터 가이드", expanded=False):
            st.markdown(f"""
            ### 학습률(Learning Rate, α)
            - **역할**: 각 단계에서 이동할 거리를 결정합니다
            - **너무 크면**: 최소값을 지나치거나 발산할 수 있습니다
            - **너무 작으면**: 수렴이 매우 느리거나 지역 최소값에 갇힐 수 있습니다
            - **권장 범위**: 0.001 ~ 0.1 사이에서 시작하는 것이 좋습니다
            
            ### 시작 위치(Starting Point)
            - 시작 위치에 따라 다른 최소값에 도달할 수 있습니다
            - 특히 여러 최소값을 가진 함수에서 중요합니다
            
            ### 반복 횟수(Iterations)
            - 알고리즘이 실행될 최대 횟수입니다
            - 충분한 반복으로 수렴할 시간을 주되, 너무 많으면 불필요한 계산을 하게 됩니다
            """)
        
        # 함수 및 그래프 설정
        st.subheader("📊 함수 및 그래프 설정")
        
        # 현재 설정된 값으로 라디오 버튼 초기화
        st.radio(
            "그래프 시점(카메라 각도)",
            options=list(CAMERA_ANGLES.keys()),
            index=list(CAMERA_ANGLES.keys()).index(st.session_state.selected_camera_option_name),
            key="camera_angle_radio_key_widget", 
            on_change=lambda: setattr(st.session_state, "selected_camera_option_name", 
                                    st.session_state.camera_angle_radio_key_widget)
        )
        
        # 함수 유형 선택 UI
        st.radio(
            "함수 유형",
            options=list(PRESETS.keys()),
            index=list(PRESETS.keys()).index(st.session_state.selected_func_type),
            key="func_radio_key_widget", 
            on_change=handle_func_type_change
        )
        
        # 사용자 정의 함수 입력 필드
        if st.session_state.selected_func_type == "사용자 정의 함수 입력":
            st.text_input(
                "함수 f(x, y) 입력", 
                value=st.session_state.user_func_input,
                help="Python 구문으로 함수를 입력하세요. 예: x**2 + y**2, x*sin(y) 등",
                key="user_func_text_input_key_widget", 
                on_change=lambda: setattr(st.session_state, "user_func_input", 
                                        st.session_state.user_func_text_input_key_widget)
            )
        else:
            # 선택된 함수 표시 (읽기 전용)
            func_formula = PRESETS[st.session_state.selected_func_type]["formula"]
            st.text_input("선택된 함수 f(x, y)", value=func_formula, disabled=True)
        
        # 함수에 대한 교육적 설명 표시
        st.info(get_current_educational_tip())
        
        # x, y 범위 설정
        st.slider(
            "x 범위", -10.0, 10.0, st.session_state.x_min_max_slider, step=0.1, 
            key="x_slider_key_widget", 
            on_change=lambda: setattr(st.session_state, "x_min_max_slider", 
                                     st.session_state.x_slider_key_widget)
        )
        st.slider(
            "y 범위", -10.0, 10.0, st.session_state.y_min_max_slider, step=0.1, 
            key="y_slider_key_widget", 
            on_change=lambda: setattr(st.session_state, "y_min_max_slider", 
                                     st.session_state.y_slider_key_widget)
        )
        
        # 경사 하강법 파라미터
        st.subheader("🔩 경사 하강법 파라미터")
        
        # 현재 x, y 범위에 맞게 시작점 슬라이더 범위 설정
        current_x_min_ui, current_x_max_ui = st.session_state.x_min_max_slider
        current_y_min_ui, current_y_max_ui = st.session_state.y_min_max_slider
        
        st.slider(
            "시작 x 위치", float(current_x_min_ui), float(current_x_max_ui), 
            st.session_state.start_x_slider, step=0.1, 
            key="start_x_key_widget", 
            on_change=lambda: setattr(st.session_state, "start_x_slider", 
                                     st.session_state.start_x_key_widget)
        )
        st.slider(
            "시작 y 위치", float(current_y_min_ui), float(current_y_max_ui), 
            st.session_state.start_y_slider, step=0.1, 
            key="start_y_key_widget", 
            on_change=lambda: setattr(st.session_state, "start_y_slider", 
                                     st.session_state.start_y_key_widget)
        )
        
        # 학습률 설정
        st.number_input(
            "학습률 (Learning Rate, α)", 
            min_value=0.0001, max_value=1.0, 
            value=st.session_state.learning_rate_input, 
            step=0.001, format="%.4f", 
            help="각 스텝에서 경사 방향으로 얼마나 이동할지 결정합니다. 너무 크면 발산하고, 너무 작으면 수렴이 느립니다.",
            key="lr_key_widget", 
            on_change=lambda: setattr(st.session_state, "learning_rate_input", 
                                     st.session_state.lr_key_widget)
        )
        
        # 반복 횟수 설정
        st.slider(
            "최대 반복 횟수", 1, 100, st.session_state.steps_slider, 
            help="경사 하강법을 몇 번 반복할지 설정합니다.", 
            key="steps_key_widget", 
            on_change=lambda: setattr(st.session_state, "steps_slider", 
                                     st.session_state.steps_key_widget)
        )
        
        # 교육 모드 설정
        st.checkbox(
            "교육 모드 (자세한 정보 표시)", 
            value=st.session_state.get("educational_mode", False),
            help="함수값 변화, 기울기 크기 등 학습에 도움이 되는 추가 정보를 표시합니다",
            key="educational_mode_checkbox",
            on_change=lambda: setattr(st.session_state, "educational_mode", 
                                     st.session_state.educational_mode_checkbox)
        )
        
        # SciPy 최적화 결과 섹션
        st.subheader("🔬 SciPy 최적화 결과 (참고용)")
        scipy_result_placeholder = st.empty()
        
        return scipy_result_placeholder

def create_main_interface():
    """메인 인터페이스 구성"""
    # 구분선
    st.markdown("---")
    
    # 제어 버튼
    col_btn1, col_btn2, col_btn3, col_btn4 = st.columns([1, 1, 1, 1])
    
    with col_btn1: 
        step_btn = st.button("🚶 한 스텝 진행", use_container_width=True)
    with col_btn2: 
        play_btn = st.button("▶️ 전체 실행", key="playbtn_widget_key", use_container_width=True)
    with col_btn3: 
        reset_btn = st.button("🔄 초기화", key="resetbtn_widget_key", use_container_width=True)
    with col_btn4:
        analytics_btn = st.button("📊 분석 보기", key="analytics_btn_key", use_container_width=True)
    
    # 그래프 표시 영역
    graph_placeholder = st.empty()
    
    # 분석 결과 표시 영역
    analytics_placeholder = st.empty()
    
    return step_btn, play_btn, reset_btn, analytics_btn, graph_placeholder, analytics_placeholder

# 함수 유형 변경 시 콜백
def handle_func_type_change():
    """함수 유형 변경 시 호출되는 콜백 함수"""
    new_func_type = st.session_state.func_radio_key_widget
    st.session_state.selected_func_type = new_func_type  # 먼저 selected_func_type 업데이트
    apply_preset_for_func_type(new_func_type)  # 그 다음, 이 새 func_type에 맞는 프리셋 적용

# ----- 6. 데이터 분석 및 시각화 함수 -----
def display_analytics(f_np_func, gd_path, logs):
    """경사 하강법 분석 결과 표시"""
    if not logs or len(logs) == 0:
        return "아직 경사 하강법을 실행하지 않았습니다. 먼저 '한 스텝 진행' 또는 '전체 실행' 버튼을 눌러보세요."
    
    # 분석 컨테이너 시작
    analytics_md = """
    ## 📈 경사 하강법 분석 결과
    
    ### 성능 요약
    """
    
    # 시작점과 최종점 정보
    start_x, start_y = gd_path[0]
    final_x, final_y = gd_path[-1]
    
    try:
        start_value = f_np_func(start_x, start_y)
        final_value = f_np_func(final_x, final_y)
        total_improvement = start_value - final_value
        
        analytics_md += f"""
        - **시작 위치**: ({start_x:.4f}, {start_y:.4f})
        - **시작 함수값**: {start_value:.4f}
        - **최종 위치**: ({final_x:.4f}, {final_y:.4f})
        - **최종 함수값**: {final_value:.4f}
        - **총 개선값**: {total_improvement:.4f} ({(total_improvement/start_value)*100:.2f}%)
        - **스텝 수**: {len(gd_path) - 1}
        """
        
        # 기울기 수렴 분석
        if len(logs) > 0:
            final_gradient_mag = logs[-1].get("gradient_magnitude", 0)
            analytics_md += f"- **최종 기울기 크기**: {final_gradient_mag:.6f}\n"
            
            if final_gradient_mag < 0.01:
                analytics_md += "- **수렴 상태**: ✅ 기울기가 매우 작아 최적점에 수렴했습니다\n"
            elif final_gradient_mag < 0.1:
                analytics_md += "- **수렴 상태**: ⚠️ 기울기가 작아지고 있으나 아직 완전히 수렴하지 않았습니다\n"
            else:
                analytics_md += "- **수렴 상태**: ❌ 기울기가 여전히 큽니다. 더 많은 반복이 필요합니다\n"
        
        # 학습 곡선 데이터 준비
        if len(logs) > 0:
            steps = [log.get("step", i+1) for i, log in enumerate(logs)]
            function_values = [log.get("current_value", 0) for log in logs]
            gradient_magnitudes = [log.get("gradient_magnitude", 0) for log in logs]
            improvements = [log.get("improvement", 0) for log in logs]
            
            # 학습 곡선 차트 추가
            analytics_md += """
            ### 학습 곡선
            
            아래 차트는 경사 하강법이 진행됨에 따른 주요 지표의 변화를 보여줍니다.
            """
            
            # 데이터프레임 생성
            import pandas as pd
            df = pd.DataFrame({
                "스텝": steps,
                "함수값": function_values,
                "기울기 크기": gradient_magnitudes,
                "개선값": improvements
            })
            
            # 데이터프레임 표시
            analytics_md += "\n#### 스텝별 상세 데이터\n"
            return analytics_md, df
            
    except Exception as e:
        return f"분석 중 오류가 발생했습니다: {str(e)}", None
    
    return analytics_md, None

# ----- 7. 메인 애플리케이션 실행 -----
def main():
    """메인 애플리케이션 실행"""
    # 세션 상태 초기화
    initialize_session_state()
    
    # 교육 모드 초기화
    if "educational_mode" not in st.session_state:
        st.session_state.educational_mode = False
    
    if "educational_logs" not in st.session_state:
        st.session_state.educational_logs = []
    
    # 사이드바 생성
    scipy_result_placeholder = create_sidebar()
    
    # 메인 인터페이스 생성
    step_btn, play_btn, reset_btn, analytics_btn, graph_placeholder, analytics_placeholder = create_main_interface()
    
    # 현재 함수 준비
    current_func = get_current_function_string()
    f_np_func, dx_np_func, dy_np_func, func_error = prepare_function_and_gradients(current_func)
    
    if func_error:
        st.error(f"🚨 함수 정의 오류: {func_error}. 함수 수식을 확인해주세요.")
        st.stop()
    
    if not callable(f_np_func):
        st.error("함수 변환 실패.")
        st.stop()
    
    # SciPy 최적화 결과 계산
    min_point_scipy_coords, scipy_error = find_scipy_minimum(
        f_np_func, 
        st.session_state.start_x_slider, 
        st.session_state.start_y_slider,
        st.session_state.selected_func_type
    )
    
    if min_point_scipy_coords:
        min_x_sp, min_y_sp, min_z_sp = min_point_scipy_coords
        scipy_result_placeholder.markdown(
            f"""- **위치 (x, y)**: `({min_x_sp:.3f}, {min_y_sp:.3f})` <br> - **함수 값 f(x,y)**: `{min_z_sp:.4f}`""", 
            unsafe_allow_html=True
        )
    else:
        scipy_result_placeholder.info(scipy_error if scipy_error else "SciPy 최적점을 찾지 못했습니다.")
    
    # 버튼 동작 처리
    if reset_btn:
        # 기본 함수 유형으로 리셋
        st.session_state.selected_func_type = list(PRESETS.keys())[0]
        apply_preset_for_func_type(st.session_state.selected_func_type)
        st.session_state.user_func_input = "x**2 + y**2"
        
        # 현재 설정된 값 사용
        current_start_x_on_reset = st.session_state.start_x_slider
        current_start_y_on_reset = st.session_state.start_y_slider
        current_func_input_on_reset = PRESETS[st.session_state.selected_func_type]["formula"]
        
        # 경로 초기화
        st.session_state.gd_path = [(float(current_start_x_on_reset), float(current_start_y_on_reset))]
        st.session_state.gd_step = 0
        st.session_state.play = False
        st.session_state.animation_camera_eye = CAMERA_ANGLES[st.session_state.selected_camera_option_name]
        st.session_state.messages = []
        st.session_state.educational_logs = []
        
        # 현재 상태 저장
        st.session_state.last_func_eval = current_func_input_on_reset
        st.session_state.last_start_x_eval = current_start_x_on_reset
        st.session_state.last_start_y_eval = current_start_y_on_reset
        st.session_state.last_lr_eval = st.session_state.learning_rate_input
        
        st.rerun()
    
    # 한 스텝 진행 버튼
    if step_btn and st.session_state.gd_step < st.session_state.steps_slider:
        st.session_state.play = False
        
        # 경사 하강법 한 스텝 실행
        next_point, step_result = gradient_descent_step(
            f_np_func, 
            dx_np_func, 
            dy_np_func, 
            st.session_state.gd_path[-1], 
            st.session_state.learning_rate_input
        )
        
        if isinstance(step_result, dict):  # 성공적인 스텝
            st.session_state.gd_path.append(next_point)
            st.session_state.gd_step += 1
            st.session_state.educational_logs.append(step_result)
        else:  # 오류 발생
            st.session_state.messages.append(("error", step_result))
        
        st.rerun()
    
    # 전체 실행 버튼 - 애니메이션 대신 모든 계산을 즉시 수행
    if play_btn:
        # 메시지 초기화
        st.session_state.messages = []
        
        # 경로 초기화 - 시작점만 포함
        st.session_state.gd_path = [(float(st.session_state.start_x_slider), float(st.session_state.start_y_slider))]
        st.session_state.gd_step = 0
        st.session_state.educational_logs = []
        
        # 모든 스텝을 한번에 계산
        for _ in range(st.session_state.steps_slider):
            next_point, step_result = gradient_descent_step(
                f_np_func, 
                dx_np_func, 
                dy_np_func, 
                st.session_state.gd_path[-1], 
                st.session_state.learning_rate_input
            )
            
            if isinstance(step_result, dict):  # 성공적인 스텝
                st.session_state.gd_path.append(next_point)
                st.session_state.gd_step += 1
                st.session_state.educational_logs.append(step_result)
            else:  # 오류 발생
                st.session_state.messages.append(("error", step_result))
                break
        
        # 카메라 각도 설정
        st.session_state.animation_camera_eye = CAMERA_ANGLES[st.session_state.selected_camera_option_name]
        
        # 재실행하여 최종 결과 표시
        st.rerun()
    
    # 정적 그래프 표시
    current_display_cam = CAMERA_ANGLES[st.session_state.selected_camera_option_name]
    
    fig_static = plot_gd(
        f_np_func, 
        dx_np_func, 
        dy_np_func, 
        st.session_state.x_min_max_slider, 
        st.session_state.y_min_max_slider,
        st.session_state.gd_path, 
        min_point_scipy_coords, 
        current_display_cam,
        st.session_state.educational_mode
    )
    graph_placeholder.plotly_chart(fig_static, use_container_width=True, key="main_chart_static")
    
    # 분석 보기 버튼
    if analytics_btn:
        analytics_md, df = display_analytics(
            f_np_func, 
            st.session_state.gd_path, 
            st.session_state.educational_logs
        )
        
        with analytics_placeholder.container():
            st.markdown(analytics_md)
            if df is not None:
                st.dataframe(df)
                
                # 학습 곡선 차트
                st.subheader("🔍 학습 곡선 시각화")
                chart_tab1, chart_tab2, chart_tab3 = st.tabs(["함수값 변화", "기울기 크기 변화", "개선값 변화"])
                
                with chart_tab1:
                    st.line_chart(df, x="스텝", y="함수값")
                    st.caption("스텝이 진행됨에 따라 함수값이 감소하는 것이 이상적입니다.")
                
                with chart_tab2:
                    st.line_chart(df, x="스텝", y="기울기 크기")
                    st.caption("기울기 크기가 0에 가까워질수록 최적점에 근접한 것입니다.")
                
                with chart_tab3:
                    st.line_chart(df, x="스텝", y="개선값")
                    st.caption("각 스텝에서의 함수값 감소량입니다. 양수일수록 좋습니다.")
    
    # 메시지 표시
    temp_messages = st.session_state.get("messages", [])
    for msg_type, msg_content in temp_messages:
        if msg_type == "error":
            st.error(msg_content)
        elif msg_type == "warning":
            st.warning(msg_content)
        elif msg_type == "success":
            st.success(msg_content)
    
    # 메시지 초기화
    st.session_state.messages = []
        
    # 최종 상태 표시
    if len(st.session_state.gd_path) > 1:
        last_x_final, last_y_final = st.session_state.gd_path[-1]
        try:
            last_z_final = f_np_func(last_x_final, last_y_final)
            grad_x_final = dx_np_func(last_x_final, last_y_final)
            grad_y_final = dy_np_func(last_x_final, last_y_final)
            grad_norm_final = np.sqrt(grad_x_final**2 + grad_y_final**2)
            
            if np.isnan(last_z_final) or np.isinf(last_z_final):
                st.error("🚨 함수 값이 발산했습니다! (NaN 또는 무한대)")
            #elif st.session_state.gd_step >= st.session_state.steps_slider and grad_norm_final > 1e-2:
            #    st.warning(f"⚠️ 최대 반복({st.session_state.steps_slider}) 도달, 기울기({grad_norm_final:.4f})가 아직 충분히 작지 않음.")
            elif grad_norm_final < 1e-2 and not (np.isnan(grad_norm_final) or np.isinf(grad_norm_final)):
                st.success(f"🎉 기울기({grad_norm_final:.4f})가 매우 작아 최적점 또는 안장점에 근접했습니다!")
        except Exception:
            pass

# 애플리케이션 실행
if __name__ == "__main__":
    main()
