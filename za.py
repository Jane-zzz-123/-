import streamlit as st
import pandas as pd
import requests
from io import BytesIO
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from plotly.subplots import make_subplots
from math import ceil


# ------------------------------
# 新增：用户认证与权限管理
# ------------------------------
def check_credentials():
    """验证用户密码并返回用户权限（负责的店铺），用户名改为下拉框选择"""
    # 用户-密码-权限映射（建议实际使用时放在secrets中）
    USER_PERMISSIONS = {
        "黄怡": ("syc-huangyi123", ["思业成-US"]),  # 用户1能看的店铺
        "泽恒": ("dx-zeheng123", ["定行-US"]),  # 用户2能看的店铺
        "小娇": ("pt and ys-xiaojiao", ["拼途-US","艺胜-US"]),  # 用户3能看的店铺
        "楷纯": ("zy and cr-kaichun", ["争艳-US","辰瑞-US"]),  # 用户4能看的店铺
        "淑谊": ("sx and jy-shuyi", ["势兴-US","进益-US"]),  # 用户5能看的店铺
        "佰英": ("cq-baiying123", ["创奇-US"]),  # 用户6能看的店铺
        "李珊": ("dm-lishan123", ["大卖-US"]),  # 用户7能看的店铺
        "admin": ("admin1234", None)  # 管理员能看所有店铺
    }

    # 获取所有用户名作为下拉选项
    all_users = list(USER_PERMISSIONS.keys())

    def verify():
        # 从会话状态获取选择的用户名和输入的密码
        username = st.session_state.get("selected_user", "")
        password = st.session_state.get("password", "")

        if username in USER_PERMISSIONS:
            stored_pwd, stores = USER_PERMISSIONS[username]
            if password == stored_pwd:
                st.session_state["authenticated"] = True
                st.session_state["allowed_stores"] = stores  # 保存用户可访问的店铺
                del st.session_state["password"]  # 清除密码
            else:
                st.session_state["authenticated"] = False
        else:
            st.session_state["authenticated"] = False

    # 未认证状态：显示登录表单（下拉框选择用户名）
    if "authenticated" not in st.session_state or not st.session_state["authenticated"]:
        st.title("用户登录")

        # 用户名下拉框（而非输入框）
        st.selectbox(
            "请选择用户名",
            options=all_users,
            key="selected_user",
            on_change=verify  # 选择变化时触发验证
        )

        # 密码输入框
        st.text_input(
            "请输入密码",
            type="password",
            key="password",
            on_change=verify  # 输入密码时触发验证
        )

        # 显示错误信息
        if "authenticated" in st.session_state and not st.session_state["authenticated"]:
            st.error("密码错误，请重新输入")
        return False
    return True


# 验证不通过则终止
if not check_credentials():
    st.stop()

# 全局配置
st.set_page_config(page_title="年份品滞销风险分析仪表盘", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
    .dataframe th {font-size: 14px; text-align: center;}
    .dataframe td {font-size: 13px; text-align: center;}
    .metric-card {background-color: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; margin-bottom: 15px;}
    .metric-title {font-size: 15px; color: #666; margin-bottom: 5px;}
    .metric-value {font-size: 24px; font-weight: bold;}
    .metric-change {font-size: 14px;}
    .positive-change {color: #2E8B57;}
    .negative-change {color: #DC143C;}
    .neutral-change {color: #000000;}
    /* 固定侧边栏不滚动 */
    [data-testid="stSidebar"] {
        position: fixed;
        height: 100%;
        overflow: auto;
    }
</style>
""", unsafe_allow_html=True)

# 颜色配置
STATUS_COLORS = {
    "健康": "#2E8B57",  # 绿色
    "低滞销风险": "#4169E1",  # 蓝色
    "中滞销风险": "#FFD700",  # 黄色
    "高滞销风险": "#DC143C"  # 红色
}
TARGET_DATE = datetime(2025, 12, 1)  # 目标消耗完成日期
END_DATE = datetime(2025, 12, 31)  # 预测截止日期


# ------------------------------
# 1. 数据加载与预处理函数
# ------------------------------
@st.cache_data(ttl=3600)  # 记得保留缓存装饰器（如果之前有）
def load_and_preprocess_data_from_df(df):
    """加载Excel数据并进行预处理，包含所有列的计算逻辑"""
    try:
        # 1. 时间段系数配置（您已定义，保持不变）
        TIME_PERIODS = [
            {"name": "october_late", "start": datetime(2025, 10, 15), "end": datetime(2025, 10, 31),
             "coefficient": 0.95},
            {"name": "november_early", "start": datetime(2025, 11, 1), "end": datetime(2025, 11, 15),
             "coefficient": 0.91},
            {"name": "november_late", "start": datetime(2025, 11, 16), "end": datetime(2025, 11, 30),
             "coefficient": 0.72},
            {"name": "december", "start": datetime(2025, 12, 1), "end": datetime(2025, 12, 31), "coefficient": 0.43}
        ]
        # 2. 基础列检查与数据类型转换（您已实现，保持不变）
        required_base_cols = [
            "MSKU", "品名", "店铺", "记录时间", "日均",
            "FBA库存", "FBA在途", "海外仓在途", "本地可用",
            "待检待上架量", "待交付"
        ]
        missing_cols = [col for col in required_base_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Excel文件缺少必要的基础列：{', '.join(missing_cols)}")
            return None

        df["记录时间"] = pd.to_datetime(df["记录时间"]).dt.normalize()
        numeric_cols = ["日均", "FBA库存", "FBA在途", "海外仓在途",
                        "本地可用", "待检待上架量", "待交付"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # ------------------------------
        # 新增：生成4个时间段的系数列和调整后日均列
        # ------------------------------
        # 系数列（固定值，用于表格展示）
        df["10月15-31日系数"] = 0.95
        df["11月1-15日系数"] = 0.91
        df["11月16-30日系数"] = 0.72
        df["12月1-31日系数"] = 0.43

        # 调整后日均列（基础日均 × 对应系数，保留2位小数）
        df["10月15-31日调整后日均"] = (df["日均"] * 0.95).round(2)
        df["11月1-15日调整后日均"] = (df["日均"] * 0.91).round(2)
        df["11月16-30日调整后日均"] = (df["日均"] * 0.72).round(2)
        df["12月1-31日调整后日均"] = (df["日均"] * 0.43).round(2)

        # ------------------------------
        # 核心计算逻辑（修改：依赖分阶段调整后的日均）
        # ------------------------------
        # 1. FBA+AWD+在途库存（保持原逻辑）
        df["FBA+AWD+在途库存"] = (df["FBA库存"] + df["FBA在途"] + df["海外仓在途"]).round().astype(int)

        # 2. 全部总库存（保持原逻辑）
        df["全部总库存"] = (
                df["FBA+AWD+在途库存"] + df["本地可用"] + df["待检待上架量"] + df["待交付"]
        ).round().astype(int)

        # 3. 新增：分阶段计算库存耗尽日期的核心函数（替换原固定日均计算）
        def calculate_exhaust_date(row, stock_col):
            """
            按时间段系数计算库存耗尽日期
            stock_col: 库存列名（"FBA+AWD+在途库存" 或 "全部总库存"）
            """
            record_date = row["记录时间"]
            stock = row[stock_col]
            base_avg = row["日均"] if row["日均"] > 0 else 0.1  # 基础日均（避免为0）
            remaining_stock = stock
            current_date = record_date

            # 若库存为0，直接返回记录日期
            if remaining_stock <= 0:
                return record_date

            # 阶段1：记录日期 → 2025-10-14（系数=1.0）
            phase1_end = datetime(2025, 10, 14)
            if current_date <= phase1_end:
                days_in_phase = (phase1_end - current_date).days + 1  # 包含首尾日期
                sales_possible = base_avg * days_in_phase  # 此阶段无系数调整
                if remaining_stock <= sales_possible:
                    # 库存在此阶段耗尽，计算精确天数
                    days_needed = remaining_stock / base_avg
                    return current_date + pd.Timedelta(days=days_needed)
                # 库存未耗尽，扣除此阶段销量，进入下一阶段
                remaining_stock -= sales_possible
                current_date = phase1_end + pd.Timedelta(days=1)

            # 阶段2：处理4个特殊时间段（按时间顺序）
            for period in TIME_PERIODS:
                if current_date > period["end"] or remaining_stock <= 0:
                    break  # 超出当前时间段或库存已耗尽，跳过
                # 取当前时间段与剩余时间的交集（避免跨时间段计算错误）
                period_start = max(current_date, period["start"])
                if period_start > period["end"]:
                    continue
                # 计算此时间段内的可售天数和调整后日均
                days_in_period = (period["end"] - period_start).days + 1
                adjusted_avg = base_avg * period["coefficient"]  # 应用对应系数
                sales_possible = adjusted_avg * days_in_period
                if remaining_stock <= sales_possible:
                    # 库存在此时间段耗尽
                    days_needed = remaining_stock / adjusted_avg
                    return period_start + pd.Timedelta(days=days_needed)
                # 库存未耗尽，扣除销量，进入下一阶段
                remaining_stock -= sales_possible
                current_date = period["end"] + pd.Timedelta(days=1)

            # 阶段3：2026-01-01之后（系数=1.0，恢复基础日均）
            if remaining_stock > 0:
                days_needed = remaining_stock / base_avg
                return current_date + pd.Timedelta(days=days_needed)
            return current_date

        # 4. 预计FBA+AWD+在途用完时间（调用分阶段计算函数）
        df["预计FBA+AWD+在途用完时间"] = df.apply(
            lambda row: calculate_exhaust_date(row, "FBA+AWD+在途库存"), axis=1
        )

        # 5. 预计总库存用完时间（调用分阶段计算函数）
        df["预计总库存用完"] = df.apply(
            lambda row: calculate_exhaust_date(row, "全部总库存"), axis=1
        )

        # 6. 新增：分阶段计算滞销库存的核心函数（替换原固定日均计算）
        def calculate_overstock(row, stock_col):
            """
            按时间段系数计算目标日期前的滞销库存
            stock_col: 库存列名（"FBA+AWD+在途库存" 或 "全部总库存"）
            """
            record_date = row["记录时间"]
            stock = row[stock_col]
            base_avg = row["日均"] if row["日均"] > 0 else 0.1
            target_date = TARGET_DATE  # 目标消耗完成日期（需在函数外定义，如datetime(2025,12,1)）
            remaining_stock = stock
            current_date = record_date
            sold_by_target = 0  # 目标日期前可售出的库存

            # 若记录日期≥目标日期或库存为0，无滞销
            if current_date >= target_date or remaining_stock <= 0:
                return 0

            # 阶段1：记录日期 → 2025-10-14（系数=1.0）
            phase1_end = datetime(2025, 10, 14)
            if current_date <= phase1_end:
                actual_end = min(phase1_end, target_date)  # 不超过目标日期
                days_in_phase = (actual_end - current_date).days + 1
                sales = base_avg * days_in_phase
                sales = min(sales, remaining_stock)  # 最多售出剩余库存
                sold_by_target += sales
                remaining_stock -= sales
                current_date = actual_end + pd.Timedelta(days=1)
                # 若已达目标日期或库存耗尽，提前返回
                if current_date > target_date or remaining_stock <= 0:
                    return max(0, stock - sold_by_target)

            # 阶段2：处理4个特殊时间段
            for period in TIME_PERIODS:
                if current_date >= target_date or remaining_stock <= 0:
                    break
                period_start = max(current_date, period["start"])
                period_end = min(period["end"], target_date)
                if period_start > period_end:
                    continue
                days_in_period = (period_end - period_start).days + 1
                adjusted_avg = base_avg * period["coefficient"]
                sales = adjusted_avg * days_in_period
                sales = min(sales, remaining_stock)
                sold_by_target += sales
                remaining_stock -= sales
                current_date = period_end + pd.Timedelta(days=1)

            # 滞销库存 = 总库存 - 目标日期前可售出库存（取非负）
            return max(0, stock - sold_by_target)

        # 7. 预计用完时间比目标时间多出来的天数（基于分阶段计算的耗尽日期）
        days_diff = (df["预计总库存用完"] - TARGET_DATE).dt.days
        df["预计用完时间比目标时间多出来的天数"] = np.where(days_diff > 0, days_diff, 0).astype(int)

        # 8. 状态判断（保持原逻辑，但依赖新的耗尽日期）
        def determine_status(days):
            if days >= 20:
                return "高滞销风险"
            elif days >= 10:
                return "中滞销风险"
            elif days > 0:
                return "低滞销风险"
            else:  # days == 0
                return "健康"
        df["状态判断"] = df["预计用完时间比目标时间多出来的天数"].apply(determine_status)

        # 9. 环比上周库存滞销情况变化（保持原逻辑）
        df = df.sort_values(["MSKU", "记录时间"])
        df["上周状态"] = df.groupby("MSKU")["状态判断"].shift(1)
        status_severity = {"健康": 0, "低滞销风险": 1, "中滞销风险": 2, "高滞销风险": 3}
        def compare_status(current, previous):
            if pd.isna(previous):
                return "-"
            if current == previous:
                return "维持不变"
            current_sev = status_severity.get(current, 0)
            prev_sev = status_severity.get(previous, 0)
            return "改善" if current_sev < prev_sev else "恶化"
        df["环比上周库存滞销情况变化"] = df.apply(
            lambda row: compare_status(row["状态判断"], row["上周状态"]), axis=1
        )

        # 10. FBA+AWD+在途滞销数量（调用分阶段滞销计算函数）
        df["FBA+AWD+在途滞销数量"] = df.apply(
            lambda row: calculate_overstock(row, "FBA+AWD+在途库存"), axis=1
        ).round().astype(int)

        # 11. 总滞销库存（调用分阶段滞销计算函数）
        df["总滞销库存"] = df.apply(
            lambda row: calculate_overstock(row, "全部总库存"), axis=1
        ).round().astype(int)

        # 12. 本地滞销数量（保持原逻辑）
        df["本地滞销数量"] = (df["总滞销库存"] - df["FBA+AWD+在途滞销数量"]).round().astype(int)
        df["本地滞销数量"] = np.maximum(df["本地滞销数量"], 0)

        # 13. 预计总库存需要消耗天数（基于分阶段耗尽日期计算）
        df["预计总库存需要消耗天数"] = (
            (df["预计总库存用完"] - df["记录时间"]).dt.total_seconds() / (24 * 3600)
        ).round().astype(int)

        # 14. 清库存的目标日均（保持原逻辑，但依赖新的总库存和天数）
        days_available = (TARGET_DATE - df["记录时间"]).dt.days
        days_available = np.maximum(days_available, 1)
        df["清库存的目标日均"] = np.where(
            df["状态判断"] == "健康",
            df["日均"],  # 健康状态用基础日均
            df["全部总库存"] / days_available  # 其他状态按目标日期计算
        ).round(2)

        # 排序（保持原逻辑）
        df = df.sort_values("记录时间", ascending=False).reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"数据加载失败：{str(e)}")
        return None


def get_week_data(df, target_date):
    """获取指定日期的数据"""
    target_date = pd.to_datetime(target_date).normalize()
    week_data = df[df["记录时间"] == target_date].copy()
    return week_data if not week_data.empty else None


def get_previous_week_data(df, current_date):
    """获取上一周数据（用于环比计算）"""
    current_date = pd.to_datetime(current_date).normalize()
    all_dates = sorted(df["记录时间"].unique())

    if current_date not in all_dates:
        return None
    current_idx = all_dates.index(current_date)

    if current_idx > 0:
        prev_date = all_dates[current_idx - 1]
        return get_week_data(df, prev_date)
    return None


def calculate_status_metrics(data):
    """计算状态分布指标"""
    if data is None or data.empty:
        return {"总MSKU数": 0, "健康": 0, "低滞销风险": 0, "中滞销风险": 0, "高滞销风险": 0}

    total = len(data)
    status_counts = data["状态判断"].value_counts().to_dict()

    metrics = {"总MSKU数": total}
    for status in ["健康", "低滞销风险", "中滞销风险", "高滞销风险"]:
        metrics[status] = status_counts.get(status, 0)

    return metrics


def compare_with_previous(current_metrics, prev_metrics):
    """计算环比变化"""
    comparison = {}
    for key in current_metrics:
        curr_val = current_metrics[key]
        prev_val = prev_metrics.get(key, 0) if prev_metrics else 0

        diff = curr_val - prev_val
        pct = (diff / prev_val) * 100 if prev_val != 0 else 0

        # 确定颜色
        if key == "总MSKU数":
            color = "#000000"
        elif key in ["健康"]:
            color = "#2E8B57" if diff >= 0 else "#DC143C"
        else:
            color = "#2E8B57" if diff <= 0 else "#DC143C"

        comparison[key] = {
            "当前值": curr_val,
            "变化值": diff,
            "变化率(%)": round(pct, 1),
            "颜色": color
        }
    return comparison


# ------------------------------
# 2. 可视化组件函数
# ------------------------------
def render_metric_card(title, current, diff=None, pct=None, color="#000000"):
    """渲染带环比的指标卡片"""
    if diff is None:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value" style="color:{color}">{current}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        diff_symbol = "+" if diff > 0 else ""
        pct_symbol = "+" if pct > 0 else ""
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value" style="color:{color}">{current}</div>
            <div class="metric-change" style="color:{color}">
                {diff_symbol}{diff} ({pct_symbol}{pct}%)
            </div>
        </div>
        """, unsafe_allow_html=True)


# 首先，添加一个通用的多级索引表格渲染函数
def render_multi_index_table(data, index_columns, value_columns, page=1, page_size=30, table_id=""):
    """
    渲染支持交互式多级索引的表格

    参数:
    - data: 要显示的数据
    - index_columns: 作为索引的列名列表（多级索引）
    - value_columns: 作为值的列名列表
    - page: 当前页码
    - page_size: 每页显示的记录数
    - table_id: 表格唯一标识，用于确保Streamlit组件key的唯一性
    """
    if data.empty:
        st.info("没有数据可显示")
        return 0

    total_rows = len(data)
    total_pages = max(1, (total_rows + page_size - 1) // page_size)

    # 创建多级索引
    multi_index_data = data.set_index(index_columns)

    # 分页处理
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    paginated_data = multi_index_data.iloc[start_idx:end_idx]

    # 转换为HTML显示，保留多级索引结构
    html = paginated_data.to_html(
        classes=["dataframe", "table", "table-striped", "table-hover"],
        escape=False,
        na_rep="",
        border=0
    )

    # 添加自定义CSS美化多级索引表格
    st.markdown("""
    <style>
    .dataframe th {
        background-color: #f8f9fa;
        text-align: left;
        padding: 8px 12px;
        border-bottom: 2px solid #ddd;
    }
    .dataframe td {
        padding: 8px 12px;
        border-bottom: 1px solid #ddd;
    }
    .dataframe tr:hover {
        background-color: #f1f1f1;
    }
    .dataframe .level0 {
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(html, unsafe_allow_html=True)

    # 分页控制
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if page > 1:
            if st.button("上一页", key=f"prev_page_{table_id}"):
                st.session_state[f"current_page_{table_id}"] = page - 1
                st.rerun()
    with col2:
        st.write(f"第 {page} 页，共 {total_pages} 页，共 {total_rows} 条记录")
    with col3:
        if page < total_pages:
            if st.button("下一页", key=f"next_page_{table_id}"):
                st.session_state[f"current_page_{table_id}"] = page + 1
                st.rerun()

    return total_rows


def render_status_distribution_chart(metrics, title):
    """渲染状态分布柱状图"""
    status_data = pd.DataFrame({
        "状态": ["健康", "低滞销风险", "中滞销风险", "高滞销风险"],
        "MSKU数": [metrics[status] for status in ["健康", "低滞销风险", "中滞销风险", "高滞销风险"]]
    })

    fig = px.bar(
        status_data,
        x="状态",
        y="MSKU数",
        color="状态",
        color_discrete_map=STATUS_COLORS,
        title=title,
        text="MSKU数",
        height=400,
        # 添加自定义数据，用于点击时识别筛选条件
        custom_data = ["状态"]  # 传递“状态”字段作为筛选标识
    )

    fig.update_traces(
        textposition="outside",
        textfont=dict(size=12, weight="bold"),
        marker=dict(line=dict(color="#ffffff", width=1))
    )
    fig.update_layout(
        xaxis_title="风险状态",
        yaxis_title="MSKU数量",
        showlegend=False,
        plot_bgcolor="#f8f9fa",
        margin=dict(t=50, b=20, l=20, r=20)
    )
    return fig


def render_days_distribution_chart(data, title):
    """渲染库存可用天数分布图表（使用预计总库存需要消耗天数）"""
    if data is None or data.empty:
        fig = go.Figure()
        fig.add_annotation(text="无数据可展示", x=0.5, y=0.5, showarrow=False, font=dict(size=16))
        fig.update_layout(title=title, plot_bgcolor="#f8f9fa", height=400)
        return fig

    # 使用预计总库存需要消耗天数作为横坐标
    valid_days = data["预计总库存需要消耗天数"].clip(lower=0)
    today = data["记录时间"].iloc[0]
    days_to_target = (TARGET_DATE - today).days

    # 状态阈值
    thresholds = {
        "高滞销风险": days_to_target,
        "中滞销风险": days_to_target - 14,
        "低滞销风险": days_to_target - 30
    }

    fig = px.histogram(
        valid_days,
        nbins=30,
        title=title,
        labels={"value": "预计总库存需要消耗天数", "count": "MSKU数量"},
        color_discrete_sequence=["#87CEEB"],
        height=400
    )

    # 添加阈值线（虚线）
    for status, threshold in thresholds.items():
        if threshold >= 0:  # 只显示合理的阈值线
            fig.add_vline(
                x=threshold,
                line_dash="dash",
                line_color=STATUS_COLORS[status],
                annotation_text=f"{status}阈值",
                annotation_position="top right",
                annotation_font=dict(color=STATUS_COLORS[status])
            )

    fig.update_layout(
        plot_bgcolor="#f8f9fa",
        margin=dict(t=50, b=20, l=20, r=20),
        xaxis_title="预计总库存需要消耗天数",
        yaxis_title="MSKU数量"
    )
    return fig


def render_store_status_table(current_data, prev_data):
    """渲染店铺状态分布表（带环比）"""
    if current_data is None or current_data.empty:
        st.markdown("<p>无店铺数据可展示</p>", unsafe_allow_html=True)
        return

    # 生成当前店铺状态分布
    current_pivot = pd.pivot_table(
        current_data,
        index="店铺",
        columns="状态判断",
        values="MSKU",
        aggfunc="count",
        fill_value=0
    ).reindex(columns=["健康", "低滞销风险", "中滞销风险", "高滞销风险"], fill_value=0)

    # 生成上周店铺状态分布
    prev_pivot = pd.pivot_table(
        prev_data,
        index="店铺",
        columns="状态判断",
        values="MSKU",
        aggfunc="count",
        fill_value=0
    ).reindex(columns=["健康", "低滞销风险", "中滞销风险", "高滞销风险"],
              fill_value=0) if prev_data is not None else None

    # 合并并计算环比
    html = "<table style='width:100%; border-collapse:collapse;'>"
    html += "<tr><th style='border:1px solid #ddd; padding:8px;'>店铺</th>"
    for status in ["健康", "低滞销风险", "中滞销风险", "高滞销风险"]:
        html += f"<th style='border:1px solid #ddd; padding:8px; background-color:{STATUS_COLORS[status]}20;'>{status}</th>"
    html += "</tr>"

    for store in current_pivot.index:
        html += f"<tr><td style='border:1px solid #ddd; padding:8px; font-weight:bold;'>{store}</td>"
        for status in ["健康", "低滞销风险", "中滞销风险", "高滞销风险"]:
            curr = current_pivot.loc[store, status]
            prev = prev_pivot.loc[store, status] if (prev_pivot is not None and store in prev_pivot.index) else 0
            diff = curr - prev

            # 确定颜色
            if status == "健康":
                color = "#2E8B57" if diff >= 0 else "#DC143C"
            else:
                color = "#2E8B57" if diff <= 0 else "#DC143C"

            diff_symbol = "+" if diff > 0 else ""
            html += f"<td style='border:1px solid #ddd; padding:8px;'>{curr}<br><span style='color:{color}; font-size:12px;'>{diff_symbol}{diff}</span></td>"
        html += "</tr>"
    html += "</table>"

    st.markdown(html, unsafe_allow_html=True)


def render_product_detail_table(data, prev_data=None, page=1, page_size=30, table_id=""):
    """渲染产品风险详情表（带环比和分页功能）"""
    if data is None or data.empty:
        st.markdown("<p style='color:#666'>无匹配产品数据</p>", unsafe_allow_html=True)
        return 0
    # 1. 添加自定义排序逻辑
    # 定义风险状态的排序优先级
    status_order = {
        "高滞销风险": 0,
        "中滞销风险": 1,
        "低滞销风险": 2,
        "健康": 3
    }
    # 添加排序辅助列
    data = data.copy()
    data["_sort_key"] = data["状态判断"].map(status_order)
    # 先按风险状态优先级升序（高风险在前），再按总滞销库存降序
    data = data.sort_values(by=["_sort_key", "总滞销库存"], ascending=[True, False])
    # 删除临时排序列
    data = data.drop(columns=["_sort_key"])
    # 定义要显示的列（新增：4个时间段的系数+调整后日均）
    display_cols = [
        "MSKU", "品名", "店铺",
        # 基础日均列（保留原逻辑）
        "日均", "7天日均", "14天日均", "28天日均",
        # 新增：四个时间段的系数+调整后日均（按时间顺序插入）
        "10月15-31日系数", "10月15-31日调整后日均",
        "11月1-15日系数", "11月1-15日调整后日均",
        "11月16-30日系数", "11月16-30日调整后日均",
        "12月1-31日系数", "12月1-31日调整后日均",
        # 原有其他列（保持不变）
        "FBA+AWD+在途库存", "本地可用", "全部总库存",
        "预计FBA+AWD+在途用完时间", "预计总库存用完",
        "状态判断", "清库存的目标日均",
        "FBA+AWD+在途滞销数量", "本地滞销数量", "总滞销库存",
        "预计总库存需要消耗天数", "预计用完时间比目标时间多出来的天数",
        "环比上周库存滞销情况变化"
    ]

    # 确保所有列都存在（自动过滤数据中没有的列）
    available_cols = [col for col in display_cols if col in data.columns]
    table_data = data[available_cols].copy()
    total_rows = len(table_data)

    # 计算分页
    total_pages = ceil(total_rows / page_size)
    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, total_rows)
    paginated_data = table_data.iloc[start_idx:end_idx].copy()

    # 格式化日期（保持原逻辑）
    date_cols = ["预计FBA+AWD+在途用完时间", "预计总库存用完"]
    for col in date_cols:
        if col in paginated_data.columns:
            paginated_data[col] = pd.to_datetime(paginated_data[col]).dt.strftime("%Y-%m-%d")

    # 添加状态颜色（保持原逻辑）
    if "状态判断" in paginated_data.columns:
        paginated_data["状态判断"] = paginated_data["状态判断"].apply(
            lambda x: f"<span style='color:{STATUS_COLORS[x]}; font-weight:bold;'>{x}</span>"
        )

    # 添加环比（核心修改：新增调整后日均的环比对比）
    if prev_data is not None and not prev_data.empty:
        # 新增：将调整后日均列加入prev_map（用于环比计算）
        prev_map = prev_data.set_index("MSKU")[
            ["日均", "7天日均", "14天日均", "28天日均",
             "FBA+AWD+在途库存","本地可用",
             "全部总库存", "FBA+AWD+在途滞销数量", "本地滞销数量", "总滞销库存",
             "预计总库存需要消耗天数", "预计用完时间比目标时间多出来的天数",
             # 新增：四个时间段的调整后日均（系数固定，无需环比）
             "10月15-31日调整后日均", "11月1-15日调整后日均",
             "11月16-30日调整后日均", "12月1-31日调整后日均"]
        ].to_dict("index")

        def add_compare(row, col):
            msku = row["MSKU"]
            curr_val = row[col]
            prev_val = prev_map.get(msku, {}).get(col, 0)

            if prev_val == 0:
                return f"{curr_val:.2f}<br><span style='color:#666'>无数据</span>"

            diff = curr_val - prev_val
            pct = (diff / prev_val) * 100
            # 确定颜色：日均上升好，滞销数量下降好
            if col in ["日均", "7天日均", "14天日均", "28天日均",
                       # 新增：调整后日均也属于"日均类"，上升为好
                       "10月15-31日调整后日均", "11月1-15日调整后日均",
                       "11月16-30日调整后日均", "12月1-31日调整后日均"]:
                color = "#2E8B57" if diff >= 0 else "#DC143C"
            else:  # 库存和滞销数量相关列（保持原逻辑）
                color = "#2E8B57" if diff <= 0 else "#DC143C"

            diff_symbol = "+" if diff > 0 else ""
            pct_symbol = "+" if pct > 0 else ""
            return f"{curr_val:.2f}<br><span style='color:{color}'>{diff_symbol}{diff:.2f} ({pct_symbol}{pct:.1f}%)</span>"

        # 需要比较的数值列（新增：调整后日均列）
        numeric_cols = [
            "日均", "7天日均", "14天日均", "28天日均",
            # 新增：四个时间段的调整后日均（系数固定，无需环比）
            "10月15-31日调整后日均", "11月1-15日调整后日均",
            "11月16-30日调整后日均", "12月1-31日调整后日均",
            # 原有其他列（保持不变）
            "FBA+AWD+在途库存","本地可用",
            "全部总库存", "FBA+AWD+在途滞销数量", "本地滞销数量", "总滞销库存",
            "预计总库存需要消耗天数", "预计用完时间比目标时间多出来的天数"
        ]
        for col in numeric_cols:
            if col in paginated_data.columns:
                paginated_data[col] = paginated_data.apply(lambda x: add_compare(x, col), axis=1)

    # 显示表格（保持原逻辑）
    st.markdown(paginated_data.to_html(escape=False, index=False), unsafe_allow_html=True)

    # 显示分页按钮（保持原逻辑）
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if page > 1:
            # 添加table_id参数使key唯一
            if st.button("上一页", key=f"prev_page_{table_id}"):
                st.session_state.current_page = page - 1
                st.rerun()
    with col2:
        st.write(f"第 {page} 页，共 {total_pages} 页，共 {total_rows} 条记录")
    with col3:
        if page < total_pages:
            # 添加table_id参数使key唯一
            if st.button("下一页", key=f"next_page_{table_id}"):
                st.session_state.current_page = page + 1
                st.rerun()

    return total_rows


def render_four_week_comparison_table(df, date_list):
    """渲染近四周概览表（带环比变化值）"""
    if len(date_list) < 1:
        st.markdown("<p>无数据可展示</p>", unsafe_allow_html=True)
        return

    # 确保我们有最多四周的数据（核心修改点：将3改为4）
    display_dates = date_list[-4:] if len(date_list) >= 4 else date_list
    date_labels = [d.strftime("%Y-%m-%d") for d in display_dates]

    # 创建比较表数据
    comparison_data = []

    for i, date in enumerate(display_dates):
        data = get_week_data(df, date)
        metrics = calculate_status_metrics(data)

        # 计算环比
        if i > 0:
            prev_data = get_week_data(df, display_dates[i - 1])
            prev_metrics = calculate_status_metrics(prev_data)
            comparisons = compare_with_previous(metrics, prev_metrics)
        else:
            comparisons = None

        # 添加行数据
        row = {
            "日期": date_labels[i],
            "总MSKU数": metrics["总MSKU数"],
            "健康": metrics["健康"],
            "低滞销风险": metrics["低滞销风险"],
            "中滞销风险": metrics["中滞销风险"],
            "高滞销风险": metrics["高滞销风险"]
        }

        # 添加环比变化
        if comparisons:
            row["总MSKU数变化"] = comparisons["总MSKU数"]["变化值"]
            row["健康变化"] = comparisons["健康"]["变化值"]
            row["低滞销风险变化"] = comparisons["低滞销风险"]["变化值"]
            row["中滞销风险变化"] = comparisons["中滞销风险"]["变化值"]
            row["高滞销风险变化"] = comparisons["高滞销风险"]["变化值"]

            row["总MSKU数变化率"] = comparisons["总MSKU数"]["变化率(%)"]
            row["健康变化率"] = comparisons["健康"]["变化率(%)"]
            row["低滞销风险变化率"] = comparisons["低滞销风险"]["变化率(%)"]
            row["中滞销风险变化率"] = comparisons["中滞销风险"]["变化率(%)"]
            row["高滞销风险变化率"] = comparisons["高滞销风险"]["变化率(%)"]

        comparison_data.append(row)

    # 创建HTML表格
    html = "<table style='width:100%; border-collapse:collapse;'>"
    html += "<tr><th style='border:1px solid #ddd; padding:8px;'>日期</th>"
    html += "<th style='border:1px solid #ddd; padding:8px;'>总MSKU数</th>"
    html += "<th style='border:1px solid #ddd; padding:8px;'>健康</th>"
    html += "<th style='border:1px solid #ddd; padding:8px;'>低滞销风险</th>"
    html += "<th style='border:1px solid #ddd; padding:8px;'>中滞销风险</th>"
    html += "<th style='border:1px solid #ddd; padding:8px;'>高滞销风险</th></tr>"

    for row in comparison_data:
        html += f"<tr><td style='border:1px solid #ddd; padding:8px; font-weight:bold;'>{row['日期']}</td>"

        # 总MSKU数
        if "总MSKU数变化" in row:
            diff = row["总MSKU数变化"]
            color = "#2E8B57" if diff >= 0 else "#DC143C"
            symbol = "+" if diff > 0 else ""
            html += f"<td style='border:1px solid #ddd; padding:8px;'>{row['总MSKU数']}<br><span style='color:{color}; font-size:12px;'>{symbol}{diff}</span></td>"
        else:
            html += f"<td style='border:1px solid #ddd; padding:8px;'>{row['总MSKU数']}</td>"

        # 健康
        if "健康变化" in row:
            diff = row["健康变化"]
            color = "#2E8B57" if diff >= 0 else "#DC143C"
            symbol = "+" if diff > 0 else ""
            html += f"<td style='border:1px solid #ddd; padding:8px; color:{STATUS_COLORS['健康']};'>{row['健康']}<br><span style='color:{color}; font-size:12px;'>{symbol}{diff}</span></td>"
        else:
            html += f"<td style='border:1px solid #ddd; padding:8px; color:{STATUS_COLORS['健康']};'>{row['健康']}</td>"

        # 低滞销风险
        if "低滞销风险变化" in row:
            diff = row["低滞销风险变化"]
            color = "#2E8B57" if diff <= 0 else "#DC143C"
            symbol = "+" if diff > 0 else ""
            html += f"<td style='border:1px solid #ddd; padding:8px; color:{STATUS_COLORS['低滞销风险']};'>{row['低滞销风险']}<br><span style='color:{color}; font-size:12px;'>{symbol}{diff}</span></td>"
        else:
            html += f"<td style='border:1px solid #ddd; padding:8px; color:{STATUS_COLORS['低滞销风险']};'>{row['低滞销风险']}</td>"

        # 中滞销风险
        if "中滞销风险变化" in row:
            diff = row["中滞销风险变化"]
            color = "#2E8B57" if diff <= 0 else "#DC143C"
            symbol = "+" if diff > 0 else ""
            html += f"<td style='border:1px solid #ddd; padding:8px; color:{STATUS_COLORS['中滞销风险']};'>{row['中滞销风险']}<br><span style='color:{color}; font-size:12px;'>{symbol}{diff}</span></td>"
        else:
            html += f"<td style='border:1px solid #ddd; padding:8px; color:{STATUS_COLORS['中滞销风险']};'>{row['中滞销风险']}</td>"

        # 高滞销风险
        if "高滞销风险变化" in row:
            diff = row["高滞销风险变化"]
            color = "#2E8B57" if diff <= 0 else "#DC143C"
            symbol = "+" if diff > 0 else ""
            html += f"<td style='border:1px solid #ddd; padding:8px; color:{STATUS_COLORS['高滞销风险']};'>{row['高滞销风险']}<br><span style='color:{color}; font-size:12px;'>{symbol}{diff}</span></td>"
        else:
            html += f"<td style='border:1px solid #ddd; padding:8px; color:{STATUS_COLORS['高滞销风险']};'>{row['高滞销风险']}</td>"

        html += "</tr>"

    html += "</table>"
    st.markdown(html, unsafe_allow_html=True)


def render_four_week_status_chart(df, date_list):
    """四周状态变化趋势（柱状图版本）"""
    if len(date_list) < 1:
        fig = go.Figure()
        fig.add_annotation(text="无数据可展示", x=0.5, y=0.5, showarrow=False, font=dict(size=16))
        fig.update_layout(title="四周状态变化趋势", plot_bgcolor="#f8f9fa", height=400)
        return fig

    # 获取最多四周数据（核心修改点：将3改为4）
    display_dates = date_list[-4:] if len(date_list) >= 4 else date_list
    date_labels = [d.strftime("%Y-%m-%d") for d in display_dates]

    # 准备数据
    trend_data = []
    for date, label in zip(display_dates, date_labels):
        data = get_week_data(df, date)
        metrics = calculate_status_metrics(data)

        for status in ["健康", "低滞销风险", "中滞销风险", "高滞销风险"]:
            trend_data.append({
                "日期": label,
                "状态": status,
                "MSKU数": metrics[status]
            })

    trend_df = pd.DataFrame(trend_data)

    # 创建柱状图
    fig = px.bar(
        trend_df,
        x="状态",
        y="MSKU数",
        color="日期",
        barmode="group",
        title="四周状态变化趋势",  # 修改标题
        text="MSKU数",
        height=400
    )

    fig.update_traces(
        textposition="outside",
        textfont=dict(size=12)
    )

    fig.update_layout(
        xaxis_title="风险状态",
        yaxis_title="MSKU数量",
        plot_bgcolor="#f8f9fa",
        margin=dict(t=50, b=20, l=20, r=20)
    )

    return fig



def render_store_trend_charts(df, date_list):
    """渲染每个店铺的状态趋势折线图（分两列显示）"""
    if len(date_list) < 1:
        st.markdown("<p>无数据可展示</p>", unsafe_allow_html=True)
        return

    # 获取所有店铺
    all_data = pd.concat([get_week_data(df, date) for date in date_list])
    if all_data is None or all_data.empty:
        st.markdown("<p>无店铺数据可展示</p>", unsafe_allow_html=True)
        return

    stores = sorted(all_data["店铺"].unique())
    date_labels = [d.strftime("%Y-%m-%d") for d in date_list]

    # 分两列显示图表
    cols = st.columns(2)
    for i, store in enumerate(stores):
        # 准备店铺数据
        store_data = []
        for date, label in zip(date_list, date_labels):
            data = get_week_data(df, date)
            if data is not None and not data.empty:
                store_status_data = data[data["店铺"] == store]
                metrics = calculate_status_metrics(store_status_data)

                for status in ["健康", "低滞销风险", "中滞销风险", "高滞销风险"]:
                    store_data.append({
                        "日期": label,
                        "状态": status,
                        "MSKU数": metrics[status]
                    })

        if not store_data:
            continue

        store_df = pd.DataFrame(store_data)

        # 创建折线图
        fig = go.Figure()
        for status in ["健康", "低滞销风险", "中滞销风险", "高滞销风险"]:
            status_data = store_df[store_df["状态"] == status]
            fig.add_trace(go.Scatter(
                x=status_data["日期"],
                y=status_data["MSKU数"],
                mode="lines+markers",
                name=status,
                line=dict(color=STATUS_COLORS[status], width=2),
                marker=dict(size=8)
            ))

        fig.update_layout(
            title=f"{store} 状态变化趋势",
            xaxis_title="日期",
            yaxis_title="MSKU数量",
            plot_bgcolor="#f8f9fa",
            height=300,
            margin=dict(t=50, b=20, l=20, r=20)
        )

        # 在对应列显示图表
        with cols[i % 2]:
            st.plotly_chart(fig, use_container_width=True)


def render_store_weekly_changes(df, date_list):
    """渲染店铺每周变化情况表"""
    if len(date_list) < 1:
        st.markdown("<p>无数据可展示</p>", unsafe_allow_html=True)
        return

    # 获取所有店铺
    all_data = pd.concat([get_week_data(df, date) for date in date_list])
    if all_data is None or all_data.empty:
        st.markdown("<p>无店铺数据可展示</p>", unsafe_allow_html=True)
        return

    stores = sorted(all_data["店铺"].unique())
    date_labels = [d.strftime("%Y-%m-%d") for d in date_list]

    # 创建HTML表格
    html = "<table style='width:100%; border-collapse:collapse;'>"
    html += "<tr><th style='border:1px solid #ddd; padding:8px;'>店铺</th>"
    html += "<th style='border:1px solid #ddd; padding:8px;'>日期</th>"
    html += "<th style='border:1px solid #ddd; padding:8px;'>总MSKU数</th>"
    html += "<th style='border:1px solid #ddd; padding:8px; background-color:#2E8B5720;'>健康</th>"
    html += "<th style='border:1px solid #ddd; padding:8px; background-color:#4169E120;'>低滞销风险</th>"
    html += "<th style='border:1px solid #ddd; padding:8px; background-color:#FFD70020;'>中滞销风险</th>"
    html += "<th style='border:1px solid #ddd; padding:8px; background-color:#DC143C20;'>高滞销风险</th></tr>"

    for store in stores:
        for i, (date, label) in enumerate(zip(date_list, date_labels)):
            data = get_week_data(df, date)
            if data is not None and not data.empty:
                store_status_data = data[data["店铺"] == store]
                metrics = calculate_status_metrics(store_status_data)

                # 获取上周数据
                prev_metrics = None
                if i > 0:
                    prev_data = get_week_data(df, date_list[i - 1])
                    if prev_data is not None and not prev_data.empty:
                        prev_store_data = prev_data[prev_data["店铺"] == store]
                        prev_metrics = calculate_status_metrics(prev_store_data)

                # 开始行
                html += f"<tr><td style='border:1px solid #ddd; padding:8px; font-weight:bold;'>{store}</td>"
                html += f"<td style='border:1px solid #ddd; padding:8px;'>{label}</td>"

                # 总MSKU数
                if prev_metrics:
                    diff = metrics["总MSKU数"] - prev_metrics["总MSKU数"]
                    color = "#2E8B57" if diff >= 0 else "#DC143C"
                    symbol = "+" if diff > 0 else ""
                    html += f"<td style='border:1px solid #ddd; padding:8px;'>{metrics['总MSKU数']}<br><span style='color:{color}; font-size:12px;'>{symbol}{diff}</span></td>"
                else:
                    html += f"<td style='border:1px solid #ddd; padding:8px;'>{metrics['总MSKU数']}</td>"

                # 健康
                if prev_metrics:
                    diff = metrics["健康"] - prev_metrics["健康"]
                    color = "#2E8B57" if diff >= 0 else "#DC143C"
                    symbol = "+" if diff > 0 else ""
                    html += f"<td style='border:1px solid #ddd; padding:8px;'>{metrics['健康']}<br><span style='color:{color}; font-size:12px;'>{symbol}{diff}</span></td>"
                else:
                    html += f"<td style='border:1px solid #ddd; padding:8px;'>{metrics['健康']}</td>"

                # 低滞销风险
                if prev_metrics:
                    diff = metrics["低滞销风险"] - prev_metrics["低滞销风险"]
                    color = "#2E8B57" if diff <= 0 else "#DC143C"
                    symbol = "+" if diff > 0 else ""
                    html += f"<td style='border:1px solid #ddd; padding:8px;'>{metrics['低滞销风险']}<br><span style='color:{color}; font-size:12px;'>{symbol}{diff}</span></td>"
                else:
                    html += f"<td style='border:1px solid #ddd; padding:8px;'>{metrics['低滞销风险']}</td>"

                # 中滞销风险
                if prev_metrics:
                    diff = metrics["中滞销风险"] - prev_metrics["中滞销风险"]
                    color = "#2E8B57" if diff <= 0 else "#DC143C"
                    symbol = "+" if diff > 0 else ""
                    html += f"<td style='border:1px solid #ddd; padding:8px;'>{metrics['中滞销风险']}<br><span style='color:{color}; font-size:12px;'>{symbol}{diff}</span></td>"
                else:
                    html += f"<td style='border:1px solid #ddd; padding:8px;'>{metrics['中滞销风险']}</td>"

                # 高滞销风险
                if prev_metrics:
                    diff = metrics["高滞销风险"] - prev_metrics["高滞销风险"]
                    color = "#2E8B57" if diff <= 0 else "#DC143C"
                    symbol = "+" if diff > 0 else ""
                    html += f"<td style='border:1px solid #ddd; padding:8px;'>{metrics['高滞销风险']}<br><span style='color:{color}; font-size:12px;'>{symbol}{diff}</span></td>"
                else:
                    html += f"<td style='border:1px solid #ddd; padding:8px;'>{metrics['高滞销风险']}</td>"

                html += "</tr>"

    html += "</table>"
    st.markdown(html, unsafe_allow_html=True)


def render_status_change_table(data, page=1, page_size=30):
    """渲染环比上周库存滞销情况变化表（带多级索引和分页）"""
    if data is None or data.empty:
        st.markdown("<p style='color:#666'>无数据可展示</p>", unsafe_allow_html=True)
        return 0

    # 核心修改：在display_cols中新增8个列（4个系数+4个调整后日均）
    display_cols = [
        "MSKU", "品名", "店铺", "记录时间",
        # 基础日均列（保留原逻辑）
        "日均", "7天日均", "14天日均", "28天日均",
        # 新增：四个时间段的系数+调整后日均（按时间顺序插入）
        "10月15-31日系数", "10月15-31日调整后日均",
        "11月1-15日系数", "11月1-15日调整后日均",
        "11月16-30日系数", "11月16-30日调整后日均",
        "12月1-31日系数", "12月1-31日调整后日均",
        # 原有其他列（保持不变）
        "FBA+AWD+在途库存","本地可用", "全部总库存", "预计FBA+AWD+在途用完时间", "预计总库存用完",
        "状态判断", "清库存的目标日均", "FBA+AWD+在途滞销数量", "本地滞销数量", "总滞销库存",
        "预计总库存需要消耗天数", "预计用完时间比目标时间多出来的天数", "环比上周库存滞销情况变化"
    ]

    # 确保所有列都存在（自动过滤数据中没有的列，新增列会被自动纳入）
    available_cols = [col for col in display_cols if col in data.columns]
    table_data = data[available_cols].copy()
    total_rows = len(table_data)

    # 计算分页（保持原逻辑，新增列不影响分页）
    total_pages = ceil(total_rows / page_size)
    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, total_rows)
    paginated_data = table_data.iloc[start_idx:end_idx].copy()

    # 格式化日期（保持原逻辑，不影响新增列）
    date_cols = ["记录时间", "预计FBA+AWD+在途用完时间", "预计总库存用完"]
    for col in date_cols:
        if col in paginated_data.columns:
            paginated_data[col] = pd.to_datetime(paginated_data[col]).dt.strftime("%Y-%m-%d")

    # 添加状态颜色（保持原逻辑）
    if "状态判断" in paginated_data.columns:
        paginated_data["状态判断"] = paginated_data["状态判断"].apply(
            lambda x: f"<span style='color:{STATUS_COLORS[x]}; font-weight:bold;'>{x}</span>"
        )

    # 添加环比上周库存滞销情况变化颜色（保持原逻辑）
    if "环比上周库存滞销情况变化" in paginated_data.columns:
        def color_status_change(x):
            if x == "改善":
                return f"<span style='color:#2E8B57; font-weight:bold;'>{x}</span>"
            elif x == "恶化":
                return f"<span style='color:#DC143C; font-weight:bold;'>{x}</span>"
            else:  # 维持不变
                return f"<span style='color:#000000; font-weight:bold;'>{x}</span>"

        paginated_data["环比上周库存滞销情况变化"] = paginated_data["环比上周库存滞销情况变化"].apply(color_status_change)

    # 显示表格（保持原逻辑，新增列会自动渲染）
    st.markdown(paginated_data.to_html(escape=False, index=False), unsafe_allow_html=True)

    # 显示分页按钮（保持原逻辑）
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if page > 1:
            # 使用不同的key值
            if st.button("上一页", key="prev_page_status"):
                st.session_state.current_status_page = page - 1
                st.rerun()
    with col2:
        st.write(f"第 {page} 页，共 {total_pages} 页，共 {total_rows} 条记录")
    with col3:
        if page < total_pages:
            # 使用不同的key值
            if st.button("下一页", key="next_page_status"):
                st.session_state.current_status_page = page + 1
                st.rerun()

    return total_rows


# 在现有代码的可视化组件区域添加以下代码




def render_risk_summary_table(summary_df):
    """在Streamlit中渲染风险汇总表格"""
    st.subheader("库存风险状态汇总表")

    # 自定义表格样式
    st.markdown("""
    <style>
    .summary-table {
        width: 100%;
        border-collapse: collapse;
        margin: 20px 0;
    }
    .summary-table th, .summary-table td {
        padding: 12px 15px;
        text-align: left;
        border-bottom: 1px solid #ddd;
    }
    .summary-table th {
        background-color: #f8f9fa;
        font-weight: bold;
    }
    .summary-table tr:hover {
        background-color: #f5f5f5;
    }
    .positive-change {
        color: #28a745;  /* 绿色：健康状态增加/风险状态减少 */
    }
    .negative-change {
        color: #dc3545;  /* 红色：健康状态减少/风险状态增加 */
    }
    </style>
    """, unsafe_allow_html=True)

    # 渲染表格
    html = "<table class='summary-table'>"
    # 表头
    html += "<tr>"
    for col in summary_df.columns:
        html += f"<th>{col}</th>"
    html += "</tr>"

    # 表内容
    for _, row in summary_df.iterrows():
        html += "<tr>"
        for col, value in row.items():
            if col == "状态判断":
                # 为状态添加颜色（复用全局STATUS_COLORS，合并状态用默认色）
                color = STATUS_COLORS.get(value, "#000000")  # 非基础状态用黑色
                html += f"<td style='color:{color}; font-weight:bold;'>{value}</td>"
            elif "环比变化" in col:
                # 标记正负变化（区分状态类型）
                if '(' in str(value):
                    change_val = float(value.split()[0])  # 提取变化值
                    status = row["状态判断"]  # 获取当前行的状态

                    # 健康状态：增加为正；风险状态：减少为正
                    if status == "健康":
                        is_positive = change_val >= 0
                    else:
                        is_positive = change_val <= 0

                    if is_positive:
                        html += f"<td class='positive-change'>{value}</td>"
                    else:
                        html += f"<td class='negative-change'>{value}</td>"
                else:
                    html += f"<td>{value}</td>"
            else:
                html += f"<td>{value}</td>"
        html += "</tr>"
    html += "</table>"

    st.markdown(html, unsafe_allow_html=True)


def create_risk_summary_table(current_data, previous_data):
    statuses = [
        "健康",
        "低滞销风险",
        "中滞销风险",
        "高滞销风险",
        "低滞销风险+中滞销风险+高滞销风险",
        "中滞销风险+高滞销风险"
    ]
    status_mappings = {
        "健康": ["健康"],
        "低滞销风险": ["低滞销风险"],
        "中滞销风险": ["中滞销风险"],
        "高滞销风险": ["高滞销风险"],
        "低滞销风险+中滞销风险+高滞销风险": ["低滞销风险", "中滞销风险", "高滞销风险"],
        "中滞销风险+高滞销风险": ["中滞销风险", "高滞销风险"]
    }

    # 计算当前周期的总MSKU和总滞销库存（用于计算占比）
    total_current_msku = current_data['MSKU'].nunique() if current_data is not None and not current_data.empty else 0
    total_current_inventory = current_data[
        '总滞销库存'].sum() if current_data is not None and not current_data.empty else 0

    summary_data = []
    for status in statuses:
        original_statuses = status_mappings[status]

        # 筛选当前周期数据
        current_filtered = current_data[current_data['状态判断'].isin(original_statuses)] if (
                    current_data is not None and not current_data.empty) else pd.DataFrame()
        current_msku = current_filtered['MSKU'].nunique() if not current_filtered.empty else 0
        current_inventory = current_filtered['总滞销库存'].sum() if not current_filtered.empty else 0

        # 处理上一周期数据
        if previous_data is not None and not previous_data.empty:
            prev_filtered = previous_data[previous_data['状态判断'].isin(original_statuses)]
            prev_msku = prev_filtered['MSKU'].nunique() if not prev_filtered.empty else 0
            prev_inventory = prev_filtered['总滞销库存'].sum() if not prev_filtered.empty else 0
        else:
            prev_msku = 0
            prev_inventory = 0

        # 计算环比
        msku_change = current_msku - prev_msku
        msku_change_pct = (msku_change / prev_msku * 100) if prev_msku != 0 else 0
        inventory_change = current_inventory - prev_inventory
        inventory_change_pct = (inventory_change / prev_inventory * 100) if prev_inventory != 0 else 0

        # 新增：计算占比（当前状态值 / 总值）
        msku_ratio = (current_msku / total_current_msku * 100) if total_current_msku != 0 else 0
        inventory_ratio = (current_inventory / total_current_inventory * 100) if total_current_inventory != 0 else 0

        summary_data.append({
            "状态判断": status,
            "MSKU数": current_msku,
            "MSKU占比": f"{msku_ratio:.1f}%",  # 新增：MSKU占比（保留1位小数）
            "MSKU环比变化": f"{msku_change} ({msku_change_pct:.1f}%)",
            "总滞销库存数": round(current_inventory, 2),
            "总滞销库存占比": f"{inventory_ratio:.1f}%",  # 新增：库存占比（保留1位小数）
            "库存环比变化": f"{round(inventory_change, 2)} ({inventory_change_pct:.1f}%)"
        })

    return pd.DataFrame(summary_data)

def render_stock_forecast_chart(data, msku):
    """渲染单个MSKU的库存预测图表（从记录时间到2025年12月31日，含分阶段系数）"""
    if data is None or data.empty:
        fig = go.Figure()
        fig.add_annotation(text="无数据可展示", x=0.5, y=0.5, showarrow=False, font=dict(size=16))
        fig.update_layout(title=f"{msku} 库存预测", plot_bgcolor="#f8f9fa", height=400)
        return fig

    row = data.iloc[0]
    start_date = row["记录时间"]  # 记录时间
    end_date = END_DATE  # 2025年12月31日
    base_avg = row["日均"] if row["日均"] > 0 else 0.1  # 基础日均
    total_stock = row["全部总库存"]
    remaining_stock = total_stock

    # 1. 定义时间段系数（需与load_and_preprocess_data_from_df保持一致）
    TIME_PERIODS = [
        {"start": datetime(2025, 10, 15), "end": datetime(2025, 10, 31), "coefficient": 0.95},
        {"start": datetime(2025, 11, 1), "end": datetime(2025, 11, 15), "coefficient": 0.91},
        {"start": datetime(2025, 11, 16), "end": datetime(2025, 11, 30), "coefficient": 0.72},
        {"start": datetime(2025, 12, 1), "end": datetime(2025, 12, 31), "coefficient": 0.43}
    ]

    # 2. 分阶段计算剩余库存（核心修改）
    forecast_dates = []
    forecast_stock = []
    current_date = start_date

    while current_date <= end_date and remaining_stock > 0:
        # 确定当前日期对应的系数
        current_coeff = 1.0  # 默认系数1.0
        for period in TIME_PERIODS:
            if period["start"] <= current_date <= period["end"]:
                current_coeff = period["coefficient"]
                break

        # 计算当天的销量（基础日均×当前系数）
        daily_sales = base_avg * current_coeff
        # 扣减库存（确保不小于0）
        remaining_stock = max(remaining_stock - daily_sales, 0)

        # 记录日期和库存
        forecast_dates.append(current_date)
        forecast_stock.append(remaining_stock)

        # 日期+1天
        current_date += timedelta(days=1)

    # 3. 生成折线图（使用分阶段计算的结果）
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_stock,
        mode="lines+markers",
        line=dict(color="#4169E1", width=2),
        name="预计库存（分阶段系数）"
    ))

    # 4. 添加目标日期线（保持原逻辑）
    fig.add_vline(
        x=TARGET_DATE.timestamp() * 1000,  # 转换为毫秒级时间戳
        line_dash="dash",
        line_color="#DC143C",  # 红色
        annotation_text="目标消耗日期",
        annotation_position="top right",
        annotation_font=dict(color="#DC143C")
    )

    # 5. 添加时间段系数标注（可选，增强可读性）
    for period in TIME_PERIODS:
        # 在时间段起始位置添加系数标注
        fig.add_annotation(
            x=period["start"],
            y=max(forecast_stock) * 0.9,  # 标注位置在库存峰值的90%处
            text=f"{period['start'].strftime('%m-%d')}起系数: {period['coefficient']}",
            showarrow=True,
            arrowhead=1,
            arrowcolor=STATUS_COLORS["低滞销风险"],
            font=dict(size=10, color=STATUS_COLORS["低滞销风险"])
        )

    # 6. 图表布局（保持原逻辑）
    fig.update_layout(
        title=f"{msku} 库存消耗预测（含分阶段销量系数）",
        xaxis_title="日期",
        yaxis_title="剩余库存",
        plot_bgcolor="#f8f9fa",
        height=400,
        margin=dict(t=50, b=20, l=20, r=20)
    )

    # 7. 横坐标设置（保持原逻辑）
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=30, label="30天", step="day", stepmode="backward"),
                dict(count=1, label="1月", step="month", stepmode="backward"),
                dict(step="all", label="全部")
            ])
        ),
        type="date",
        tickformat="%Y年%m月%d日",
        dtick=864000000,  # 10天的毫秒数
        ticklabelmode="period"
    )

    return fig


def render_product_detail_chart(df, msku):
    """渲染单个产品的历史库存预测对比图（含分阶段系数）"""
    if df is None or df.empty:
        fig = go.Figure()
        fig.add_annotation(text="无数据可展示", x=0.5, y=0.5, showarrow=False, font=dict(size=16))
        fig.update_layout(title=f"{msku} 历史库存预测", plot_bgcolor="#f8f9fa", height=400)
        return fig

    # 获取该MSKU的所有记录
    product_data = df[df["MSKU"] == msku].sort_values("记录时间")
    if product_data.empty:
        fig = go.Figure()
        fig.add_annotation(text="无此产品数据", x=0.5, y=0.5, showarrow=False, font=dict(size=16))
        fig.update_layout(title=f"{msku} 历史库存预测", plot_bgcolor="#f8f9fa", height=400)
        return fig

    # 1. 定义时间段系数（需与其他函数保持一致）
    TIME_PERIODS = [
        {"start": datetime(2025, 10, 15), "end": datetime(2025, 10, 31), "coefficient": 0.95},
        {"start": datetime(2025, 11, 1), "end": datetime(2025, 11, 15), "coefficient": 0.91},
        {"start": datetime(2025, 11, 16), "end": datetime(2025, 11, 30), "coefficient": 0.72},
        {"start": datetime(2025, 12, 1), "end": datetime(2025, 12, 31), "coefficient": 0.43}
    ]

    # 2. 创建图表
    fig = go.Figure()

    # 3. 为每个记录时间添加预测线（核心修改：分阶段计算）
    for _, row in product_data.iterrows():
        record_date = row["记录时间"]
        label = record_date.strftime("%Y-%m-%d")
        base_avg = row["日均"] if row["日均"] > 0 else 0.1
        total_stock = row["全部总库存"]
        remaining_stock = total_stock
        end_date = END_DATE

        # 分阶段计算该记录时间的预测
        forecast_dates = []
        forecast_stock = []
        current_date = record_date

        while current_date <= end_date and remaining_stock > 0:
            # 确定当前日期的系数
            current_coeff = 1.0
            for period in TIME_PERIODS:
                if period["start"] <= current_date <= period["end"]:
                    current_coeff = period["coefficient"]
                    break

            # 计算当天销量并扣减库存
            daily_sales = base_avg * current_coeff
            remaining_stock = max(remaining_stock - daily_sales, 0)

            # 记录数据
            forecast_dates.append(current_date)
            forecast_stock.append(remaining_stock)

            # 日期+1天
            current_date += timedelta(days=1)

        # 添加该记录时间的预测线
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast_stock,
            mode="lines",
            name=f"{label}（记录）",
            line=dict(width=2)
        ))

    # 4. 添加目标日期线（保持原逻辑）
    fig.add_vline(
        x=TARGET_DATE.timestamp() * 1000,
        line_dash="dash",
        line_color="#DC143C",
        annotation_text="目标消耗日期",
        annotation_position="top right",
        annotation_font=dict(color="#DC143C")
    )

    # 5. 图表布局（保持原逻辑）
    fig.update_layout(
        title=f"{msku} 不同记录时间的库存预测对比（含分阶段系数）",
        xaxis_title="日期",
        yaxis_title="剩余库存",
        plot_bgcolor="#f8f9fa",
        height=400,
        margin=dict(t=50, b=20, l=20, r=20),
        legend_title="记录时间"
    )

    # 6. 横坐标设置（保持原逻辑）
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=30, label="30天", step="day", stepmode="backward"),
                dict(count=1, label="1月", step="month", stepmode="backward"),
                dict(step="all", label="全部")
            ])
        ),
        type="date",
        tickformat="%Y年%m月%d日",
        dtick=864000000,
        ticklabelmode="period"
    )

    return fig


# 在render_stock_forecast_chart函数中修改目标日期线
def render_stock_forecast_chart(data, msku):
    """渲染单个MSKU的库存预测图表（从记录时间到2025年12月31日，含分阶段系数）"""
    if data is None or data.empty:
        fig = go.Figure()
        fig.add_annotation(text="无数据可展示", x=0.5, y=0.5, showarrow=False, font=dict(size=16))
        fig.update_layout(title=f"{msku} 库存预测", plot_bgcolor="#f8f9fa", height=400)
        return fig

    row = data.iloc[0]
    start_date = row["记录时间"]  # 记录时间
    end_date = END_DATE  # 2025年12月31日
    base_avg = row["日均"] if row["日均"] > 0 else 0.1  # 基础日均
    total_stock = row["全部总库存"]
    remaining_stock = total_stock

    # 1. 定义时间段系数（需与load_and_preprocess_data_from_df保持一致）
    TIME_PERIODS = [
        {"start": datetime(2025, 10, 15), "end": datetime(2025, 10, 31), "coefficient": 0.95},
        {"start": datetime(2025, 11, 1), "end": datetime(2025, 11, 15), "coefficient": 0.91},
        {"start": datetime(2025, 11, 16), "end": datetime(2025, 11, 30), "coefficient": 0.72},
        {"start": datetime(2025, 12, 1), "end": datetime(2025, 12, 31), "coefficient": 0.43}
    ]

    # 2. 分阶段计算剩余库存（核心修改）
    forecast_dates = []
    forecast_stock = []
    current_date = start_date

    while current_date <= end_date and remaining_stock > 0:
        # 确定当前日期对应的系数
        current_coeff = 1.0  # 默认系数1.0
        for period in TIME_PERIODS:
            if period["start"] <= current_date <= period["end"]:
                current_coeff = period["coefficient"]
                break

        # 计算当天的销量（基础日均×当前系数）
        daily_sales = base_avg * current_coeff
        # 扣减库存（确保不小于0）
        remaining_stock = max(remaining_stock - daily_sales, 0)

        # 记录日期和库存
        forecast_dates.append(current_date)
        forecast_stock.append(remaining_stock)

        # 日期+1天
        current_date += timedelta(days=1)

    # 3. 生成折线图（使用分阶段计算的结果）
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_stock,
        mode="lines+markers",
        line=dict(color="#4169E1", width=2),
        name="预计库存（分阶段系数）"
    ))

    # 4. 添加目标日期线（保持原逻辑）
    fig.add_vline(
        x=TARGET_DATE.timestamp() * 1000,  # 转换为毫秒级时间戳
        line_dash="dash",
        line_color="#DC143C",  # 红色
        annotation_text="目标消耗日期",
        annotation_position="top right",
        annotation_font=dict(color="#DC143C")
    )

    # 5. 添加时间段系数标注（可选，增强可读性）
    for period in TIME_PERIODS:
        # 在时间段起始位置添加系数标注
        fig.add_annotation(
            x=period["start"],
            y=max(forecast_stock) * 0.9,  # 标注位置在库存峰值的90%处
            text=f"{period['start'].strftime('%m-%d')}起系数: {period['coefficient']}",
            showarrow=True,
            arrowhead=1,
            arrowcolor=STATUS_COLORS["低滞销风险"],
            font=dict(size=10, color=STATUS_COLORS["低滞销风险"])
        )

    # 6. 图表布局（保持原逻辑）
    fig.update_layout(
        title=f"{msku} 库存消耗预测（含分阶段销量系数）",
        xaxis_title="日期",
        yaxis_title="剩余库存",
        plot_bgcolor="#f8f9fa",
        height=400,
        margin=dict(t=50, b=20, l=20, r=20)
    )

    # 7. 横坐标设置（保持原逻辑）
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=30, label="30天", step="day", stepmode="backward"),
                dict(count=1, label="1月", step="month", stepmode="backward"),
                dict(step="all", label="全部")
            ])
        ),
        type="date",
        tickformat="%Y年%m月%d日",
        dtick=864000000,  # 10天的毫秒数
        ticklabelmode="period"
    )

    return fig

# ------------------------------
# 3. 主函数（页面布局）
# ------------------------------
def main():
    # 初始化会话状态
    if "current_page" not in st.session_state:
        st.session_state.current_page = 1
    if "current_status_page" not in st.session_state:
        st.session_state.current_status_page = 1

    # 侧边栏内容（固定）
    with st.sidebar:
        st.title("侧栏信息")
        from datetime import datetime  # 正确导入方式
        # 首先确保导入了需要的类
        from datetime import datetime, timedelta  # 关键：导入timedelta
        # 显示日期信息
        # 计算当周周一的日期
        today = datetime.now().date()
        # weekday()返回0-6，其中0是周一，6是周日
        # 如果今天是周一，直接使用今天；否则计算上一个周一
        days_to_monday = today.weekday()  # 距离本周一的天数（0表示今天就是周一）
        monday_of_week = today - timedelta(days=days_to_monday)

        # 显示当周周一信息
        st.info(f"当周周一：{monday_of_week.strftime('%Y年%m月%d日')}")

        # 显示目标日期和剩余天数
        days_remaining = (TARGET_DATE.date() - monday_of_week).days
        st.info(f"目标消耗完成日期：{TARGET_DATE.strftime('%Y年%m月%d日')}")
        st.warning(f"距离目标日期剩余：{days_remaining}天")
        # 添加MSKU滞销风险分类说明
        st.subheader("MSKU滞销风险分类：")
        st.markdown("""
        - **健康**：预计总库存用完时间≤2025年12月1日；
        - **低滞销风险**：预计用完时间比目标时间多出来的天在0-10天内；
        - **中滞销风险**：预计用完时间比目标时间多出来的天10-20天内；
        - **高滞销风险**：预计用完时间比目标时间多出来的天数>20天。
        """)

        # 注释掉文件上传部分
        # st.subheader("数据上传")
        # uploaded_file = st.file_uploader("上传Excel文件", type=["xlsx"])

        # 新增：直接读取GitHub仓库中的数据文件
        st.subheader("数据加载中...")
        try:
            # 正确的Raw格式链接
            data_url = "https://raw.githubusercontent.com/Jane-zzz-123/-/main/weekday.xlsx"

            # 从URL读取数据
            response = requests.get(data_url)
            response.raise_for_status()  # 检查请求是否成功
            excel_data = BytesIO(response.content)
            import pandas as pd  # 导入pandas库并命名为pd

            # 只读取存在的"当前数据"sheet
            current_data = pd.read_excel(
                excel_data,
                sheet_name="当前数据",
                engine='openpyxl'  # 明确指定引擎
            )

            # ------------------------------
            # 新增：调用预处理函数，执行计算逻辑
            # （包括生成"状态判断"等所有衍生列）
            # ------------------------------
            df = load_and_preprocess_data_from_df(current_data)  # 关键修改：执行计算
            if df is None:  # 处理预处理失败的情况
                st.error("数据预处理失败，无法继续")
                st.stop()

            # ------------------------------
            # 新增：根据用户权限筛选店铺
            # ------------------------------
            allowed_stores = st.session_state.get("allowed_stores")
            if allowed_stores is not None:  # 非管理员（有店铺限制）
                # 筛选df中"店铺"列属于allowed_stores的行
                df = df[df["店铺"].isin(allowed_stores)].copy()
                # 检查筛选后是否有数据
                if df.empty:
                    st.error(f"您有权限的店铺（{', '.join(allowed_stores)}）没有数据")
                    st.stop()  # 无数据则停止运行

            st.success("数据加载成功！")
        except Exception as e:
            st.error(f"数据加载失败：{str(e)}")
            # 增加调试信息，帮助确认问题
            try:
                # 尝试获取文件中的所有sheet名称
                excel_data.seek(0)
                xl = pd.ExcelFile(excel_data, engine='openpyxl')
                st.error(f"Excel文件中实际存在的sheet：{xl.sheet_names}")
            except:
                pass
            st.stop()  # 加载失败则停止运行

    # 主内容区标题
    st.title("年份品滞销风险分析仪表盘")

    # 初始化session_state存储筛选状态
    if "filter_status" not in st.session_state:
        st.session_state.filter_status = None
    if "current_page" not in st.session_state:
        st.session_state.current_page = 1


    # 获取所有记录时间并排序
    all_dates = sorted(df["记录时间"].unique())
    latest_date = all_dates[-1] if all_dates else None

    # ------------------------------
    # 第一部分：整体风险分析
    # ------------------------------
    st.header("一、整体风险分析")

    # 记录时间筛选器
    selected_date = st.selectbox(
        "选择记录时间",
        options=all_dates,
        index=len(all_dates) - 1 if all_dates else 0,
        format_func=lambda x: x.strftime("%Y年%m月%d日")
    )

    # 获取当前周和上周数据
    current_data = get_week_data(df, selected_date)
    prev_data = get_previous_week_data(df, selected_date)

    st.subheader("1 店铺整体分析")
    if current_data is not None and not current_data.empty:
        stores = sorted(current_data["店铺"].unique())
        selected_store = st.selectbox("选择店铺进行分析", options=stores)

        if selected_store:
            # 当前店铺数据与指标
            store_current_data = current_data[current_data["店铺"] == selected_store].copy()
            store_current_metrics = calculate_status_metrics(store_current_data)

            # 获取上周店铺数据与指标（修改：返回完整数据+滞销库存，用于后续对比）
            def get_store_last_week_metrics():
                from datetime import timedelta
                current_date = pd.to_datetime(store_current_data["记录时间"].iloc[0])
                last_week_start = current_date - timedelta(days=14)
                last_week_end = current_date - timedelta(days=7)

                if 'prev_data' in locals() and prev_data is not None and not prev_data.empty:
                    prev_data_filtered = prev_data[prev_data["店铺"] == selected_store].copy()
                    prev_data_filtered['记录时间'] = pd.to_datetime(prev_data_filtered['记录时间'])
                    last_week_data = prev_data_filtered[
                        (prev_data_filtered['记录时间'] >= last_week_start) &
                        (prev_data_filtered['记录时间'] <= last_week_end)
                        ]
                    if not last_week_data.empty:
                        # 计算上周状态指标+总滞销库存
                        metrics = calculate_status_metrics(last_week_data)
                        metrics["总滞销库存"] = last_week_data[
                            "总滞销库存"].sum() if "总滞销库存" in last_week_data.columns else 0
                        return metrics, last_week_data  # 新增返回上周原始数据
                # 无数据时返回默认值（含总滞销库存）
                return {
                    "总MSKU数": 0, "健康": 0, "低滞销风险": 0, "中滞销风险": 0, "高滞销风险": 0,
                    "总滞销库存": 0
                }, None

            # 调用修改：接收上周指标+原始数据
            store_last_week_metrics, last_week_data = get_store_last_week_metrics()

            # 新增1：计算状态变化（变好/不变/变差的MSKU数）
            status_change = {
                "健康": {"改善": 0, "不变": 0, "恶化": 0},
                "低滞销风险": {"改善": 0, "不变": 0, "恶化": 0},
                "中滞销风险": {"改善": 0, "不变": 0, "恶化": 0},
                "高滞销风险": {"改善": 0, "不变": 0, "恶化": 0}
            }
            # 状态严重程度排序（用于判断变化方向：健康 < 低风险 < 中风险 < 高风险）
            status_severity = {"健康": 0, "低滞销风险": 1, "中滞销风险": 2, "高滞销风险": 3}

            # 匹配MSKU计算状态变化
            if last_week_data is not None and not last_week_data.empty and "MSKU" in store_current_data.columns:
                merged_data = pd.merge(
                    store_current_data[["MSKU", "状态判断"]],
                    last_week_data[["MSKU", "状态判断"]],
                    on="MSKU",
                    suffixes=("_current", "_prev"),
                    how="inner"
                )
                for _, row in merged_data.iterrows():
                    current_status = row["状态判断_current"]
                    prev_status = row["状态判断_prev"]
                    if current_status not in status_severity or prev_status not in status_severity:
                        continue
                    if current_status == prev_status:
                        status_change[current_status]["不变"] += 1
                    elif status_severity[current_status] < status_severity[prev_status]:
                        status_change[current_status]["改善"] += 1  # 当前状态更轻=变好
                    else:
                        status_change[current_status]["恶化"] += 1  # 当前状态更重=变差

            # 计算对比指标（修改：百分比保留两位小数）
            store_metrics = {}
            for metric in ["总MSKU数", "健康", "低滞销风险", "中滞销风险", "高滞销风险"]:
                current = int(store_current_metrics[metric])
                last_week = int(store_last_week_metrics[metric])
                diff = current - last_week
                pct = (diff / last_week) * 100 if last_week != 0 else 0.0
                store_metrics[metric] = {
                    "current": current,
                    "last_week": last_week,
                    "diff": diff,
                    "pct": round(pct, 2)  # 原1位→2位小数
                }

            # 新增2：生成滞销库存对比文本（保留两位小数+环比百分比）
            def get_overstock_compare_text(current_overstock, last_week_overstock, status=None):
                # 处理数值格式（保留两位小数）
                current = round(float(current_overstock), 2)
                last_week = round(float(last_week_overstock), 2)

                if last_week == 0:
                    return f"<br><span style='color:#666; font-size:0.8em;'>{status + ' ' if status else ''}总滞销库存: {current:.2f}</span>"

                # 计算差异和环比
                diff = current - last_week
                trend = "↑" if diff > 0 else "↓" if diff < 0 else "→"
                color = "#DC143C" if diff > 0 else "#2E8B57" if diff < 0 else "#666"
                pct = (diff / last_week) * 100 if last_week != 0 else 0.0
                pct_text = f"{abs(pct):.2f}%"

                return f"<br><span style='color:{color}; font-size:0.8em;'>{status + ' ' if status else ''}总滞销库存: {current:.2f} ({trend}{abs(diff):.2f} {pct_text})</span>"

            # 新增3：生成状态变化文本（颜色区分变好/不变/变差）
            def get_status_change_text(status):
                changes = status_change[status]
                total = changes["改善"] + changes["不变"] + changes["恶化"]
                if total == 0:
                    return "<br><span style='color:#666; font-size:0.8em;'>状态变化: 无数据</span>"

                return f"""<br>
                <span style='color:#2E8B57; font-size:0.8em;'>改善: {changes['改善']}</span> | 
                <span style='color:#666; font-size:0.8em;'>不变: {changes['不变']}</span> | 
                <span style='color:#DC143C; font-size:0.8em;'>恶化: {changes['恶化']}</span>
                """

            # 生成对比文本（修改：百分比显示两位小数）
            def get_compare_text(metric_data, metric_name):
                if metric_data["last_week"] == 0:
                    return "<br><span style='color:#666; font-size:0.8em;'>无上周数据</span>"

                trend = "↑" if metric_data["diff"] > 0 else "↓" if metric_data["diff"] < 0 else "→"
                color = "#DC143C" if metric_data["diff"] > 0 else "#2E8B57" if metric_data["diff"] < 0 else "#666"
                pct_text = f"{abs(metric_data['pct']):.2f}%"  # 原1位→2位小数

                if metric_name == "总MSKU数":
                    return f"<br><span style='color:{color}; font-size:0.8em;'>{trend} 上周{metric_data['last_week']}，变化{metric_data['diff']} ({pct_text})</span>"
                else:
                    status = "上升" if metric_data["diff"] > 0 else "下降" if metric_data["diff"] < 0 else "无变化"
                    return f"<br><span style='color:{color}; font-size:0.8em;'>{trend} 上周{metric_data['last_week']}，{status}{abs(metric_data['diff'])} ({pct_text})</span>"

            # 显示带上周对比的指标卡片（核心修改：新增滞销库存+状态变化）
            cols = st.columns(5)
            with cols[0]:
                data = store_metrics["总MSKU数"]
                compare_text = get_compare_text(data, "总MSKU数")
                # 总滞销库存（全状态合计）
                total_overstock = store_current_data[
                    "总滞销库存"].sum() if "总滞销库存" in store_current_data.columns else 0
                last_week_total_overstock = store_last_week_metrics.get("总滞销库存", 0)
                overstock_text = get_overstock_compare_text(total_overstock, last_week_total_overstock)

                render_metric_card(
                    f"{selected_store} 总MSKU数{compare_text}{overstock_text}",
                    data["current"],
                    data["diff"],
                    data["pct"],
                    "#000000"
                )
            with cols[1]:
                data = store_metrics["健康"]
                compare_text = get_compare_text(data, "健康")
                # 健康状态专属滞销库存
                healthy_overstock = store_current_data[store_current_data["状态判断"] == "健康"][
                    "总滞销库存"].sum() if (
                            "状态判断" in store_current_data.columns and "总滞销库存" in store_current_data.columns) else 0
                last_week_healthy_overstock = last_week_data[last_week_data["状态判断"] == "健康"][
                    "总滞销库存"].sum() if (
                            last_week_data is not None and "状态判断" in last_week_data.columns and "总滞销库存" in last_week_data.columns) else 0
                overstock_text = get_overstock_compare_text(healthy_overstock, last_week_healthy_overstock,
                                                            status="健康")
                # 状态变化文本
                change_text = get_status_change_text("健康")

                render_metric_card(
                    f"{selected_store} 健康{compare_text}{overstock_text}{change_text}",
                    data["current"],
                    data["diff"],
                    data["pct"],
                    STATUS_COLORS["健康"]
                )
            with cols[2]:
                data = store_metrics["低滞销风险"]
                compare_text = get_compare_text(data, "低滞销风险")
                # 低风险专属滞销库存
                low_risk_overstock = store_current_data[store_current_data["状态判断"] == "低滞销风险"][
                    "总滞销库存"].sum() if (
                            "状态判断" in store_current_data.columns and "总滞销库存" in store_current_data.columns) else 0
                last_week_low_risk_overstock = last_week_data[last_week_data["状态判断"] == "低滞销风险"][
                    "总滞销库存"].sum() if (
                            last_week_data is not None and "状态判断" in last_week_data.columns and "总滞销库存" in last_week_data.columns) else 0
                overstock_text = get_overstock_compare_text(low_risk_overstock, last_week_low_risk_overstock,
                                                            status="低风险")
                # 状态变化文本
                change_text = get_status_change_text("低滞销风险")

                render_metric_card(
                    f"{selected_store} 低滞销风险{compare_text}{overstock_text}{change_text}",
                    data["current"],
                    data["diff"],
                    data["pct"],
                    STATUS_COLORS["低滞销风险"]
                )
            with cols[3]:
                data = store_metrics["中滞销风险"]
                compare_text = get_compare_text(data, "中滞销风险")
                # 中风险专属滞销库存
                mid_risk_overstock = store_current_data[store_current_data["状态判断"] == "中滞销风险"][
                    "总滞销库存"].sum() if (
                            "状态判断" in store_current_data.columns and "总滞销库存" in store_current_data.columns) else 0
                last_week_mid_risk_overstock = last_week_data[last_week_data["状态判断"] == "中滞销风险"][
                    "总滞销库存"].sum() if (
                            last_week_data is not None and "状态判断" in last_week_data.columns and "总滞销库存" in last_week_data.columns) else 0
                overstock_text = get_overstock_compare_text(mid_risk_overstock, last_week_mid_risk_overstock,
                                                            status="中风险")
                # 状态变化文本
                change_text = get_status_change_text("中滞销风险")

                render_metric_card(
                    f"{selected_store} 中滞销风险{compare_text}{overstock_text}{change_text}",
                    data["current"],
                    data["diff"],
                    data["pct"],
                    STATUS_COLORS["中滞销风险"]
                )
            with cols[4]:
                data = store_metrics["高滞销风险"]
                compare_text = get_compare_text(data, "高滞销风险")
                # 高风险专属滞销库存
                high_risk_overstock = store_current_data[store_current_data["状态判断"] == "高滞销风险"][
                    "总滞销库存"].sum() if (
                            "状态判断" in store_current_data.columns and "总滞销库存" in store_current_data.columns) else 0
                last_week_high_risk_overstock = last_week_data[last_week_data["状态判断"] == "高滞销风险"][
                    "总滞销库存"].sum() if (
                            last_week_data is not None and "状态判断" in last_week_data.columns and "总滞销库存" in last_week_data.columns) else 0
                overstock_text = get_overstock_compare_text(high_risk_overstock, last_week_high_risk_overstock,
                                                            status="高风险")
                # 状态变化文本
                change_text = get_status_change_text("高滞销风险")

                render_metric_card(
                    f"{selected_store} 高滞销风险{compare_text}{overstock_text}{change_text}",
                    data["current"],
                    data["diff"],
                    data["pct"],
                    STATUS_COLORS["高滞销风险"]
                )

            # 图表部分：修改为「一行三列（状态分布+状态占比+状态变化）+ 下方组合图」布局（修复图例方向参数错误）
            # 1. 第一行：三列布局
            col1, col2, col3 = st.columns(3)

            # 1.1 第一列：原状态分布柱状图（保持不变）
            with col1:
                status_data = pd.DataFrame({
                    "状态": ["健康", "低滞销风险", "中滞销风险", "高滞销风险"],
                    "MSKU数": [store_current_metrics[stat] for stat in
                               ["健康", "低滞销风险", "中滞销风险", "高滞销风险"]]
                })

                fig_status = px.bar(
                    status_data,
                    x="状态",
                    y="MSKU数",
                    color="状态",
                    color_discrete_map=STATUS_COLORS,
                    title=f"{selected_store} 状态分布",
                    text="MSKU数",
                    height=400
                )

                fig_status.update_traces(
                    textposition="outside",
                    textfont=dict(size=12, weight="bold"),
                    marker=dict(line=dict(color="#fff", width=1))
                )

                fig_status.update_layout(
                    xaxis_title="风险状态",
                    yaxis_title="MSKU数量",
                    showlegend=True,
                    plot_bgcolor="#f8f9fa",
                    margin=dict(t=50, b=20, l=20, r=20)
                )

                st.plotly_chart(fig_status, use_container_width=True)

            # 1.2 第二列：状态判断饼图（保持不变）
            with col2:
                pie_data = pd.DataFrame({
                    "状态": ["健康", "低滞销风险", "中滞销风险", "高滞销风险"],
                    "MSKU数": [store_current_metrics[stat] for stat in
                               ["健康", "低滞销风险", "中滞销风险", "高滞销风险"]]
                })
                total_msku = pie_data["MSKU数"].sum()
                pie_data["占比(%)"] = pie_data["MSKU数"].apply(
                    lambda x: round((x / total_msku) * 100, 1) if total_msku != 0 else 0.0
                )
                pie_data["自定义标签"] = pie_data.apply(
                    lambda row: f"{row['状态']}<br>{row['MSKU数']}个<br>({row['占比(%)']}%)",
                    axis=1
                )

                fig_pie = px.pie(
                    pie_data,
                    values="MSKU数",
                    names="状态",
                    color="状态",
                    color_discrete_map=STATUS_COLORS,
                    title=f"{selected_store} 状态占比",
                    height=400,
                    labels={"MSKU数": "MSKU数量"}
                )

                fig_pie.update_traces(
                    text=pie_data["自定义标签"],
                    textinfo="text",
                    textfont=dict(size=10, weight="bold"),
                    hovertemplate="%{label}: %{value}个 (%{percent:.1%})"
                )

                fig_pie.update_layout(
                    showlegend=True,
                    legend_title="风险状态",
                    plot_bgcolor="#f8f9fa",
                    margin=dict(t=50, b=20, l=20, r=20)
                )

                st.plotly_chart(fig_pie, use_container_width=True)

            # 1.3 第三列：环比上周库存滞销情况变化柱形图（保持不变）
            with col3:
                change_data = pd.DataFrame({
                    "状态": ["健康", "低滞销风险", "中滞销风险", "高滞销风险"],
                    "本周MSKU数": [store_current_metrics[stat] for stat in
                                   ["健康", "低滞销风险", "中滞销风险", "高滞销风险"]],
                    "上周MSKU数": [store_last_week_metrics[stat] for stat in
                                   ["健康", "低滞销风险", "中滞销风险", "高滞销风险"]]
                })
                change_data_long = pd.melt(
                    change_data,
                    id_vars="状态",
                    value_vars=["本周MSKU数", "上周MSKU数"],
                    var_name="周期",
                    value_name="MSKU数"
                )

                fig_change = px.bar(
                    change_data_long,
                    x="状态",
                    y="MSKU数",
                    color="周期",
                    barmode="group",
                    color_discrete_map={"本周MSKU数": "#2E86AB", "上周MSKU数": "#A23B72"},
                    title=f"{selected_store} 状态变化对比",
                    height=400,
                    text="MSKU数"
                )

                fig_change.update_traces(
                    textposition="outside",
                    textfont=dict(size=10, weight="bold"),
                    marker=dict(line=dict(color="#fff", width=1))
                )

                fig_change.update_layout(
                    xaxis_title="风险状态",
                    yaxis_title="MSKU数量",
                    showlegend=True,
                    legend_title="周期",
                    plot_bgcolor="#f8f9fa",
                    margin=dict(t=50, b=20, l=20, r=20)
                )

                st.plotly_chart(fig_change, use_container_width=True)
            if df is not None and not df.empty and selected_store:
                # 获取当前选中店铺的当前周期数据（筛选店铺）
                current_week_full_data = get_week_data(df, selected_date)
                current_week_store_data = current_week_full_data[
                    current_week_full_data["店铺"] == selected_store] if current_week_full_data is not None else None

                # 获取当前选中店铺的上一周期数据（筛选店铺）
                previous_week_full_data = get_previous_week_data(df, selected_date)
                previous_week_store_data = previous_week_full_data[
                    previous_week_full_data["店铺"] == selected_store] if previous_week_full_data is not None else None

                # 创建店铺级别的汇总表数据
                store_summary_df = create_risk_summary_table(current_week_store_data, previous_week_store_data)

                # 渲染店铺级别的汇总表
                render_risk_summary_table(store_summary_df)

            # 2. 第二部分：下方组合图（修复图例方向参数）
            st.subheader(f"{selected_store} 库存消耗天数分布（MSKU数+总滞销库存）")
            today = pd.to_datetime(store_current_data["记录时间"].iloc[0])
            days_to_target = (TARGET_DATE - today).days

            # 2.1 数据预处理
            valid_days = store_current_data["预计总库存需要消耗天数"].clip(lower=0)
            max_days = valid_days.max() if not valid_days.empty else 0
            bin_width = 20
            num_bins = int((max_days + bin_width - 1) // bin_width)
            bins = [i * bin_width for i in range(num_bins + 1)]
            bin_labels = [f"{bins[i]}-{bins[i + 1]}" for i in range(len(bins) - 1)]

            # 2.2 计算天数区间对应的MSKU数量和总滞销库存
            msku_count = pd.cut(
                valid_days,
                bins=bins,
                labels=bin_labels,
                include_lowest=True
            ).value_counts().sort_index()

            temp_df = store_current_data[["预计总库存需要消耗天数", "总滞销库存"]].copy()
            temp_df["预计总库存需要消耗天数"] = temp_df["预计总库存需要消耗天数"].clip(lower=0)
            temp_df["天数区间"] = pd.cut(
                temp_df["预计总库存需要消耗天数"],
                bins=bins,
                labels=bin_labels,
                include_lowest=True
            )
            overstock_sum = temp_df.groupby("天数区间")["总滞销库存"].sum().sort_index()

            # 2.3 合并数据
            combined_data = pd.DataFrame({
                "天数区间": bin_labels,
                "MSKU数量": [msku_count.get(label, 0) for label in bin_labels],
                "总滞销库存": [overstock_sum.get(label, 0.0) for label in bin_labels]
            })

            # 2.4 创建组合图
            fig_combined = px.bar(
                combined_data,
                x="天数区间",
                y="总滞销库存",
                color_discrete_sequence=["#F18F01"],
                title="库存消耗天数 vs 总滞销库存",
                height=400,
                text="总滞销库存"
            )

            # 添加折线图
            fig_combined.add_scatter(
                x=combined_data["天数区间"],
                y=combined_data["MSKU数量"],
                mode="lines+markers",
                name="MSKU数量",
                yaxis="y2",
                line=dict(color="#C73E1D", width=3),
                marker=dict(color="#C73E1D", size=6),
                text=combined_data["MSKU数量"],
                textposition="top center"
            )

            # 2.5 图表样式优化（核心修复：将orientation="horizontal"改为orientation="h"）
            fig_combined.update_layout(
                yaxis=dict(
                    title=dict(
                        text="总滞销库存",
                        font=dict(color="#F18F01")
                    ),
                    tickfont=dict(color="#F18F01"),
                    showgrid=True,
                    gridcolor="#eee"
                ),
                yaxis2=dict(
                    title=dict(
                        text="MSKU数量",
                        font=dict(color="#C73E1D")
                    ),
                    tickfont=dict(color="#C73E1D"),
                    showgrid=False,
                    overlaying="y",
                    side="right"
                ),
                xaxis=dict(
                    title="库存消耗天数区间（天）",
                    tickangle=45,
                    tickfont=dict(size=10)
                ),
                showlegend=True,
                # 修复：将"horizontal"改为简写"h"
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                plot_bgcolor="#f8f9fa",
                margin=dict(t=50, b=80, l=20, r=20)
            )

            # 2.6 数值显示优化
            fig_combined.update_traces(
                selector=dict(type="bar"),
                texttemplate="%.2f",
                textposition="outside",
                textfont=dict(size=10, weight="bold")
            )
            fig_combined.update_traces(
                selector=dict(type="scatter"),
                texttemplate="%d",
                textfont=dict(size=10, weight="bold")
            )

            st.plotly_chart(fig_combined, use_container_width=True)

            # 产品列表与下载功能
            st.subheader(f"{selected_store} 产品列表")

            # 核心修改1：在display_columns中新增8个列（4个系数+4个调整后日均）
            display_columns = [
                "店铺", "MSKU", "品名", "记录时间",
                # 基础日均列（保留原逻辑）
                "日均", "7天日均", "14天日均", "28天日均",
                # 新增：四个时间段的系数+调整后日均（按时间顺序插入）
                "10月15-31日系数", "10月15-31日调整后日均",
                "11月1-15日系数", "11月1-15日调整后日均",
                "11月16-30日系数", "11月16-30日调整后日均",
                "12月1-31日系数", "12月1-31日调整后日均",
                # 原有其他列（保持不变）
                "FBA+AWD+在途库存", "本地可用", "全部总库存", "预计FBA+AWD+在途用完时间",
                "预计总库存用完", "状态判断", "清库存的目标日均", "FBA+AWD+在途滞销数量",
                "本地滞销数量", "总滞销库存",
                "预计总库存需要消耗天数", "预计用完时间比目标时间多出来的天数", "环比上周库存滞销情况变化"
            ]

            render_product_detail_table(
                store_current_data,
                prev_data[prev_data["店铺"] == selected_store] if (
                        prev_data is not None and not prev_data.empty) else None,
                page=st.session_state.current_page,
                page_size=30,
                table_id=f"store_{selected_store}"
            )

            if not store_current_data.empty:
                # 核心修改2：筛选下载数据时包含新增的8个列（自动过滤不存在的列）
                existing_cols = [col for col in display_columns if col in store_current_data.columns]
                download_data = store_current_data[existing_cols].copy()

                # 日期格式化（保持原逻辑）
                date_cols = ["记录时间", "预计FBA+AWD+在途用完时间", "预计总库存用完"]
                for col in date_cols:
                    if col in download_data.columns:
                        download_data[col] = pd.to_datetime(download_data[col]).dt.strftime("%Y-%m-%d")

                # 生成CSV并提供下载（保持原逻辑）
                csv = download_data.to_csv(index=False, encoding='utf-8-sig')
                file_name = f"{selected_store}_产品列表_{today.strftime('%Y%m%d')}.csv"

                st.download_button(
                    label="下载筛选结果 (CSV)",
                    data=csv,
                    file_name=file_name,
                    mime="text/csv",
                    key=f"download_{selected_store}"
                )
    else:
        st.warning("无店铺数据可分析")

    # 单个MSKU分析
    st.subheader("单个MSKU分析")
    if current_data is not None and not current_data.empty:
        msku_list = sorted(current_data["MSKU"].unique())
        # 添加MSKU查询输入框
        col1, col2 = st.columns([3, 1])
        with col1:
            msku_query = st.text_input(
                "输入MSKU查询",
                placeholder="粘贴MSKU代码快速查询...",
                key="msku_query"
            )
        # 根据查询内容过滤下拉选项
        if msku_query:
            filtered_mskus = [msku for msku in msku_list if msku_query.strip().lower() in msku.lower()]
            if not filtered_mskus:
                st.warning(f"未找到包含 '{msku_query}' 的MSKU，请检查输入")
                filtered_mskus = msku_list  # 未找到时显示全部
        else:
            filtered_mskus = msku_list
        # 下拉选择框（会根据查询内容动态过滤）
        with col2:
            selected_msku = st.selectbox("或从列表选择", options=filtered_mskus, key="msku_select")
        if selected_msku:
            product_data = current_data[current_data["MSKU"] == selected_msku]
            product_info = product_data.iloc[0].to_dict()
            # 产品信息表格
            st.subheader("产品基本信息")
            # 核心修改：在display_cols中新增8个列（4个系数+4个调整后日均）
            display_cols = [
                "MSKU", "品名", "店铺",
                # 基础日均列（保留原逻辑）
                "日均", "7天日均", "14天日均", "28天日均",
                # 新增：四个时间段的系数+调整后日均（按时间顺序插入）
                "10月15-31日系数", "10月15-31日调整后日均",
                "11月1-15日系数", "11月1-15日调整后日均",
                "11月16-30日系数", "11月16-30日调整后日均",
                "12月1-31日系数", "12月1-31日调整后日均",
                # 原有其他列（保持不变）
                "FBA+AWD+在途库存", "本地可用", "全部总库存", "预计FBA+AWD+在途用完时间", "预计总库存用完",
                "状态判断", "清库存的目标日均", "FBA+AWD+在途滞销数量", "本地滞销数量", "总滞销库存",
                "预计总库存需要消耗天数", "预计用完时间比目标时间多出来的天数", "环比上周库存滞销情况变化"
            ]
            # 创建信息表格（保持原逻辑，自动适配新增列）
            info_df = product_data[display_cols].copy()
            # 格式化日期（保持原逻辑）
            date_cols = ["预计FBA+AWD+在途用完时间", "预计总库存用完"]
            for col in date_cols:
                if col in info_df.columns:
                    info_df[col] = pd.to_datetime(info_df[col]).dt.strftime("%Y-%m-%d")
            # 添加状态颜色（保持原逻辑）
            if "状态判断" in info_df.columns:
                info_df["状态判断"] = info_df["状态判断"].apply(
                    lambda x: f"<span style='color:{STATUS_COLORS[x]}; font-weight:bold;'>{x}</span>"
                )
            # 新增：（可选）格式化系数列显示（保留2位小数，增强可读性）
            coefficient_cols = [
                "10月15-31日系数", "11月1-15日系数",
                "11月16-30日系数", "12月1-31日系数"
            ]
            for col in coefficient_cols:
                if col in info_df.columns:
                    info_df[col] = info_df[col].round(2)  # 系数固定为2位小数
            # 显示表格（保持原逻辑）
            st.markdown(info_df.to_html(escape=False, index=False), unsafe_allow_html=True)
            # 库存预测图表（保持原逻辑，图表已基于分阶段日均计算）
            forecast_fig = render_stock_forecast_chart(product_data, selected_msku)
            st.plotly_chart(forecast_fig, use_container_width=True)
    else:
        st.warning("无产品数据可分析")

    # ------------------------------
    # 第二部分：趋势与变化分析
    # ------------------------------
    st.header("2 近一个月的趋势与变化分析")

    # 2.1 三周状态变化趋势（柱状图版本）
    st.subheader("2.1 近一个月状态变化趋势")
    trend_fig = render_four_week_status_chart(df, all_dates)
    st.plotly_chart(trend_fig, use_container_width=True)

    # 2.2 店铺周变化情况
    st.subheader("2.2 店铺周变化情况")
    render_store_weekly_changes(df, all_dates)

    # 店铺趋势图表（分两列显示）
    st.subheader("2.3 店铺状态趋势图")
    render_store_trend_charts(df, all_dates)

    # 2.4 店铺与状态变化联合分析
    st.subheader("2.4 店铺与状态变化联合分析")
    if df is not None and not df.empty:
        # 店铺筛选器（保持原逻辑）
        all_stores = sorted(df["店铺"].unique())
        selected_analysis_store = st.selectbox(
            "选择店铺进行联合分析",
            options=["全部"] + all_stores
        )

        # 筛选数据（保持原逻辑）
        analysis_data = df.copy()
        if selected_analysis_store != "全部":
            analysis_data = analysis_data[analysis_data["店铺"] == selected_analysis_store]

        # 按店铺和MSKU进行排序（保持原逻辑）
        analysis_data = analysis_data.sort_values(by=["店铺", "MSKU"])

        # 核心修改1：在display_columns中新增8个列（4个系数+4个调整后日均）
        display_columns = [
            "MSKU", "品名", "店铺",
            # 基础日均列（保留原逻辑）
            "日均", "7天日均", "14天日均", "28天日均",
            # 新增：四个时间段的系数+调整后日均（按时间顺序插入）
            "10月15-31日系数", "10月15-31日调整后日均",
            "11月1-15日系数", "11月1-15日调整后日均",
            "11月16-30日系数", "11月16-30日调整后日均",
            "12月1-31日系数", "12月1-31日调整后日均",
            # 原有其他列（保持不变）
            "FBA+AWD+在途库存", "本地可用",
            "全部总库存", "预计FBA+AWD+在途用完时间",
            "预计总库存用完", "状态判断", "清库存的目标日均",
            "FBA+AWD+在途滞销数量", "本地滞销数量", "总滞销库存",
            "预计总库存需要消耗天数", "预计用完时间比目标时间多出来的天数",
            "环比上周库存滞销情况变化"
        ]

        # 可选：检查并添加数据中实际存在的类似列（保持原逻辑）
        # if "上期状态" in analysis_data.columns:
        #     display_columns.insert(5, "上期状态")

        # 显示状态变化表（保持原逻辑，自动适配新增列）
        render_status_change_table(
            analysis_data,
            page=st.session_state.current_status_page,
            page_size=30
        )

        # 添加下载按钮（下载筛选后的所有数据）
        if not analysis_data.empty:
            # 核心修改2：在expected_columns中新增8个列（与display_columns同步）
            expected_columns = [
                "MSKU", "品名", "店铺", "记录时间",
                # 基础日均列
                "日均", "7天日均", "14天日均", "28天日均",
                # 新增：四个时间段的系数+调整后日均
                "10月15-31日系数", "10月15-31日调整后日均",
                "11月1-15日系数", "11月1-15日调整后日均",
                "11月16-30日系数", "11月16-30日调整后日均",
                "12月1-31日系数", "12月1-31日调整后日均",
                # 原有其他列
                "FBA+AWD+在途库存", "本地可用",
                "全部总库存", "预计FBA+AWD+在途用完时间",
                "预计总库存用完", "状态判断", "清库存的目标日均",
                "FBA+AWD+在途滞销数量", "本地滞销数量", "总滞销库存",
                "预计总库存需要消耗天数", "预计用完时间比目标时间多出来的天数",
                "环比上周库存滞销情况变化"
            ]

            # 2. 检查数据中实际存在的列（保持原逻辑）
            filtered_data = analysis_data.copy()
            actual_columns = filtered_data.columns.tolist()

            # 3. 找出存在的有效列和缺失的列（保持原逻辑，自动包含新增列）
            valid_columns = [col for col in expected_columns if col in actual_columns]
            missing_columns = [col for col in expected_columns if col not in actual_columns]

            # 4. 显示缺失列的警告信息（保持原逻辑）
            if missing_columns:
                st.warning(f"数据中缺少以下列，已自动跳过：{', '.join(missing_columns)}")

            # 5. 确保至少有一列可用于下载（保持原逻辑）
            if valid_columns:
                download_data = filtered_data[valid_columns]
            else:
                st.error("没有找到有效的列用于生成下载数据，请检查数据格式是否正确")
                download_data = pd.DataFrame()  # 创建空DataFrame避免后续错误

            # 格式化日期列（保持原逻辑）
            if "记录时间" in download_data.columns:
                download_data["记录时间"] = pd.to_datetime(download_data["记录时间"]).dt.strftime("%Y-%m-%d")

            # 生成CSV（保持原逻辑）
            csv = download_data.to_csv(index=False, encoding='utf-8-sig')

            # 构建文件名（保持原逻辑）
            store_part = selected_analysis_store if selected_analysis_store != "全部" else "所有店铺"
            file_name = f"店铺状态变化联合分析_{store_part}.csv"

            # 下载按钮（保持原逻辑）
            st.download_button(
                label="下载筛选结果 (CSV)",
                data=csv,
                file_name=file_name,
                mime="text/csv",
                key="download_status_change_analysis"
            )
    else:
        st.warning("无数据可进行联合分析")

    # 2.5 单个产品详细分析
    st.subheader("2.5 单个产品详细分析")
    if df is not None and not df.empty:
        # 获取所有MSKU并排序（保持原逻辑）
        all_mskus = sorted(df["MSKU"].unique())

        # 添加搜索框（保持原逻辑）
        search_term = st.text_input(
            "搜索产品（MSKU或品名）",
            placeholder="输入关键词搜索..."
        )

        # 根据搜索词过滤产品（保持原逻辑）
        if search_term:
            search_lower = search_term.lower()
            filtered_mskus = []
            for msku in all_mskus:
                product_names = df[df["MSKU"] == msku]["品名"].unique()
                if (search_lower in str(msku).lower() or
                        any(search_lower in str(name).lower() for name in product_names)):
                    filtered_mskus.append(msku)
            if not filtered_mskus:
                st.info(f"没有找到包含 '{search_term}' 的产品，请尝试其他关键词")
                filtered_mskus = all_mskus
        else:
            filtered_mskus = all_mskus

        # 产品筛选器（保持原逻辑）
        selected_analysis_msku = st.selectbox(
            "选择产品进行详细分析",
            options=filtered_mskus
        )

        if selected_analysis_msku:
            # 获取该产品的所有记录（保持原逻辑）
            product_history_data = df[df["MSKU"] == selected_analysis_msku].sort_values("记录时间", ascending=False)

            # 核心修改：在display_cols中新增8个列（4个系数+4个调整后日均）
            display_cols = [
                "MSKU", "品名", "店铺", "记录时间",
                # 基础日均列（保留原逻辑）
                "日均", "7天日均", "14天日均", "28天日均",
                # 新增：四个时间段的系数+调整后日均（按时间顺序插入）
                "10月15-31日系数", "10月15-31日调整后日均",
                "11月1-15日系数", "11月1-15日调整后日均",
                "11月16-30日系数", "11月16-30日调整后日均",
                "12月1-31日系数", "12月1-31日调整后日均",
                # 原有其他列（保持不变）
                "FBA+AWD+在途库存", "本地可用", "全部总库存", "预计FBA+AWD+在途用完时间", "预计总库存用完",
                "状态判断", "清库存的目标日均", "FBA+AWD+在途滞销数量", "本地滞销数量", "总滞销库存",
                "预计总库存需要消耗天数", "预计用完时间比目标时间多出来的天数", "环比上周库存滞销情况变化"
            ]

            # 筛选并格式化表格数据（保持原逻辑，自动适配新增列）
            table_data = product_history_data[display_cols].copy()

            # 格式化日期列（保持原逻辑）
            date_cols = ["记录时间", "预计FBA+AWD+在途用完时间", "预计总库存用完"]
            for col in date_cols:
                if col in table_data.columns:
                    table_data[col] = pd.to_datetime(table_data[col]).dt.strftime("%Y-%m-%d")

            # 添加状态颜色（保持原逻辑）
            if "状态判断" in table_data.columns:
                table_data["状态判断"] = table_data["状态判断"].apply(
                    lambda x: f"<span style='color:{STATUS_COLORS[x]}; font-weight:bold;'>{x}</span>"
                )

            # 添加环比上周库存滞销情况变化颜色（保持原逻辑）
            if "环比上周库存滞销情况变化" in table_data.columns:
                def color_status_change(x):
                    if x == "改善":
                        return f"<span style='color:#2E8B57; font-weight:bold;'>{x}</span>"
                    elif x == "恶化":
                        return f"<span style='color:#DC143C; font-weight:bold;'>{x}</span>"
                    else:  # 维持不变
                        return f"<span style='color:#000000; font-weight:bold;'>{x}</span>"

                table_data["环比上周库存滞销情况变化"] = table_data["环比上周库存滞销情况变化"].apply(
                    color_status_change)

            # 显示产品历史数据表格（保持原逻辑）
            st.subheader("产品历史数据")
            st.markdown(table_data.to_html(escape=False, index=False), unsafe_allow_html=True)

            # 生成库存预测对比图（已在之前修改，含分阶段系数）
            forecast_chart = render_product_detail_chart(df, selected_analysis_msku)
            st.plotly_chart(forecast_chart, use_container_width=True)
    else:
        st.warning("无产品数据可进行详细分析")


if __name__ == "__main__":
    main()
