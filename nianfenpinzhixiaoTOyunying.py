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
# 用户认证与权限管理
def check_credentials():
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
    all_users = list(USER_PERMISSIONS.keys())
    def verify():
        username = st.session_state.get("selected_user", "")
        password = st.session_state.get("password", "")
        if username in USER_PERMISSIONS:
            stored_pwd, stores = USER_PERMISSIONS[username]
            if password == stored_pwd:
                st.session_state["authenticated"] = True
                st.session_state["allowed_stores"] = stores
                del st.session_state["password"]
            else:
                st.session_state["authenticated"] = False
        else:
            st.session_state["authenticated"] = False
    if "authenticated" not in st.session_state or not st.session_state["authenticated"]:
        st.title("用户登录")
        st.selectbox(
            "请选择用户名",
            options=all_users,
            key="selected_user",
            on_change=verify
        )
        st.text_input(
            "请输入密码",
            type="password",
            key="password",
            on_change=verify
        )
        if "authenticated" in st.session_state and not st.session_state["authenticated"]:
            st.error("密码错误，请重新输入")
        return False
    return True
if not check_credentials():
    st.stop()

st.set_page_config(page_title="库存滞销风险分析平台", layout="wide", initial_sidebar_state="expanded")
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
STATUS_COLORS = {
    "健康": "#2E8B57",                    # 深绿
    "低滞销风险": "#FFD700",                # 金色
    "中滞销风险": "#FF8C00",                # 橙红
    "高滞销风险": "#DC143C",                # 深红
    "非年份品（无目标日期风险）": "#808080"  # 灰色（区分非年份品）
}
TARGET_DATE = datetime(2026, 10, 31)  # 目标消耗完成日期
END_DATE = datetime(2026, 12, 31)  # 预测截止日期
# ========== 新增：库存周转状态的颜色映射（专门给新列用） ==========
TURNOVER_STATUS_COLORS = {
    "库存周转健康": "#2E8B57",       # 绿色（和原健康色一致）
    "轻度滞销风险": "#4169E1",     # 蓝色（和原低风险色一致）
    "中度滞销风险": "#FFD700",     # 黄色（和原中风险色一致）
    "严重滞销风险": "#DC143C",     # 红色（和原高风险色一致）
    "数据异常": "#808080"          # 灰色（处理空值/负数的兜底）
}

# 1. 数据加载与预处理函数
@st.cache_data(ttl=3600)
def load_and_preprocess_data_from_df(df):
    try:
        TIME_PERIODS = [
            {"name": "december", "start": datetime(2026, 10, 16), "end": datetime(2026, 11, 15), "coefficient": 1},
            {"name": "december", "start": datetime(2026, 11, 16), "end": datetime(2026, 11, 30), "coefficient": 1},
            {"name": "december", "start": datetime(2026, 11, 15), "end": datetime(2026, 12, 31), "coefficient": 1}
        ]
        required_base_cols = [
            "MSKU", "品名", "店铺", "记录时间", "日均",
            "FBA库存", "FBA在途", "海外仓可用", "海外仓在途", "本地可用",
            "待检待上架量", "待交付"
        ]
        missing_cols = [col for col in required_base_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Excel文件缺少必要的基础列：{', '.join(missing_cols)}")
            return None

        # 基础数据清洗
        df["记录时间"] = pd.to_datetime(df["记录时间"]).dt.normalize()
        numeric_cols = ["日均", "FBA库存", "FBA在途", "海外仓可用", "海外仓在途",
                        "本地可用", "待检待上架量", "待交付"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # 系数列（暂时都设为1）
        df["10月16-11月15日系数"] = 1
        df["11月16-30日系数"] = 1
        df["12月1-31日系数"] = 1
        df["10月16-11月15日调整后日均"] = (df["日均"] * 1).round(2)
        df["11月16-30日调整后日均"] = (df["日均"] * 1).round(2)
        df["12月1-31日调整后日均"] = (df["日均"] * 1).round(2)

        # 1. FBA+AWD+在途库存
        df["FBA+AWD+在途库存"] = (df["FBA库存"] + df["FBA在途"] + df["海外仓可用"] + df["海外仓在途"]).round().astype(
            int)
        # 2. 全部总库存
        df["全部总库存"] = (
                df["FBA+AWD+在途库存"] + df["本地可用"] + df["待检待上架量"] + df["待交付"]
        ).round().astype(int)

        # 3. 分阶段计算库存耗尽日期
        def calculate_exhaust_date(row, stock_col):
            record_date = row["记录时间"]
            stock = row[stock_col]
            base_avg = row["日均"] if row["日均"] > 0 else 0.1  # 基础日均（避免为0）
            remaining_stock = stock
            current_date = record_date
            # 库存为0返回记录时间
            if remaining_stock <= 0:
                return record_date
            # 阶段1：2026-10-15 前的日均都不打折
            phase1_end = datetime(2026, 10, 15)
            if current_date <= phase1_end:
                days_in_phase = (phase1_end - current_date).days + 1
                sales_possible = base_avg * days_in_phase  # 此阶段无系数调整
                if remaining_stock <= sales_possible:
                    days_needed = remaining_stock / base_avg
                    return current_date + pd.Timedelta(days=days_needed)
                # 库存未耗尽，扣除此阶段销量，进入下一阶段
                remaining_stock -= sales_possible
                current_date = phase1_end + pd.Timedelta(days=1)
            # 阶段2：处理特殊时间段
            for period in TIME_PERIODS:
                if current_date > period["end"] or remaining_stock <= 0:
                    break  # 超出当前时间段或库存已耗尽，跳过
                period_start = max(current_date, period["start"])
                if period_start > period["end"]:
                    continue
                # 计算时间段内的可售天数和调整后日均
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
            # 阶段3：2027-01-01之后（系数=1.0，恢复基础日均）
            if remaining_stock > 0:
                days_needed = remaining_stock / base_avg
                return current_date + pd.Timedelta(days=days_needed)
            return current_date

        # 4. 预计FBA+AWD+在途用完时间
        df["预计FBA+AWD+在途用完时间"] = df.apply(
            lambda row: calculate_exhaust_date(row, "FBA+AWD+在途库存"), axis=1
        )
        # 5. 预计总库存用完时间
        df["预计总库存用完"] = df.apply(
            lambda row: calculate_exhaust_date(row, "全部总库存"), axis=1
        )

        # 6. 分阶段计算滞销库存的核心函数
        def calculate_overstock(row, stock_col):
            record_date = row["记录时间"]
            stock = row[stock_col]
            base_avg = row["日均"] if row["日均"] > 0 else 0.1
            target_date = TARGET_DATE
            remaining_stock = stock
            current_date = record_date
            sold_by_target = 0
            # 若记录日期≥目标日期或库存为0，无滞销
            if current_date >= target_date or remaining_stock <= 0:
                return 0
            # 阶段1：记录日期 → 2026-10-15（系数=1.0）
            phase1_end = datetime(2026, 10, 15)
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
            # 阶段2：处理3个特殊时间段
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
            # 滞销库存 = 总库存 - 目标日期前可售出库存
            return max(0, stock - sold_by_target)

        # 7. 预计用完时间比目标时间多出来的天数
        days_diff = (df["预计总库存用完"] - TARGET_DATE).dt.days
        df["预计用完时间比目标时间多出来的天数"] = np.where(days_diff > 0, days_diff, 0).astype(int)

        # 8. 新增1：区分年份品/非年份品
        df["是否年份品"] = df["品名"].astype(str).str.contains("2026", na=False)

        # 年份品清仓风险函数（优化：处理nan）
        def determine_status(row):
            days = row["预计用完时间比目标时间多出来的天数"]
            is_year_product = row["是否年份品"]

            # 先判断非年份品，避免nan干扰
            if not is_year_product:
                return "非年份品（无目标日期风险）"
            # 处理年份品的nan/异常值
            if pd.isna(days) or days < 0:
                return "健康"
            if days >= 20:
                return "高滞销风险"
            elif days >= 10:
                return "中滞销风险"
            elif days > 0:
                return "低滞销风险"
            else:
                return "健康"

        # 新增4：非年份品隔离逻辑（先执行）
        non_year_mask = df["是否年份品"] == False
        df.loc[non_year_mask, "预计用完时间比目标时间多出来的天数"] = np.nan

        # 生成年份品清仓风险列（关键：先生成，再算环比）
        df["年份品清仓风险"] = df.apply(determine_status, axis=1)

        # 9. 环比上周库年份品滞销风险变化（移到年份品清仓风险之后！）
        df = df.sort_values(["MSKU", "记录时间"])
        df["上周状态"] = df.groupby("MSKU")["年份品清仓风险"].shift(1)
        status_severity = {"健康": 0, "低滞销风险": 1, "中滞销风险": 2, "高滞销风险": 3,
                           "非年份品（无目标日期风险）": 0}

        def compare_status(current, previous):
            if pd.isna(previous):
                return "-"
            if current == previous:
                return "维持不变"
            current_sev = status_severity.get(current, 0)
            prev_sev = status_severity.get(previous, 0)
            return "改善" if current_sev < prev_sev else "恶化"

        df["环比上周库年份品滞销风险变化"] = df.apply(
            lambda row: compare_status(row["年份品清仓风险"], row["上周状态"]), axis=1
        )

        # 10. FBA+AWD+在途滞销数量
        df["FBA+AWD+在途滞销数量"] = df.apply(
            lambda row: calculate_overstock(row, "FBA+AWD+在途库存"), axis=1
        ).round().astype(int)
        # 11. 总滞销库存
        df["总滞销库存"] = df.apply(
            lambda row: calculate_overstock(row, "全部总库存"), axis=1
        ).round().astype(int)
        # 12. 本地滞销数量
        df["本地滞销数量"] = (df["总滞销库存"] - df["FBA+AWD+在途滞销数量"]).round().astype(int)
        df["本地滞销数量"] = np.maximum(df["本地滞销数量"], 0)

        # 13. 预计总库存需要消耗天数
        df["预计总库存需要消耗天数"] = (
                (df["预计总库存用完"] - df["记录时间"]).dt.total_seconds() / (24 * 3600)
        ).round().astype(int)

        # 新增2：库存周转状态判断列
        def judge_inventory_turnover(days):
            if pd.isna(days) or days <= 0:
                return "数据异常"
            elif days <= 100:
                return "库存周转健康"
            elif 100 < days <= 150:
                return "轻度滞销风险"
            elif 150 < days <= 180:
                return "中度滞销风险"
            else:
                return "严重滞销风险"

        df["库存周转状态判断"] = df["预计总库存需要消耗天数"].apply(judge_inventory_turnover)

        # 新增3：100天内达标日均列
        df["总库存周转天数100天内达标日均"] = (df["全部总库存"] / 100).round(2)
        df["总库存周转天数100天内达标日均"] = df["总库存周转天数100天内达标日均"].clip(lower=0).fillna(0)

        # 新增5：周转天数100天内滞销数量（核心逻辑）
        def calculate_turnover_overstock(row):
            total_stock = row["全部总库存"]  # 总库存
            daily_avg = row["日均"] if row["日均"] > 0 else 0.1  # 避免日均为0
            # 100天内可售出的数量
            sales_in_100_days = daily_avg * 100
            # 滞销数量 = 总库存 - 100天可售量（最小为0）
            overstock = max(0, total_stock - sales_in_100_days)
            # 可选：如果只想统计年份品的周转滞销，非年份品置为0/nan
            # if not row["是否年份品"]:
            #     overstock = 0  # 或 np.nan
            return round(overstock)

        # 生成新列
        df["周转天数超过100天的滞销数量"] = df.apply(calculate_turnover_overstock, axis=1).astype(int)
        # 14. 清库存的目标日均
        def calculate_target_sales(row):
            record_date = row["记录时间"]
            base_avg = row["日均"] if row["日均"] > 0 else 0.1
            target_date = TARGET_DATE
            total_sales = 0
            current_date = record_date
            if current_date >= target_date:
                return 0
            # 阶段1：2026-10-15前
            phase1_end = datetime(2026, 10, 15)
            if current_date <= phase1_end:
                actual_end = min(phase1_end, target_date)
                days_in_phase = (actual_end - current_date).days + 1
                sales = base_avg * days_in_phase
                total_sales += sales
                current_date = actual_end + pd.Timedelta(days=1)
            # 阶段2：特殊时间段
            for period in TIME_PERIODS:
                if current_date >= target_date:
                    break
                period_start = max(current_date, period["start"])
                period_end = min(period["end"], target_date)
                if period_start > period_end:
                    continue
                days_in_period = (period_end - period_start).days + 1
                adjusted_avg = base_avg * period["coefficient"]
                sales = adjusted_avg * days_in_period
                total_sales += sales
                current_date = period_end + pd.Timedelta(days=1)
            return total_sales

        days_available = (TARGET_DATE - df["记录时间"]).dt.days
        days_available = np.maximum(days_available, 1)
        df["目标日期前分阶段可售总量"] = df.apply(calculate_target_sales, axis=1)
        df["清库存的目标日均"] = np.where(
            df["年份品清仓风险"] == "健康",
            (df["目标日期前分阶段可售总量"] / days_available).round(2),
            (df["全部总库存"] / days_available).round(2)
        )

        # 15. 预计清完FBA+AWD+在途需要的日均
        def calculate_fba_awd_target_avg(row):
            record_date = row["记录时间"]
            fba_awd_stock = row["FBA+AWD+在途库存"]
            target_date = TARGET_DATE
            days_available = (target_date - record_date).days
            days_available = max(days_available, 1)
            fba_sales_possible = calculate_target_sales(row)
            if fba_awd_stock <= fba_sales_possible:
                return round(fba_sales_possible / days_available, 2)
            else:
                return round(fba_awd_stock / days_available, 2)

        df["预计清完FBA+AWD+在途需要的日均"] = df.apply(calculate_fba_awd_target_avg, axis=1)
        df = df.drop(columns=["目标日期前分阶段可售总量"], errors="ignore")

        # 最终排序
        df = df.sort_values("记录时间", ascending=False).reset_index(drop=True)
        return df

    except Exception as e:
        st.error(f"数据加载失败：{str(e)}")
        import traceback
        st.error(f"详细错误信息：{traceback.format_exc()}")  # 新增：打印详细错误，方便调试
        return None
# ========== 1. 先定义 get_week_data（基础函数） ==========
def get_week_data(df, target_date):
    """获取指定日期的全量数据（年份品+非年份品）"""
    target_date = pd.to_datetime(target_date).normalize()
    week_data = df[df["记录时间"] == target_date].copy()
    # 注意：这里已经删掉了剔除非年份品的行！
    return week_data if not week_data.empty else None

# ========== 2. 再定义 get_week_data_year_product（依赖上面的函数） ==========
def get_week_data_year_product(df, target_date):
    """获取指定日期的年份品数据（给指标/图表用）"""
    week_data = get_week_data(df, target_date)  # 调用上面的基础函数
    if week_data is not None and not week_data.empty:
        week_data = week_data[week_data["是否年份品"] == True].copy()  # 只留年份品
    return week_data

# ========== 3. 最后定义 get_previous_week_data（依赖上面的年份品函数） ==========
def get_previous_week_data(df, current_date):
    """获取上一周的年份品数据（用于环比计算）"""
    current_date = pd.to_datetime(current_date).normalize()
    all_dates = sorted(df["记录时间"].unique())
    if current_date not in all_dates:
        return None
    current_idx = all_dates.index(current_date)
    if current_idx > 0:
        prev_date = all_dates[current_idx - 1]
        return get_week_data_year_product(df, prev_date)  # 调用年份品函数
    return None

def get_previous_week_data(df, current_date):
    """获取上一周数据（用于环比计算）"""
    current_date = pd.to_datetime(current_date).normalize()
    all_dates = sorted(df["记录时间"].unique())
    if current_date not in all_dates:
        return None
    current_idx = all_dates.index(current_date)
    if current_idx > 0:
        prev_date = all_dates[current_idx - 1]
        # ========== 关键修改：用新的年份品函数 ==========
        return get_week_data_year_product(df, prev_date)
    return None

# ========== 新增：获取上周周转数据（全量商品，兼容非年份品） ==========
def get_previous_week_turnover_data(df, current_date):
    """
    修复：获取上一周的全量周转数据（用于周转指标环比）
    优化点：
    1. 按自然周逻辑匹配上周数据（而非仅上一个记录日期）
    2. 统一返回格式（空数据返回空DataFrame而非None）
    3. 补全必要列，避免后续统计报错
    """
    # 标准化日期格式
    current_date = pd.to_datetime(current_date).normalize()
    df["记录时间"] = pd.to_datetime(df["记录时间"]).dt.normalize()

    # 计算当前日期所在自然周的时间范围（周一至周日）
    current_weekday = current_date.weekday()  # 0=周一, 6=周日
    current_week_start = current_date - timedelta(days=current_weekday)
    current_week_end = current_week_start + timedelta(days=6)

    # 计算上一周的时间范围
    prev_week_start = current_week_start - timedelta(days=7)
    prev_week_end = prev_week_start + timedelta(days=6)

    # 筛选上周数据（全量商品，不过滤年份品）
    prev_week_data = df[
        (df["记录时间"] >= prev_week_start) &
        (df["记录时间"] <= prev_week_end)
        ].copy()

    # 兜底：确保必要列存在，避免后续报错
    required_cols = ["MSKU", "店铺", "库存周转状态判断", "周转滞销库存"]
    for col in required_cols:
        if col not in prev_week_data.columns:
            prev_week_data[col] = np.nan

    # 统一返回空DataFrame（而非None），方便后续判断
    return prev_week_data


def compare_turnover_metrics(current_turnover_metrics, prev_turnover_metrics):
    """
    修复：计算周转指标的环比差异
    优化点：
    1. 严谨处理除0/空值/异常值
    2. 区分“新增”（上周0本周有值）和“清零”（上周有值本周0）
    3. 统一数值类型，避免计算报错
    """
    turnover_comparison = {}
    # 处理空值：若上周无数据，默认空字典
    prev_turnover_metrics = prev_turnover_metrics or {}

    for key in current_turnover_metrics:
        # 标准化数值类型，处理空值/异常值
        curr_val = float(current_turnover_metrics[key]) if pd.notna(current_turnover_metrics[key]) else 0.0
        prev_val = float(prev_turnover_metrics.get(key, 0)) if pd.notna(prev_turnover_metrics.get(key, 0)) else 0.0

        # 计算差值和变化率
        diff = curr_val - prev_val

        # 精细化处理变化率：区分除0场景
        if prev_val == 0:
            if curr_val > 0:
                pct = "新增"  # 上周无数据，本周新增
            else:
                pct = 0.0  # 两周均无数据
        elif curr_val == 0:
            pct = -100.0  # 本周清零（100%下降）
        else:
            pct = round((diff / prev_val) * 100, 2)

        turnover_comparison[key] = {
            "current": round(curr_val, 2),  # 保留2位小数，提升可读性
            "last_week": round(prev_val, 2),
            "diff": round(diff, 2),
            "pct": pct
        }
    return turnover_comparison


def calculate_turnover_status_change(current_data, prev_data):
    """
    修复：对比本周/上周的周转状态，统计各状态的改善/不变/恶化数量
    优化点：
    1. 完整统计所有MSKU（新增/流失/存量）
    2. 修正状态变化归因逻辑（累加到来源状态）
    3. 完善异常状态处理
    """
    # 初始化返回结果
    status_change = {
        "库存周转健康": {"改善": 0, "不变": 0, "恶化": 0, "新增": 0, "流失": 0},
        "轻度滞销风险": {"改善": 0, "不变": 0, "恶化": 0, "新增": 0, "流失": 0},
        "中度滞销风险": {"改善": 0, "不变": 0, "恶化": 0, "新增": 0, "流失": 0},
        "严重滞销风险": {"改善": 0, "不变": 0, "恶化": 0, "新增": 0, "流失": 0},
        "数据异常": {"改善": 0, "不变": 0, "恶化": 0, "新增": 0, "流失": 0}
    }

    # 边界处理：无数据时返回初始化结果
    if current_data is None or current_data.empty:
        return status_change
    if prev_data is None or prev_data.empty:
        # 仅本周有数据：所有MSKU标记为“新增”
        current_status_count = current_data["库存周转状态判断"].value_counts()
        for status, count in current_status_count.items():
            if status in status_change:
                status_change[status]["新增"] = count
        return status_change

    # 标准化数据：确保关键列存在且非空
    merge_keys = ["MSKU", "店铺"]
    current_data = current_data[merge_keys + ["库存周转状态判断"]].fillna("数据异常")
    prev_data = prev_data[merge_keys + ["库存周转状态判断"]].fillna("数据异常")

    # 全量合并数据（outer join），覆盖所有MSKU
    compare_df = pd.merge(
        current_data.rename(columns={"库存周转状态判断": "当前状态"}),
        prev_data.rename(columns={"库存周转状态判断": "上周状态"}),
        on=merge_keys,
        how="outer",
        indicator=True
    )

    # 定义周转状态严重程度（数值越大越健康）
    turnover_severity = {
        "数据异常": -1,
        "严重滞销风险": 0,
        "中度滞销风险": 1,
        "轻度滞销风险": 2,
        "库存周转健康": 3
    }

    # 遍历所有MSKU，统计状态变化
    for _, row in compare_df.iterrows():
        curr_status = row["当前状态"] if pd.notna(row["当前状态"]) else "数据异常"
        prev_status = row["上周状态"] if pd.notna(row["上周状态"]) else "数据异常"
        merge_flag = row["_merge"]

        # 跳过不在定义内的异常状态
        if curr_status not in turnover_severity:
            curr_status = "数据异常"
        if prev_status not in turnover_severity:
            prev_status = "数据异常"

        # 分类处理：新增/流失/存量
        if merge_flag == "left_only":  # 本周新增MSKU
            status_change[curr_status]["新增"] += 1
        elif merge_flag == "right_only":  # 本周流失MSKU
            status_change[prev_status]["流失"] += 1
        else:  # 两周都有数据，判断改善/不变/恶化
            curr_sev = turnover_severity[curr_status]
            prev_sev = turnover_severity[prev_status]

            if curr_sev > prev_sev:
                change_type = "改善"
            elif curr_sev < prev_sev:
                change_type = "恶化"
            else:
                change_type = "不变"

            # 关键修复：变化归因到「上周状态」（而非当前状态）
            status_change[prev_status][change_type] += 1

    return status_change
def calculate_status_metrics(data):
    """计算状态分布指标"""
    if data is None or data.empty:
        return {"总MSKU数": 0, "健康": 0, "低滞销风险": 0, "中滞销风险": 0, "高滞销风险": 0}
    total = len(data)
    status_counts = data["年份品清仓风险"].value_counts().to_dict()
    metrics = {"总MSKU数": total}
    # ========== 调整：兼容非年份品（但此函数仅接收年份品数据，防兜底） ==========
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
        # 颜色
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

def render_multi_index_table(data, index_columns, value_columns, page=1, page_size=30, table_id=""):
    # ========== 新增：初始化分页状态（关键修复） ==========
    if f"current_page_{table_id}" not in st.session_state:
        st.session_state[f"current_page_{table_id}"] = 1
    # 覆盖传入的page参数为session_state中的值
    page = st.session_state[f"current_page_{table_id}"]

    if data.empty:
        st.info("没有数据可显示")
        return 0
    total_rows = len(data)
    total_pages = max(1, (total_rows + page_size - 1) // page_size)
    multi_index_data = data.set_index(index_columns)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    paginated_data = multi_index_data.iloc[start_idx:end_idx]
    html = paginated_data.to_html(
        classes=["dataframe", "table", "table-striped", "table-hover"],
        escape=False,
        na_rep="",
        border=0
    )
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
    """状态分布柱状图"""
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
        custom_data=["状态"]
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
    """库存可用天数分布图表"""
    # ========== 新增：兜底剔除非年份品 ==========
    if data is not None and not data.empty:
        data = data[data["是否年份品"] == True].copy()

    if data is None or data.empty:
        fig = go.Figure()
        fig.add_annotation(text="无数据可展示", x=0.5, y=0.5, showarrow=False, font=dict(size=16))
        fig.update_layout(title=title, plot_bgcolor="#f8f9fa", height=400)
        return fig

    valid_days = data["预计总库存需要消耗天数"].clip(lower=0)
    today = data["记录时间"].iloc[0]
    days_to_target = (TARGET_DATE - today).days
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
    for status, threshold in thresholds.items():
        if threshold >= 0:
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
    """店铺状态分布表"""
    # ========== 新增：兜底剔除非年份品 ==========
    if current_data is not None and not current_data.empty:
        current_data = current_data[current_data["是否年份品"] == True].copy()
    if prev_data is not None and not prev_data.empty:
        prev_data = prev_data[prev_data["是否年份品"] == True].copy()

    if current_data is None or current_data.empty:
        st.markdown("<p>无店铺数据可展示</p>", unsafe_allow_html=True)
        return

    # ========== 调整：pivot时过滤非年份品状态（防列错乱） ==========
    current_data_filtered = current_data[
        current_data["年份品清仓风险"].isin(["健康", "低滞销风险", "中滞销风险", "高滞销风险"])]
    current_pivot = pd.pivot_table(
        current_data_filtered,
        index="店铺",
        columns="年份品清仓风险",
        values="MSKU",
        aggfunc="count",
        fill_value=0
    ).reindex(columns=["健康", "低滞销风险", "中滞销风险", "高滞销风险"], fill_value=0)

    prev_pivot = None
    if prev_data is not None and not prev_data.empty:
        prev_data_filtered = prev_data[prev_data["年份品清仓风险"].isin(["健康", "低滞销风险", "中滞销风险", "高滞销风险"])]
        prev_pivot = pd.pivot_table(
            prev_data_filtered,
            index="店铺",
            columns="年份品清仓风险",
            values="MSKU",
            aggfunc="count",
            fill_value=0
        ).reindex(columns=["健康", "低滞销风险", "中滞销风险", "高滞销风险"],
                  fill_value=0)

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
    """产品风险详情表"""
    # ========== 新增：初始化分页状态（关键修复） ==========
    if f"current_page_{table_id}" not in st.session_state:
        st.session_state[f"current_page_{table_id}"] = 1
    # 覆盖传入的page参数为session_state中的值
    page = st.session_state[f"current_page_{table_id}"]

    if data is None or data.empty:
        st.markdown("<p style='color:#666'>无匹配产品数据</p>", unsafe_allow_html=True)
        return 0
    status_order = {
        "高滞销风险": 0,
        "中滞销风险": 1,
        "低滞销风险": 2,
        "健康": 3
    }
    data = data.copy()
    # ========== 修复：原代码用了"状态判断"，实际列名是"年份品清仓风险"（关键！） ==========
    data["_sort_key"] = data["年份品清仓风险"].map(status_order)
    data = data.sort_values(by=["_sort_key", "总滞销库存"], ascending=[True, False])
    data = data.drop(columns=["_sort_key"])
    display_cols = [
        "MSKU", "品名", "店铺",
        "日均", "7天日均", "14天日均", "28天日均",
        "FBA+AWD+在途库存", "本地可用", "全部总库存",
        "预计FBA+AWD+在途用完时间", "预计总库存用完",
        "年份品清仓风险", "预计清完FBA+AWD+在途需要的日均", "清库存的目标日均",
        "FBA+AWD+在途滞销数量", "本地滞销数量", "总滞销库存",
        "预计总库存需要消耗天数", "预计用完时间比目标时间多出来的天数",
        "环比上周库年份品滞销风险变化"  # ========== 修复：列名拼写错误 ==========
    ]
    available_cols = [col for col in display_cols if col in data.columns]
    table_data = data[available_cols].copy()
    total_rows = len(table_data)
    total_pages = ceil(total_rows / page_size)
    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, total_rows)
    paginated_data = table_data.iloc[start_idx:end_idx].copy()
    date_cols = ["预计FBA+AWD+在途用完时间", "预计总库存用完"]
    for col in date_cols:
        if col in paginated_data.columns:
            paginated_data[col] = pd.to_datetime(paginated_data[col]).dt.strftime("%Y-%m-%d")
    # ========== 修复：列名从"状态判断"改为"年份品清仓风险" ==========
    if "年份品清仓风险" in paginated_data.columns:
        paginated_data["年份品清仓风险"] = paginated_data["年份品清仓风险"].apply(
            lambda x: f"<span style='color:{STATUS_COLORS[x]}; font-weight:bold;'>{x}</span>"
        )
    if prev_data is not None and not prev_data.empty:
        compare_cols = [
            "日均", "7天日均", "14天日均", "28天日均",
            "FBA+AWD+在途库存", "本地可用",
            "全部总库存", "FBA+AWD+在途滞销数量", "本地滞销数量", "总滞销库存",
            "预计总库存需要消耗天数", "预计用完时间比目标时间多出来的天数"
        ]
        valid_compare_cols = [col for col in compare_cols if col in prev_data.columns]
        prev_map = prev_data.set_index("MSKU")[valid_compare_cols].to_dict("index")
        def add_compare(row, col):
            msku = row["MSKU"]
            curr_val = row[col]
            prev_val = prev_map.get(msku, {}).get(col, 0)
            if prev_val == 0:
                return f"{curr_val:.2f}<br><span style='color:#666'>无数据</span>"
            diff = curr_val - prev_val
            pct = (diff / prev_val) * 100
            if col in ["日均", "7天日均", "14天日均", "28天日均"]:
                color = "#2E8B57" if diff >= 0 else "#DC143C"
            else:
                color = "#2E8B57" if diff <= 0 else "#DC143C"
            diff_symbol = "+" if diff > 0 else ""
            pct_symbol = "+" if pct > 0 else ""
            return f"{curr_val:.2f}<br><span style='color:{color}'>{diff_symbol}{diff:.2f} ({pct_symbol}{pct:.1f}%)</span>"
        for col in valid_compare_cols:
            if col in paginated_data.columns:
                paginated_data[col] = paginated_data.apply(lambda x: add_compare(x, col), axis=1)
    st.markdown(paginated_data.to_html(escape=False, index=False), unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if page > 1:
            if st.button("上一页", key=f"prev_page_{table_id}"):
                st.session_state[f"current_page_{table_id}"] = page - 1  # ========== 修复：键名带table_id ==========
                st.rerun()
    with col2:
        st.write(f"第 {page} 页，共 {total_pages} 页，共 {total_rows} 条记录")
    with col3:
        if page < total_pages:
            if st.button("下一页", key=f"next_page_{table_id}"):
                st.session_state[f"current_page_{table_id}"] = page + 1  # ========== 修复：键名带table_id ==========
                st.rerun()
    return total_rows


def render_four_week_comparison_table(df, date_list):
    """近四周概览表"""
    if len(date_list) < 1:
        st.markdown("<p>无数据可展示</p>", unsafe_allow_html=True)
        return
    display_dates = date_list[-4:] if len(date_list) >= 4 else date_list
    date_labels = [d.strftime("%Y-%m-%d") for d in display_dates]
    comparison_data = []
    for i, date in enumerate(display_dates):
        data = get_week_data_year_product(df, date)  # 已内置年份品筛选
        metrics = calculate_status_metrics(data)
        if i > 0:
            prev_data = get_week_data_year_product(df, display_dates[i - 1])  # 已内置年份品筛选
            prev_metrics = calculate_status_metrics(prev_data)
            comparisons = compare_with_previous(metrics, prev_metrics)
        else:
            comparisons = None
        row = {
            "日期": date_labels[i],
            "总MSKU数": metrics["总MSKU数"],
            "健康": metrics["健康"],
            "低滞销风险": metrics["低滞销风险"],
            "中滞销风险": metrics["中滞销风险"],
            "高滞销风险": metrics["高滞销风险"]
        }
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
    """四周状态变化趋势"""
    if len(date_list) < 1:
        fig = go.Figure()
        fig.add_annotation(text="无数据可展示", x=0.5, y=0.5, showarrow=False, font=dict(size=16))
        fig.update_layout(title="四周状态变化趋势", plot_bgcolor="#f8f9fa", height=400)
        return fig
    # 获取最多四周数据
    display_dates = date_list[-4:] if len(date_list) >= 4 else date_list
    date_labels = [d.strftime("%Y-%m-%d") for d in display_dates]
    # 准备数据
    trend_data = []
    for date, label in zip(display_dates, date_labels):
        data = get_week_data_year_product(df, date)  # 已内置年份品筛选
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
        title="四周状态变化趋势",
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


# ===================== 新增：全量品库存周转状态四周统计（仅新增，不修改原有代码） =====================
# 补充状态颜色（若你的代码中已定义STATUS_COLORS，可删除这一段）
if 'STATUS_COLORS' not in locals():
    STATUS_COLORS = {
        "库存周转健康": "#2E8B57",  # 全量品-周转健康
        "轻度滞销风险": "#FFD700",  # 全量品-轻度滞销
        "中度滞销风险": "#FF8C00",  # 全量品-中度滞销
        "严重滞销风险": "#DC143C",  # 全量品-严重滞销
        "数据异常": "#808080"  # 全量品-数据异常
    }


def calculate_turnover_metrics(data):
    """计算全量品库存周转状态指标"""
    if data is None or data.empty:
        return {
            "总MSKU数": 0,
            "库存周转健康": 0,
            "轻度滞销风险": 0,
            "中度滞销风险": 0,
            "严重滞销风险": 0,
            "数据异常": 0
        }

    # 统计各周转状态的MSKU数（字段名和你的数据保持一致）
    metrics = {
        "总MSKU数": len(data),
        "库存周转健康": len(data[data["库存周转状态判断"] == "库存周转健康"]),
        "轻度滞销风险": len(data[data["库存周转状态判断"] == "轻度滞销风险"]),
        "中度滞销风险": len(data[data["库存周转状态判断"] == "中度滞销风险"]),
        "严重滞销风险": len(data[data["库存周转状态判断"] == "严重滞销风险"]),
        "数据异常": len(data[data["库存周转状态判断"] == "数据异常"])
    }
    return metrics


def compare_turnover_with_previous(current_metrics, prev_metrics):
    """对比全量品周转状态环比变化"""
    comparisons = {}
    status_list = ["总MSKU数", "库存周转健康", "轻度滞销风险", "中度滞销风险", "严重滞销风险", "数据异常"]

    for status in status_list:
        current_val = current_metrics[status]
        prev_val = prev_metrics[status]
        diff = current_val - prev_val
        pct = (diff / prev_val * 100) if prev_val != 0 else 0.0
        comparisons[status] = {
            "变化值": diff,
            "变化率(%)": round(pct, 2)
        }
    return comparisons


def render_turnover_four_week_comparison_table(df, date_list):
    """近四周概览表（全量品库存周转状态）"""
    if len(date_list) < 1:
        st.markdown("<p>无数据可展示</p>", unsafe_allow_html=True)
        return
    display_dates = date_list[-4:] if len(date_list) >= 4 else date_list
    date_labels = [d.strftime("%Y-%m-%d") for d in display_dates]
    comparison_data = []
    for i, date in enumerate(display_dates):
        # 获取全量品周数据（使用你已定义的get_week_data函数，无年份品筛选）
        data = get_week_data(df, date)
        metrics = calculate_turnover_metrics(data)
        if i > 0:
            prev_data = get_week_data(df, display_dates[i - 1])
            prev_metrics = calculate_turnover_metrics(prev_data)
            comparisons = compare_turnover_with_previous(metrics, prev_metrics)
        else:
            comparisons = None
        row = {
            "日期": date_labels[i],
            "总MSKU数": metrics["总MSKU数"],
            "库存周转健康": metrics["库存周转健康"],
            "轻度滞销风险": metrics["轻度滞销风险"],
            "中度滞销风险": metrics["中度滞销风险"],
            "严重滞销风险": metrics["严重滞销风险"],
            "数据异常": metrics["数据异常"]
        }
        if comparisons:
            row["总MSKU数变化"] = comparisons["总MSKU数"]["变化值"]
            row["库存周转健康变化"] = comparisons["库存周转健康"]["变化值"]
            row["轻度滞销风险变化"] = comparisons["轻度滞销风险"]["变化值"]
            row["中度滞销风险变化"] = comparisons["中度滞销风险"]["变化值"]
            row["严重滞销风险变化"] = comparisons["严重滞销风险"]["变化值"]
            row["数据异常变化"] = comparisons["数据异常"]["变化值"]

        comparison_data.append(row)

    # 构建HTML表格（样式和原有年份品表格一致）
    html = "<table style='width:100%; border-collapse:collapse;'>"
    html += "<tr><th style='border:1px solid #ddd; padding:8px;'>日期</th>"
    html += "<th style='border:1px solid #ddd; padding:8px;'>总MSKU数</th>"
    html += "<th style='border:1px solid #ddd; padding:8px;'>库存周转健康</th>"
    html += "<th style='border:1px solid #ddd; padding:8px;'>轻度滞销风险</th>"
    html += "<th style='border:1px solid #ddd; padding:8px;'>中度滞销风险</th>"
    html += "<th style='border:1px solid #ddd; padding:8px;'>严重滞销风险</th>"
    html += "<th style='border:1px solid #ddd; padding:8px;'>数据异常</th></tr>"

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

        # 库存周转健康
        if "库存周转健康变化" in row:
            diff = row["库存周转健康变化"]
            color = "#2E8B57" if diff >= 0 else "#DC143C"
            symbol = "+" if diff > 0 else ""
            html += f"<td style='border:1px solid #ddd; padding:8px; color:{STATUS_COLORS['库存周转健康']};'>{row['库存周转健康']}<br><span style='color:{color}; font-size:12px;'>{symbol}{diff}</span></td>"
        else:
            html += f"<td style='border:1px solid #ddd; padding:8px; color:{STATUS_COLORS['库存周转健康']};'>{row['库存周转健康']}</td>"

        # 轻度滞销风险
        if "轻度滞销风险变化" in row:
            diff = row["轻度滞销风险变化"]
            color = "#2E8B57" if diff <= 0 else "#DC143C"
            symbol = "+" if diff > 0 else ""
            html += f"<td style='border:1px solid #ddd; padding:8px; color:{STATUS_COLORS['轻度滞销风险']};'>{row['轻度滞销风险']}<br><span style='color:{color}; font-size:12px;'>{symbol}{diff}</span></td>"
        else:
            html += f"<td style='border:1px solid #ddd; padding:8px; color:{STATUS_COLORS['轻度滞销风险']};'>{row['轻度滞销风险']}</td>"

        # 中度滞销风险
        if "中度滞销风险变化" in row:
            diff = row["中度滞销风险变化"]
            color = "#2E8B57" if diff <= 0 else "#DC143C"
            symbol = "+" if diff > 0 else ""
            html += f"<td style='border:1px solid #ddd; padding:8px; color:{STATUS_COLORS['中度滞销风险']};'>{row['中度滞销风险']}<br><span style='color:{color}; font-size:12px;'>{symbol}{diff}</span></td>"
        else:
            html += f"<td style='border:1px solid #ddd; padding:8px; color:{STATUS_COLORS['中度滞销风险']};'>{row['中度滞销风险']}</td>"

        # 严重滞销风险
        if "严重滞销风险变化" in row:
            diff = row["严重滞销风险变化"]
            color = "#2E8B57" if diff <= 0 else "#DC143C"
            symbol = "+" if diff > 0 else ""
            html += f"<td style='border:1px solid #ddd; padding:8px; color:{STATUS_COLORS['严重滞销风险']};'>{row['严重滞销风险']}<br><span style='color:{color}; font-size:12px;'>{symbol}{diff}</span></td>"
        else:
            html += f"<td style='border:1px solid #ddd; padding:8px; color:{STATUS_COLORS['严重滞销风险']};'>{row['严重滞销风险']}</td>"

        # 数据异常
        if "数据异常变化" in row:
            diff = row["数据异常变化"]
            color = "#DC143C" if diff >= 0 else "#2E8B57"
            symbol = "+" if diff > 0 else ""
            html += f"<td style='border:1px solid #ddd; padding:8px; color:{STATUS_COLORS['数据异常']};'>{row['数据异常']}<br><span style='color:{color}; font-size:12px;'>{symbol}{diff}</span></td>"
        else:
            html += f"<td style='border:1px solid #ddd; padding:8px; color:{STATUS_COLORS['数据异常']};'>{row['数据异常']}</td>"

        html += "</tr>"
    html += "</table>"
    st.markdown(html, unsafe_allow_html=True)


def render_turnover_four_week_status_chart(df, date_list):
    """四周状态变化趋势（全量品库存周转状态）"""
    if len(date_list) < 1:
        fig = go.Figure()
        fig.add_annotation(text="无数据可展示", x=0.5, y=0.5, showarrow=False, font=dict(size=16))
        fig.update_layout(title="全量品 - 四周库存周转状态变化趋势", plot_bgcolor="#f8f9fa", height=400)
        st.plotly_chart(fig, use_container_width=True)
        return

    # 获取最多四周数据
    display_dates = date_list[-4:] if len(date_list) >= 4 else date_list
    date_labels = [d.strftime("%Y-%m-%d") for d in display_dates]

    # 准备数据
    trend_data = []
    for date, label in zip(display_dates, date_labels):
        # 获取全量品周数据
        data = get_week_data(df, date)
        metrics = calculate_turnover_metrics(data)

        for status in ["库存周转健康", "轻度滞销风险", "中度滞销风险", "严重滞销风险", "数据异常"]:
            trend_data.append({
                "日期": label,
                "状态": status,
                "MSKU数": metrics[status]
            })

    trend_df = pd.DataFrame(trend_data)

    # 创建柱状图（样式和原有年份品图表一致）
    fig = px.bar(
        trend_df,
        x="状态",
        y="MSKU数",
        color="日期",
        barmode="group",
        title="全量品 - 四周库存周转状态变化趋势",
        text="MSKU数",
        height=400
    )
    fig.update_traces(
        textposition="outside",
        textfont=dict(size=12)
    )
    fig.update_layout(
        xaxis_title="库存周转状态",
        yaxis_title="MSKU数量",
        plot_bgcolor="#f8f9fa",
        margin=dict(t=50, b=20, l=20, r=20)
    )
    st.plotly_chart(fig, use_container_width=True)

def render_store_trend_charts(df, date_list):
    """每个店铺的状态趋势折线图"""
    if len(date_list) < 1:
        st.markdown("<p>无数据可展示</p>", unsafe_allow_html=True)
        return

    # ========== 新增1：兜底筛选年份品 + 空值兼容 ==========
    week_datas = [get_week_data_year_product(df, date) for date in date_list]
    week_datas = [d for d in week_datas if d is not None and not d.empty]
    if not week_datas:
        st.markdown("<p>无店铺数据可展示</p>", unsafe_allow_html=True)
        return
    all_data = pd.concat(week_datas)

    # ========== 新增2：过滤非年份品状态（防店铺列表包含非年份品） ==========
    all_data = all_data[all_data["年份品清仓风险"].isin(["健康", "低滞销风险", "中滞销风险", "高滞销风险"])]
    if all_data.empty:
        st.markdown("<p>无店铺数据可展示</p>", unsafe_allow_html=True)
        return

    stores = sorted(all_data["店铺"].unique())
    date_labels = [d.strftime("%Y-%m-%d") for d in date_list]
    # 分两列显示
    cols = st.columns(2)
    for i, store in enumerate(stores):
        # 准备店铺数据
        store_data = []
        for date, label in zip(date_list, date_labels):
            data = get_week_data_year_product(df, date)  # 已内置年份品筛选
            if data is not None and not data.empty:
                store_status_data = data[data["店铺"] == store]
                # ========== 新增3：过滤店铺数据的非年份品状态 ==========
                store_status_data = store_status_data[
                    store_status_data["年份品清仓风险"].isin(["健康", "低滞销风险", "中滞销风险", "高滞销风险"])]
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
        # 折线图
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
# ===================== 新增：全量品 - 每个店铺的库存周转状态趋势折线图 =====================
def render_turnover_store_trend_charts(df, date_list):
    """每个店铺的库存周转状态趋势折线图（全量品）"""
    if len(date_list) < 1:
        st.markdown("<p>无数据可展示</p>", unsafe_allow_html=True)
        return

    # 1. 兜底筛选全量品 + 空值兼容（使用get_week_data，无年份品筛选）
    week_datas = [get_week_data(df, date) for date in date_list]
    week_datas = [d for d in week_datas if d is not None and not d.empty]
    if not week_datas:
        st.markdown("<p>无店铺数据可展示</p>", unsafe_allow_html=True)
        return
    all_data = pd.concat(week_datas)

    # 2. 过滤非周转状态（防店铺列表包含异常状态）
    turnover_status_list = ["库存周转健康", "轻度滞销风险", "中度滞销风险", "严重滞销风险", "数据异常"]
    all_data = all_data[all_data["库存周转状态判断"].isin(turnover_status_list)]
    if all_data.empty:
        st.markdown("<p>无店铺数据可展示</p>", unsafe_allow_html=True)
        return

    # ========== 新增：明确配置各状态的颜色（按要求指定） ==========
    turnover_status_colors = {
        "库存周转健康": "#2E8B57",    # 绿色（健康）
        "轻度滞销风险": "#FFD700",    # 黄色（轻度）
        "中度滞销风险": "#FF8C00",    # 橙色（中度）
        "严重滞销风险": "#DC143C",    # 红色（严重）
        "数据异常": "#808080"         # 灰色（数据异常，补充）
    }

    stores = sorted(all_data["店铺"].unique())
    date_labels = [d.strftime("%Y-%m-%d") for d in date_list]
    # 分两列显示（和年份品样式一致）
    cols = st.columns(2)
    for i, store in enumerate(stores):
        # 准备店铺数据
        store_data = []
        for date, label in zip(date_list, date_labels):
            data = get_week_data(df, date)  # 全量品数据，无年份品筛选
            if data is not None and not data.empty:
                store_status_data = data[data["店铺"] == store]
                # 过滤店铺数据的非周转状态
                store_status_data = store_status_data[
                    store_status_data["库存周转状态判断"].isin(turnover_status_list)]
                # 计算全量品周转状态指标
                metrics = calculate_turnover_metrics(store_status_data)
                for status in turnover_status_list:
                    store_data.append({
                        "日期": label,
                        "状态": status,
                        "MSKU数": metrics[status]
                    })
        if not store_data:
            continue
        store_df = pd.DataFrame(store_data)
        # 折线图（样式和年份品保持一致，指定明确颜色）
        fig = go.Figure()
        for status in turnover_status_list:
            status_data = store_df[store_df["状态"] == status]
            fig.add_trace(go.Scatter(
                x=status_data["日期"],
                y=status_data["MSKU数"],
                mode="lines+markers",
                name=status,
                # ========== 关键修改：使用指定的颜色配置 ==========
                line=dict(color=turnover_status_colors[status], width=2),
                marker=dict(size=8, color=turnover_status_colors[status]),  # 标记点也用对应颜色
                hovertemplate="日期: %{x}<br>MSKU数: %{y}<br>状态: " + status  # 优化hover提示
            ))
        fig.update_layout(
            title=f"{store} 库存周转状态变化趋势",
            xaxis_title="日期",
            yaxis_title="MSKU数量",
            plot_bgcolor="#f8f9fa",
            height=300,
            margin=dict(t=50, b=20, l=20, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)  # 图例居下，避免遮挡
        )
        # 在对应列显示图表
        with cols[i % 2]:
            st.plotly_chart(fig, use_container_width=True)


def render_store_weekly_changes(df, date_list):
    """店铺每周变化情况表"""
    if len(date_list) < 1:
        st.markdown("<p>无数据可展示</p>", unsafe_allow_html=True)
        return

    # ========== 新增1：兜底筛选年份品 + 空值兼容 ==========
    week_datas = [get_week_data_year_product(df, date) for date in date_list]
    week_datas = [d for d in week_datas if d is not None and not d.empty]
    if not week_datas:
        st.markdown("<p>无店铺数据可展示</p>", unsafe_allow_html=True)
        return
    all_data = pd.concat(week_datas)

    # ========== 新增2：过滤非年份品状态（防店铺列表异常） ==========
    all_data = all_data[all_data["年份品清仓风险"].isin(["健康", "低滞销风险", "中滞销风险", "高滞销风险"])]
    if all_data.empty:
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
            data = get_week_data_year_product(df, date)  # 已内置年份品筛选
            if data is not None and not data.empty:
                store_status_data = data[data["店铺"] == store]
                # ========== 新增3：过滤店铺数据的非年份品状态 ==========
                store_status_data = store_status_data[
                    store_status_data["年份品清仓风险"].isin(["健康", "低滞销风险", "中滞销风险", "高滞销风险"])]
                metrics = calculate_status_metrics(store_status_data)

                # 获取上周数据
                prev_metrics = None
                if i > 0:
                    prev_data = get_week_data_year_product(df, date_list[i - 1])
                    if prev_data is not None and not prev_data.empty:
                        prev_store_data = prev_data[prev_data["店铺"] == store]
                        prev_store_data = prev_store_data[
                            prev_store_data["年份品清仓风险"].isin(["健康", "低滞销风险", "中滞销风险", "高滞销风险"])]
                        prev_metrics = calculate_status_metrics(prev_store_data)

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

# ===================== 新增：全量品 - 店铺每周库存周转状态变化情况表 =====================
def render_turnover_store_weekly_changes(df, date_list):
    """店铺每周库存周转状态变化情况表（全量品）"""
    if len(date_list) < 1:
        st.markdown("<p>无数据可展示</p>", unsafe_allow_html=True)
        return

    # 1. 兜底筛选全量品 + 空值兼容（使用get_week_data，无年份品筛选）
    week_datas = [get_week_data(df, date) for date in date_list]
    week_datas = [d for d in week_datas if d is not None and not d.empty]
    if not week_datas:
        st.markdown("<p>无店铺数据可展示</p>", unsafe_allow_html=True)
        return
    all_data = pd.concat(week_datas)

    # 2. 过滤非周转状态（防店铺列表异常）
    turnover_status_list = ["库存周转健康", "轻度滞销风险", "中度滞销风险", "严重滞销风险", "数据异常"]
    all_data = all_data[all_data["库存周转状态判断"].isin(turnover_status_list)]
    if all_data.empty:
        st.markdown("<p>无店铺数据可展示</p>", unsafe_allow_html=True)
        return

    stores = sorted(all_data["店铺"].unique())
    date_labels = [d.strftime("%Y-%m-%d") for d in date_list]

    # 创建HTML表格（样式和年份品保持一致，适配全量品状态）
    html = "<table style='width:100%; border-collapse:collapse;'>"
    html += "<tr><th style='border:1px solid #ddd; padding:8px;'>店铺</th>"
    html += "<th style='border:1px solid #ddd; padding:8px;'>日期</th>"
    html += "<th style='border:1px solid #ddd; padding:8px;'>总MSKU数</th>"
    html += "<th style='border:1px solid #ddd; padding:8px; background-color:#2E8B5720;'>库存周转健康</th>"
    html += "<th style='border:1px solid #ddd; padding:8px; background-color:#FFD70020;'>轻度滞销风险</th>"
    html += "<th style='border:1px solid #ddd; padding:8px; background-color:#FF8C0020;'>中度滞销风险</th>"
    html += "<th style='border:1px solid #ddd; padding:8px; background-color:#DC143C20;'>严重滞销风险</th>"
    html += "<th style='border:1px solid #ddd; padding:8px; background-color:#80808020;'>数据异常</th></tr>"

    for store in stores:
        for i, (date, label) in enumerate(zip(date_list, date_labels)):
            data = get_week_data(df, date)  # 全量品数据，无年份品筛选
            if data is not None and not data.empty:
                store_status_data = data[data["店铺"] == store]
                # 过滤店铺数据的非周转状态
                store_status_data = store_status_data[
                    store_status_data["库存周转状态判断"].isin(turnover_status_list)]
                metrics = calculate_turnover_metrics(store_status_data)

                # 获取上周数据
                prev_metrics = None
                if i > 0:
                    prev_data = get_week_data(df, date_list[i - 1])
                    if prev_data is not None and not prev_data.empty:
                        prev_store_data = prev_data[prev_data["店铺"] == store]
                        prev_store_data = prev_store_data[
                            prev_store_data["库存周转状态判断"].isin(turnover_status_list)]
                        prev_metrics = calculate_turnover_metrics(prev_store_data)

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

                # 库存周转健康
                if prev_metrics:
                    diff = metrics["库存周转健康"] - prev_metrics["库存周转健康"]
                    color = "#2E8B57" if diff >= 0 else "#DC143C"
                    symbol = "+" if diff > 0 else ""
                    html += f"<td style='border:1px solid #ddd; padding:8px;'>{metrics['库存周转健康']}<br><span style='color:{color}; font-size:12px;'>{symbol}{diff}</span></td>"
                else:
                    html += f"<td style='border:1px solid #ddd; padding:8px;'>{metrics['库存周转健康']}</td>"

                # 轻度滞销风险
                if prev_metrics:
                    diff = metrics["轻度滞销风险"] - prev_metrics["轻度滞销风险"]
                    color = "#2E8B57" if diff <= 0 else "#DC143C"
                    symbol = "+" if diff > 0 else ""
                    html += f"<td style='border:1px solid #ddd; padding:8px;'>{metrics['轻度滞销风险']}<br><span style='color:{color}; font-size:12px;'>{symbol}{diff}</span></td>"
                else:
                    html += f"<td style='border:1px solid #ddd; padding:8px;'>{metrics['轻度滞销风险']}</td>"

                # 中度滞销风险
                if prev_metrics:
                    diff = metrics["中度滞销风险"] - prev_metrics["中度滞销风险"]
                    color = "#2E8B57" if diff <= 0 else "#DC143C"
                    symbol = "+" if diff > 0 else ""
                    html += f"<td style='border:1px solid #ddd; padding:8px;'>{metrics['中度滞销风险']}<br><span style='color:{color}; font-size:12px;'>{symbol}{diff}</span></td>"
                else:
                    html += f"<td style='border:1px solid #ddd; padding:8px;'>{metrics['中度滞销风险']}</td>"

                # 严重滞销风险
                if prev_metrics:
                    diff = metrics["严重滞销风险"] - prev_metrics["严重滞销风险"]
                    color = "#2E8B57" if diff <= 0 else "#DC143C"
                    symbol = "+" if diff > 0 else ""
                    html += f"<td style='border:1px solid #ddd; padding:8px;'>{metrics['严重滞销风险']}<br><span style='color:{color}; font-size:12px;'>{symbol}{diff}</span></td>"
                else:
                    html += f"<td style='border:1px solid #ddd; padding:8px;'>{metrics['严重滞销风险']}</td>"

                # 数据异常
                if prev_metrics:
                    diff = metrics["数据异常"] - prev_metrics["数据异常"]
                    color = "#DC143C" if diff >= 0 else "#2E8B57"  # 数据异常越多越危险
                    symbol = "+" if diff > 0 else ""
                    html += f"<td style='border:1px solid #ddd; padding:8px;'>{metrics['数据异常']}<br><span style='color:{color}; font-size:12px;'>{symbol}{diff}</span></td>"
                else:
                    html += f"<td style='border:1px solid #ddd; padding:8px;'>{metrics['数据异常']}</td>"

                html += "</tr>"

    html += "</table>"
    st.markdown(html, unsafe_allow_html=True)

def render_status_change_table(data, page=1, page_size=30):
    """环比上周库年份品滞销风险变化表"""
    if data is None or data.empty:
        st.markdown("<p style='color:#666'>无数据可展示</p>", unsafe_allow_html=True)
        return 0

    # ========== 调整1：新增周转相关列到展示列表 ==========
    display_cols = [
        "MSKU", "品名", "店铺", "是否年份品", "记录时间",  # 新增：是否年份品
        "日均", "7天日均", "14天日均", "28天日均",
        "FBA+AWD+在途库存", "本地可用", "全部总库存", "预计FBA+AWD+在途用完时间", "预计总库存用完",
        "库存周转状态判断","总库存周转天数100天内达标日均",  "周转天数超过100天的滞销数量","年份品清仓风险",  # 新增：库存周转状态判断
        "预计清完FBA+AWD+在途需要的日均", "清库存的目标日均",  # 新增：100天达标日均
        "FBA+AWD+在途滞销数量", "本地滞销数量", "总滞销库存",
        "预计总库存需要消耗天数", "预计用完时间比目标时间多出来的天数", "环比上周库年份品滞销风险变化"
    ]

    available_cols = [col for col in display_cols if col in data.columns]
    table_data = data[available_cols].copy()
    total_rows = len(table_data)
    total_pages = ceil(total_rows / page_size)
    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, total_rows)
    paginated_data = table_data.iloc[start_idx:end_idx].copy()

    # 日期列格式化
    date_cols = ["记录时间", "预计FBA+AWD+在途用完时间", "预计总库存用完"]
    for col in date_cols:
        if col in paginated_data.columns:
            paginated_data[col] = pd.to_datetime(paginated_data[col]).dt.strftime("%Y-%m-%d")

    # ========== 调整2：年份品清仓风险列样式兼容非年份品 ==========
    if "年份品清仓风险" in paginated_data.columns:
        def format_status(x):
            if x == "非年份品（无目标日期风险）":
                return f"<span style='color:#808080; font-weight:bold;'>{x}</span>"
            return f"<span style='color:{STATUS_COLORS.get(x, '#000000')}; font-weight:bold;'>{x}</span>"

        paginated_data["年份品清仓风险"] = paginated_data["年份品清仓风险"].apply(format_status)

    # ========== 新增3：库存周转状态判断列加颜色样式 ==========
    if "库存周转状态判断" in paginated_data.columns:
        paginated_data["库存周转状态判断"] = paginated_data["库存周转状态判断"].apply(
            lambda x: f"<span style='color:{TURNOVER_STATUS_COLORS.get(x, '#000000')}; font-weight:bold;'>{x}</span>"
        )

    # 环比变化列样式（原有逻辑保留）
    if "环比上周库年份品滞销风险变化" in paginated_data.columns:
        def color_status_change(x):
            if x == "改善":
                return f"<span style='color:#2E8B57; font-weight:bold;'>{x}</span>"
            elif x == "恶化":
                return f"<span style='color:#DC143C; font-weight:bold;'>{x}</span>"
            else:  # 维持不变
                return f"<span style='color:#000000; font-weight:bold;'>{x}</span>"

        paginated_data["环比上周库年份品滞销风险变化"] = paginated_data["环比上周库年份品滞销风险变化"].apply(
            color_status_change)

    # 渲染表格
    st.markdown(paginated_data.to_html(escape=False, index=False), unsafe_allow_html=True)

    # ========== 调整4：修复分页session_state键名冲突 ==========
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if page > 1:
            if st.button("上一页", key="prev_page_status"):
                st.session_state["current_status_page"] = page - 1  # 加引号更规范
                st.rerun()
    with col2:
        st.write(f"第 {page} 页，共 {total_pages} 页，共 {total_rows} 条记录")
    with col3:
        if page < total_pages:
            if st.button("下一页", key="next_page_status"):
                st.session_state["current_status_page"] = page + 1  # 加引号更规范
                st.rerun()

    return total_rows


def render_risk_summary_table(summary_df):
    st.subheader("年份品清仓风险状态汇总表")
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
    .neutral-status {
        color: #808080;  /* 灰色：非年份品状态 */
    }
    </style>
    """, unsafe_allow_html=True)
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
            if col == "年份品清仓风险":
                # ========== 调整1：兼容非年份品状态样式 ==========
                if value == "非年份品（无目标日期风险）":
                    html += f"<td class='neutral-status' style='font-weight:bold;'>{value}</td>"
                else:
                    color = STATUS_COLORS.get(value, "#000000")
                    html += f"<td style='color:{color}; font-weight:bold;'>{value}</td>"
            elif "环比变化" in col:
                if '(' in str(value):
                    change_val = float(value.split()[0])
                    status = row["年份品清仓风险"]
                    # 非年份品不区分正负，直接展示
                    if status == "非年份品（无目标日期风险）":
                        html += f"<td>{value}</td>"
                    else:
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
    # ========== 调整1：新增非年份品状态到统计列表 ==========
    statuses = [
        "健康",
        "低滞销风险",
        "中滞销风险",
        "高滞销风险",
        "非年份品（无目标日期风险）",  # 新增：非年份品统计
        "低滞销风险+中滞销风险+高滞销风险",
        "中滞销风险+高滞销风险"
    ]
    status_mappings = {
        "健康": ["健康"],
        "低滞销风险": ["低滞销风险"],
        "中滞销风险": ["中滞销风险"],
        "高滞销风险": ["高滞销风险"],
        "非年份品（无目标日期风险）": ["非年份品（无目标日期风险）"],  # 新增：非年份品映射
        "低滞销风险+中滞销风险+高滞销风险": ["低滞销风险", "中滞销风险", "高滞销风险"],
        "中滞销风险+高滞销风险": ["中滞销风险", "高滞销风险"]
    }

    # ========== 调整2：总MSKU/库存统计包含非年份品 ==========
    total_current_msku = current_data['MSKU'].nunique() if current_data is not None and not current_data.empty else 0
    total_current_inventory = current_data[
        '总滞销库存'].sum() if current_data is not None and not current_data.empty else 0
    summary_data = []

    for status in statuses:
        original_statuses = status_mappings[status]
        # 过滤当前数据（兼容空值）
        current_filtered = current_data[current_data['年份品清仓风险'].isin(original_statuses)] if (
                current_data is not None and not current_data.empty) else pd.DataFrame()
        current_msku = current_filtered['MSKU'].nunique() if not current_filtered.empty else 0
        current_inventory = current_filtered['总滞销库存'].sum() if not current_filtered.empty else 0

        # 过滤历史数据（兼容空值）
        if previous_data is not None and not previous_data.empty:
            prev_filtered = previous_data[previous_data['年份品清仓风险'].isin(original_statuses)]
            prev_msku = prev_filtered['MSKU'].nunique() if not prev_filtered.empty else 0
            prev_inventory = prev_filtered['总滞销库存'].sum() if not prev_filtered.empty else 0
        else:
            prev_msku = 0
            prev_inventory = 0

        # 计算环比（兼容除数为0）
        msku_change = current_msku - prev_msku
        msku_change_pct = (msku_change / prev_msku * 100) if prev_msku != 0 else 0
        inventory_change = current_inventory - prev_inventory
        inventory_change_pct = (inventory_change / prev_inventory * 100) if prev_inventory != 0 else 0

        # 计算占比（兼容除数为0）
        msku_ratio = (current_msku / total_current_msku * 100) if total_current_msku != 0 else 0
        inventory_ratio = (current_inventory / total_current_inventory * 100) if total_current_inventory != 0 else 0

        summary_data.append({
            "年份品清仓风险": status,
            "MSKU数": current_msku,
            "MSKU占比": f"{msku_ratio:.1f}%",
            "MSKU环比变化": f"{msku_change} ({msku_change_pct:.1f}%)",
            "总滞销库存数": round(current_inventory, 2),
            "总滞销库存占比": f"{inventory_ratio:.1f}%",
            "库存环比变化": f"{round(inventory_change, 2)} ({inventory_change_pct:.1f}%)"
        })
    return pd.DataFrame(summary_data)


def render_stock_forecast_chart(data, msku):
    """单个MSKU的库存预测图表"""
    if data is None or data.empty:
        fig = go.Figure()
        fig.add_annotation(text="无数据可展示", x=0.5, y=0.5, showarrow=False, font=dict(size=16))
        fig.update_layout(title=f"{msku} 库存预测", plot_bgcolor="#f8f9fa", height=400)
        return fig

    row = data.iloc[0]
    # ========== 调整1：兼容非日期类型的记录时间 ==========
    start_date = pd.to_datetime(row["记录时间"]) if not pd.isna(row["记录时间"]) else pd.Timestamp.now()
    # 确保END_DATE/TARGET_DATE是datetime类型（避免未定义报错）
    if 'END_DATE' not in globals():
        END_DATE = start_date + pd.Timedelta(days=365)  # 兜底：默认预测1年
    if 'TARGET_DATE' not in globals():
        TARGET_DATE = start_date + pd.Timedelta(days=90)  # 兜底：默认目标90天

    # ========== 调整2：优化日均销量兜底逻辑 ==========
    base_avg = row["日均"] if (row["日均"] > 0 and not pd.isna(row["日均"])) else 0.1
    total_stock = row["全部总库存"] if (row["全部总库存"] >= 0 and not pd.isna(row["全部总库存"])) else 0
    remaining_stock = total_stock

    # 分阶段系数配置（兼容非datetime类型）
    TIME_PERIODS = [
        {"start": pd.to_datetime("2026-10-16"), "end": pd.to_datetime("2026-11-15"), "coefficient": 1},
        {"start": pd.to_datetime("2026-11-16"), "end": pd.to_datetime("2026-11-30"), "coefficient": 1},
        {"start": pd.to_datetime("2026-12-01"), "end": pd.to_datetime("2026-12-31"), "coefficient": 1}
    ]

    forecast_dates = []
    forecast_stock = []
    current_date = start_date

    # ========== 调整3：限制循环次数，避免无限循环 ==========
    max_days = 730  # 最多预测2年
    day_count = 0

    while current_date <= END_DATE and remaining_stock > 0 and day_count < max_days:
        current_coeff = 1.0
        # 匹配当前日期的系数
        for period in TIME_PERIODS:
            if period["start"] <= current_date <= period["end"]:
                current_coeff = period["coefficient"]
                break
        # 计算当日销量并更新库存
        daily_sales = base_avg * current_coeff
        remaining_stock = max(remaining_stock - daily_sales, 0)
        # 记录数据
        forecast_dates.append(current_date)
        forecast_stock.append(remaining_stock)
        # 日期递增
        current_date += pd.Timedelta(days=1)
        day_count += 1

    # 生成折线图
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_stock,
        mode="lines+markers",
        line=dict(color="#4169E1", width=2),
        name="预计库存（分阶段系数）"
    ))

    # 添加目标日期线（兼容timestamp转换）
    fig.add_vline(
        x=TARGET_DATE.timestamp() * 1000,
        line_dash="dash",
        line_color="#DC143C",  # 红色
        annotation_text="目标消耗日期",
        annotation_position="top right",
        annotation_font=dict(color="#DC143C")
    )

    # 添加时间段系数标注（避免标注超出图表范围）
    max_stock = max(forecast_stock) if forecast_stock else 100
    for period in TIME_PERIODS:
        fig.add_annotation(
            x=period["start"],
            y=max_stock * 0.9,
            text=f"{period['start'].strftime('%m-%d')}起系数: {period['coefficient']}",
            showarrow=True,
            arrowhead=1,
            arrowcolor=STATUS_COLORS.get("低滞销风险", "#4169E1"),
            font=dict(size=10, color=STATUS_COLORS.get("低滞销风险", "#4169E1"))
        )

    # 图表布局优化
    fig.update_layout(
        title=f"{msku} 库存消耗预测（含分阶段销量系数）",
        xaxis_title="日期",
        yaxis_title="剩余库存",
        plot_bgcolor="#f8f9fa",
        height=400,
        margin=dict(t=50, b=20, l=20, r=20)
    )

    # 横坐标设置（优化tick显示）
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
        dtick="M1",  # 按月显示刻度，避免过于密集
        ticklabelmode="period"
    )

    return fig


import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

# 确保全局变量兜底（如果未定义）
if 'END_DATE' not in globals():
    END_DATE = datetime.now() + timedelta(days=365)  # 默认预测1年
if 'TARGET_DATE' not in globals():
    TARGET_DATE = datetime.now() + timedelta(days=90)  # 默认目标90天
if 'STATUS_COLORS' not in globals():
    STATUS_COLORS = {
        "健康": "#28a745",
        "低滞销风险": "#4169E1",
        "中滞销风险": "#FFD700",
        "高滞销风险": "#DC143C",
        "非年份品（无目标日期风险）": "#808080"
    }


def render_product_detail_chart(df, msku):
    """单个产品的历史库存预测对比图"""
    if df is None or df.empty:
        fig = go.Figure()
        fig.add_annotation(text="无数据可展示", x=0.5, y=0.5, showarrow=False, font=dict(size=16))
        fig.update_layout(title=f"{msku} 历史库存预测", plot_bgcolor="#f8f9fa", height=400)
        return fig

    # ========== 调整1：兼容MSKU筛选和空值 ==========
    # 确保MSKU列存在且为字符串类型
    if "MSKU" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="数据缺少MSKU列", x=0.5, y=0.5, showarrow=False, font=dict(size=16))
        fig.update_layout(title=f"{msku} 历史库存预测", plot_bgcolor="#f8f9fa", height=400)
        return fig

    df["MSKU"] = df["MSKU"].astype(str)
    product_data = df[df["MSKU"] == str(msku)].sort_values("记录时间")

    if product_data.empty:
        fig = go.Figure()
        fig.add_annotation(text="无此产品数据", x=0.5, y=0.5, showarrow=False, font=dict(size=16))
        fig.update_layout(title=f"{msku} 历史库存预测", plot_bgcolor="#f8f9fa", height=400)
        return fig

    # ========== 调整2：标准化时间段系数（兼容datetime） ==========
    TIME_PERIODS = [
        {"start": pd.to_datetime("2026-10-16"), "end": pd.to_datetime("2026-11-15"), "coefficient": 1},
        {"start": pd.to_datetime("2026-11-16"), "end": pd.to_datetime("2026-11-30"), "coefficient": 1},
        {"start": pd.to_datetime("2026-12-01"), "end": pd.to_datetime("2026-12-31"), "coefficient": 1}
    ]

    # 创建图表
    fig = go.Figure()

    # ========== 调整3：限制循环次数+数据兜底，防无限循环/崩溃 ==========
    max_days = 730  # 最多预测2年
    color_palette = ["#4169E1", "#28a745", "#FFD700", "#DC143C", "#808080", "#9370DB"]  # 多线条配色
    color_idx = 0

    for _, row in product_data.iterrows():
        # 记录时间兼容（转换为datetime）
        record_date = pd.to_datetime(row["记录时间"]) if not pd.isna(row["记录时间"]) else pd.Timestamp.now()
        label = record_date.strftime("%Y-%m-%d")

        # 日均销量兜底（避免0/负数/空值）
        base_avg = row["日均"] if (row["日均"] > 0 and not pd.isna(row["日均"])) else 0.1

        # 总库存兜底（避免负数/空值）
        total_stock = row["全部总库存"] if (row["全部总库存"] >= 0 and not pd.isna(row["全部总库存"])) else 0
        remaining_stock = total_stock

        end_date = pd.to_datetime(END_DATE)  # 确保end_date是datetime类型

        forecast_dates = []
        forecast_stock = []
        current_date = record_date
        day_count = 0

        # 带次数限制的循环
        while current_date <= end_date and remaining_stock > 0 and day_count < max_days:
            current_coeff = 1.0
            # 匹配当前日期的系数
            for period in TIME_PERIODS:
                if period["start"] <= current_date <= period["end"]:
                    current_coeff = period["coefficient"]
                    break

            daily_sales = base_avg * current_coeff
            remaining_stock = max(remaining_stock - daily_sales, 0)

            forecast_dates.append(current_date)
            forecast_stock.append(remaining_stock)

            current_date += timedelta(days=1)
            day_count += 1

        # 添加预测线（循环使用配色）
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast_stock,
            mode="lines",
            name=f"{label}（记录）",
            line=dict(width=2, color=color_palette[color_idx % len(color_palette)])
        ))
        color_idx += 1

    # 添加目标日期线（兼容timestamp转换）
    fig.add_vline(
        x=pd.to_datetime(TARGET_DATE).timestamp() * 1000,
        line_dash="dash",
        line_color="#DC143C",
        annotation_text="目标消耗日期",
        annotation_position="top right",
        annotation_font=dict(color="#DC143C")
    )

    # 图表布局优化
    fig.update_layout(
        title=f"{msku} 不同记录时间的库存预测对比（含分阶段系数）",
        xaxis_title="日期",
        yaxis_title="剩余库存",
        plot_bgcolor="#f8f9fa",
        height=400,
        margin=dict(t=50, b=20, l=20, r=20),
        legend_title="记录时间",
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)  # 图例下移，避免遮挡
    )

    # 横坐标优化（按月显示刻度，避免密集）
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
        dtick="M1",  # 按月显示刻度
        ticklabelmode="period"
    )

    return fig


def render_stock_forecast_chart(data, msku):
    """单个MSKU的库存预测图表"""
    if data is None or data.empty:
        fig = go.Figure()
        fig.add_annotation(text="无数据可展示", x=0.5, y=0.5, showarrow=False, font=dict(size=16))
        fig.update_layout(title=f"{msku} 库存预测", plot_bgcolor="#f8f9fa", height=400)
        return fig

    row = data.iloc[0]

    # ========== 核心调整1：全维度数据兜底，避免崩溃 ==========
    # 记录时间兼容（转换为datetime，处理空值）
    start_date = pd.to_datetime(row["记录时间"]) if not pd.isna(row["记录时间"]) else pd.Timestamp.now()
    end_date = pd.to_datetime(END_DATE)  # 确保end_date是datetime类型

    # 日均销量兜底（避免0/负数/空值）
    base_avg = row["日均"] if (row["日均"] > 0 and not pd.isna(row["日均"])) else 0.1

    # 总库存兜底（避免负数/空值）
    total_stock = row["全部总库存"] if (row["全部总库存"] >= 0 and not pd.isna(row["全部总库存"])) else 0
    remaining_stock = total_stock

    # ========== 调整2：标准化时间段系数（兼容datetime） ==========
    TIME_PERIODS = [
        {"start": pd.to_datetime("2026-10-16"), "end": pd.to_datetime("2026-11-15"), "coefficient": 1},
        {"start": pd.to_datetime("2026-11-16"), "end": pd.to_datetime("2026-11-30"), "coefficient": 1},
        {"start": pd.to_datetime("2026-12-01"), "end": pd.to_datetime("2026-12-31"), "coefficient": 1}
    ]

    forecast_dates = []
    forecast_stock = []
    current_date = start_date

    # ========== 调整3：限制循环次数，防无限循环 ==========
    max_days = 730  # 最多预测2年
    day_count = 0

    while current_date <= end_date and remaining_stock > 0 and day_count < max_days:
        current_coeff = 1.0
        for period in TIME_PERIODS:
            if period["start"] <= current_date <= period["end"]:
                current_coeff = period["coefficient"]
                break

        daily_sales = base_avg * current_coeff
        remaining_stock = max(remaining_stock - daily_sales, 0)

        forecast_dates.append(current_date)
        forecast_stock.append(remaining_stock)

        current_date += timedelta(days=1)
        day_count += 1

    # 生成折线图
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_stock,
        mode="lines+markers",
        line=dict(color="#4169E1", width=2),
        name="预计库存（分阶段系数）"
    ))

    # ========== 调整4：兼容TARGET_DATE和STATUS_COLORS ==========
    # 添加目标日期线（兼容timestamp转换）
    fig.add_vline(
        x=pd.to_datetime(TARGET_DATE).timestamp() * 1000,
        line_dash="dash",
        line_color="#DC143C",
        annotation_text="目标消耗日期",
        annotation_position="top right",
        annotation_font=dict(color="#DC143C")
    )

    # 添加时间段系数标注（兼容空值/未定义）
    max_stock = max(forecast_stock) if forecast_stock else 100  # 兜底最大值
    for period in TIME_PERIODS:
        fig.add_annotation(
            x=period["start"],
            y=max_stock * 0.9,
            text=f"{period['start'].strftime('%m-%d')}起系数: {period['coefficient']}",
            showarrow=True,
            arrowhead=1,
            arrowcolor=STATUS_COLORS.get("低滞销风险", "#4169E1"),  # 兜底颜色
            font=dict(size=10, color=STATUS_COLORS.get("低滞销风险", "#4169E1"))
        )

    # 图表布局
    fig.update_layout(
        title=f"{msku} 库存消耗预测（含分阶段销量系数）",
        xaxis_title="日期",
        yaxis_title="剩余库存",
        plot_bgcolor="#f8f9fa",
        height=400,
        margin=dict(t=50, b=20, l=20, r=20)
    )

    # 横坐标优化（按月显示刻度，避免密集）
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
        dtick="M1",  # 按月显示刻度，替代864000000（每天）
        ticklabelmode="period"
    )

    return fig

def main():
    if "current_page" not in st.session_state:
        st.session_state.current_page = 1
    if "current_status_page" not in st.session_state:
        st.session_state.current_status_page = 1
    with st.sidebar:
        st.title("侧栏信息")
        from datetime import datetime
        from datetime import datetime, timedelta
        today = datetime.now().date()
        days_to_monday = today.weekday()
        monday_of_week = today - timedelta(days=days_to_monday)
        st.info(f"当周周一：{monday_of_week.strftime('%Y年%m月%d日')}")
        days_remaining = (TARGET_DATE.date() - monday_of_week).days
        st.info(f"目标消耗完成日期：{TARGET_DATE.strftime('%Y年%m月%d日')}")
        st.warning(f"⏰ 距离目标日期剩余：{days_remaining}天")
        # 2. 修正后的库存周转天数判断说明（和你的逻辑完全一致）
        st.subheader("📋 库存风险判断说明")
        st.markdown("""
        #### 一、滞销风险分类（按预计消耗时间）
        - **健康**：预计总库存用完时间 ≤ 2026年10月31日；
        - **低滞销风险**：预计用完时间超目标时间 0-10 天；
        - **中滞销风险**：预计用完时间超目标时间 10-20 天；
        - **高滞销风险**：预计用完时间超目标时间 > 20 天；
        - **非年份品**：无目标日期风险，仅统计库存数据。

        #### 二、库存周转状态分类（按周转天数）
        - **库存周转健康**：库存周转天数 ≤ 100 天；
        - **轻度滞销风险**：库存周转天数 100 < 天数 ≤ 150 天；
        - **中度滞销风险**：库存周转天数 150 < 天数 ≤ 180 天；
        - **严重滞销风险**：库存周转天数 > 180 天。

        #### 三、100天达标日均说明
        总库存周转天数需控制在100天内时，每日需完成的销量：
        > 100天达标日均 = 全部总库存 ÷ 100
        """, unsafe_allow_html=False)
        st.subheader("数据加载中...")
        try:
            data_url = "https://raw.githubusercontent.com/Jane-zzz-123/-/main/weekday11.xlsx"
            response = requests.get(data_url)
            response.raise_for_status()
            excel_data = BytesIO(response.content)
            import pandas as pd
            current_data = pd.read_excel(
                excel_data,
                sheet_name="当前数据",
                engine='openpyxl'
            )
            df = load_and_preprocess_data_from_df(current_data)
            if df is None:
                st.error("数据预处理失败，无法继续")
                st.stop()
            # 根据用户权限筛选店铺
            allowed_stores = st.session_state.get("allowed_stores")
            if allowed_stores is not None:
                df = df[df["店铺"].isin(allowed_stores)].copy()
                if df.empty:
                    st.error(f"您有权限的店铺（{', '.join(allowed_stores)}）没有数据")
                    st.stop()
            st.success("数据加载成功！")
        except Exception as e:
            st.error(f"数据加载失败：{str(e)}")
            try:
                excel_data.seek(0)
                xl = pd.ExcelFile(excel_data, engine='openpyxl')
                st.error(f"Excel文件中实际存在的sheet：{xl.sheet_names}")
            except:
                pass
            st.stop()
    st.title("库存滞销风险分析平台")
    if "filter_status" not in st.session_state:
        st.session_state.filter_status = None
    if "current_page" not in st.session_state:
        st.session_state.current_page = 1
    all_dates = sorted(df["记录时间"].unique())
    latest_date = all_dates[-1] if all_dates else None

    # ===================== 页面核心代码 =====================
    st.header("一、整体风险分析")
    # 记录时间筛选器
    selected_date = st.selectbox(
        "选择记录时间",
        options=all_dates,
        index=len(all_dates) - 1 if all_dates else 0,
        format_func=lambda x: x.strftime("%Y年%m月%d日")
    )

    # ========== 核心修改：分离全量数据和年份品数据 ==========
    # 1. 获取全量的当前周数据（包含年份品+非年份品）- 用于产品列表/单个MSKU
    current_data_full = get_week_data(df, selected_date)
    # 2. 过滤出年份品数据 - 用于指标/图表统计
    current_data_year = None
    if current_data_full is not None and not current_data_full.empty:
        current_data_year = current_data_full[current_data_full["是否年份品"] == True].copy()

    # 3. 获取全量的上周数据（包含年份品+非年份品）
    prev_data_full = get_previous_week_turnover_data(df, selected_date)
    # 4. 过滤出年份品的上周数据 - 用于环比统计
    prev_data_year = None
    if prev_data_full is not None and not prev_data_full.empty:
        prev_data_year = prev_data_full[prev_data_full["是否年份品"] == True].copy()

    # 赋值给原有变量（保持后续代码兼容）
    current_data = current_data_year  # 指标/图表用
    prev_data = prev_data_year  # 环比统计用

    st.subheader("1 店铺整体分析")
    if current_data is not None and not current_data.empty:
        stores = sorted(current_data["店铺"].unique())
        selected_store = st.selectbox("选择店铺进行分析", options=stores)
        if selected_store:
            # ========== 店铺数据初始化 ==========
            # 全量店铺数据（包含年份品+非年份品）- 用于产品列表/下载
            store_current_data_all = current_data_full[current_data_full["店铺"] == selected_store].copy()
            # 年份品店铺数据 - 用于指标/图表统计
            store_current_data = None
            if not store_current_data_all.empty:
                store_current_data = store_current_data_all[store_current_data_all["是否年份品"] == True].copy()
            store_current_metrics = calculate_status_metrics(
                store_current_data) if store_current_data is not None else {}
            st.subheader("年份品清仓风险分析")
            # ========== 上周数据处理 ==========
            def get_store_last_week_metrics():
                from datetime import timedelta
                if store_current_data is None or store_current_data.empty:
                    return {
                        "总MSKU数": 0, "健康": 0, "低滞销风险": 0, "中滞销风险": 0, "高滞销风险": 0,
                        "总滞销库存": 0
                    }, None

                current_date = pd.to_datetime(store_current_data["记录时间"].iloc[0])
                last_week_start = current_date - timedelta(days=14)
                last_week_end = current_date - timedelta(days=7)

                if prev_data_full is not None and not prev_data_full.empty:
                    prev_data_filtered = prev_data_full[prev_data_full["店铺"] == selected_store].copy()
                    prev_data_filtered['记录时间'] = pd.to_datetime(prev_data_filtered['记录时间'])
                    last_week_data = prev_data_filtered[
                        (prev_data_filtered['记录时间'] >= last_week_start) &
                        (prev_data_filtered['记录时间'] <= last_week_end)
                        ]
                    # 过滤年份品（只统计年份品）
                    last_week_data = last_week_data[
                        last_week_data["是否年份品"] == True].copy() if not last_week_data.empty else pd.DataFrame()

                    if not last_week_data.empty:
                        metrics = calculate_status_metrics(last_week_data)
                        metrics["总滞销库存"] = last_week_data[
                            "总滞销库存"].sum() if "总滞销库存" in last_week_data.columns else 0
                        return metrics, last_week_data

                return {
                    "总MSKU数": 0, "健康": 0, "低滞销风险": 0, "中滞销风险": 0, "高滞销风险": 0,
                    "总滞销库存": 0
                }, None

            store_last_week_metrics, last_week_data = get_store_last_week_metrics()

            # ========== 计算状态变化 ==========
            status_change = {
                "健康": {"改善": 0, "不变": 0, "恶化": 0},
                "低滞销风险": {"改善": 0, "不变": 0, "恶化": 0},
                "中滞销风险": {"改善": 0, "不变": 0, "恶化": 0},
                "高滞销风险": {"改善": 0, "不变": 0, "恶化": 0}
            }
            status_severity = {"健康": 0, "低滞销风险": 1, "中滞销风险": 2, "高滞销风险": 3}

            if last_week_data is not None and not last_week_data.empty and "MSKU" in store_current_data.columns:
                merged_data = pd.merge(
                    store_current_data[["MSKU", "年份品清仓风险"]],
                    last_week_data[["MSKU", "年份品清仓风险"]],
                    on="MSKU",
                    suffixes=("_current", "_prev"),
                    how="inner"
                )
                for _, row in merged_data.iterrows():
                    current_status = row["年份品清仓风险_current"]
                    prev_status = row["年份品清仓风险_prev"]
                    if current_status not in status_severity or prev_status not in status_severity:
                        continue
                    if current_status == prev_status:
                        status_change[current_status]["不变"] += 1
                    elif status_severity[current_status] < status_severity[prev_status]:
                        status_change[current_status]["改善"] += 1
                    else:
                        status_change[current_status]["恶化"] += 1

            # ========== 指标计算 ==========
            store_metrics = {}
            for metric in ["总MSKU数", "健康", "低滞销风险", "中滞销风险", "高滞销风险"]:
                current = int(store_current_metrics.get(metric, 0))
                last_week = int(store_last_week_metrics.get(metric, 0))
                diff = current - last_week
                pct = (diff / last_week) * 100 if last_week != 0 else 0.0
                store_metrics[metric] = {
                    "current": current,
                    "last_week": last_week,
                    "diff": diff,
                    "pct": round(pct, 2)
                }

            # ========== 辅助函数 ==========
            def get_overstock_compare_text(current_overstock, last_week_overstock, status=None):
                current = round(float(current_overstock), 2)
                last_week = round(float(last_week_overstock), 2)
                if last_week == 0:
                    return f"<br><span style='color:#666; font-size:0.8em;'>{status + ' ' if status else ''}总滞销库存: {current:.2f}</span>"
                diff = current - last_week
                trend = "↑" if diff > 0 else "↓" if diff < 0 else "→"
                color = "#DC143C" if diff > 0 else "#2E8B57" if diff < 0 else "#666"
                pct = (diff / last_week) * 100 if last_week != 0 else 0.0
                pct_text = f"{abs(pct):.2f}%"
                return f"<br><span style='color:{color}; font-size:0.8em;'>{status + ' ' if status else ''}总滞销库存: {current:.2f} ({trend}{abs(diff):.2f} {pct_text})</span>"

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

            def get_compare_text(metric_data, metric_name):
                if metric_data["last_week"] == 0:
                    return "<br><span style='color:#666; font-size:0.8em;'>无上周数据</span>"
                trend = "↑" if metric_data["diff"] > 0 else "↓" if metric_data["diff"] < 0 else "→"
                color = "#DC143C" if metric_data["diff"] > 0 else "#2E8B57" if metric_data["diff"] < 0 else "#666"
                pct_text = f"{abs(metric_data['pct']):.2f}%"
                if metric_name == "总MSKU数":
                    return f"<br><span style='color:{color}; font-size:0.8em;'>{trend} 上周{metric_data['last_week']}，变化{metric_data['diff']} ({pct_text})</span>"
                else:
                    status = "上升" if metric_data["diff"] > 0 else "下降" if metric_data["diff"] < 0 else "无变化"
                    return f"<br><span style='color:{color}; font-size:0.8em;'>{trend} 上周{metric_data['last_week']}，{status}{abs(metric_data['diff'])} ({pct_text})</span>"

            # ========== 指标卡片 ==========
            cols = st.columns(5)
            with cols[0]:
                data = store_metrics["总MSKU数"]
                compare_text = get_compare_text(data, "总MSKU数")
                total_overstock = store_current_data["总滞销库存"].sum() if (
                            store_current_data is not None and "总滞销库存" in store_current_data.columns) else 0
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
                healthy_overstock = store_current_data[store_current_data["年份品清仓风险"] == "健康"][
                    "总滞销库存"].sum() if (
                            store_current_data is not None and "年份品清仓风险" in store_current_data.columns and "总滞销库存" in store_current_data.columns) else 0
                last_week_healthy_overstock = last_week_data[last_week_data["年份品清仓风险"] == "健康"][
                    "总滞销库存"].sum() if (
                            last_week_data is not None and "年份品清仓风险" in last_week_data.columns and "总滞销库存" in last_week_data.columns) else 0
                overstock_text = get_overstock_compare_text(healthy_overstock, last_week_healthy_overstock,
                                                            status="健康")
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
                low_risk_overstock = store_current_data[store_current_data["年份品清仓风险"] == "低滞销风险"][
                    "总滞销库存"].sum() if (
                            store_current_data is not None and "年份品清仓风险" in store_current_data.columns and "总滞销库存" in store_current_data.columns) else 0
                last_week_low_risk_overstock = last_week_data[last_week_data["年份品清仓风险"] == "低滞销风险"][
                    "总滞销库存"].sum() if (
                            last_week_data is not None and "年份品清仓风险" in last_week_data.columns and "总滞销库存" in last_week_data.columns) else 0
                overstock_text = get_overstock_compare_text(low_risk_overstock, last_week_low_risk_overstock,
                                                            status="低风险")
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
                mid_risk_overstock = store_current_data[store_current_data["年份品清仓风险"] == "中滞销风险"][
                    "总滞销库存"].sum() if (
                            store_current_data is not None and "年份品清仓风险" in store_current_data.columns and "总滞销库存" in store_current_data.columns) else 0
                last_week_mid_risk_overstock = last_week_data[last_week_data["年份品清仓风险"] == "中滞销风险"][
                    "总滞销库存"].sum() if (
                            last_week_data is not None and "年份品清仓风险" in last_week_data.columns and "总滞销库存" in last_week_data.columns) else 0
                overstock_text = get_overstock_compare_text(mid_risk_overstock, last_week_mid_risk_overstock,
                                                            status="中风险")
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
                high_risk_overstock = store_current_data[store_current_data["年份品清仓风险"] == "高滞销风险"][
                    "总滞销库存"].sum() if (
                            store_current_data is not None and "年份品清仓风险" in store_current_data.columns and "总滞销库存" in store_current_data.columns) else 0
                last_week_high_risk_overstock = last_week_data[last_week_data["年份品清仓风险"] == "高滞销风险"][
                    "总滞销库存"].sum() if (
                            last_week_data is not None and "年份品清仓风险" in last_week_data.columns and "总滞销库存" in last_week_data.columns) else 0
                overstock_text = get_overstock_compare_text(high_risk_overstock, last_week_high_risk_overstock,
                                                            status="高风险")
                change_text = get_status_change_text("高滞销风险")
                render_metric_card(
                    f"{selected_store} 高滞销风险{compare_text}{overstock_text}{change_text}",
                    data["current"],
                    data["diff"],
                    data["pct"],
                    STATUS_COLORS["高滞销风险"]
                )

            # ========== 图表部分 ==========
            col1, col2, col3 = st.columns(3)
            # 1.1 状态分布柱状图
            with col1:
                status_data = pd.DataFrame({
                    "状态": ["健康", "低滞销风险", "中滞销风险", "高滞销风险"],
                    "MSKU数": [store_current_metrics.get(stat, 0) for stat in
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

            # 1.2 状态判断饼图
            with col2:
                pie_data = pd.DataFrame({
                    "状态": ["健康", "低滞销风险", "中滞销风险", "高滞销风险"],
                    "MSKU数": [store_current_metrics.get(stat, 0) for stat in
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

            # 1.3 环比上周变化柱形图
            with col3:
                change_data = pd.DataFrame({
                    "状态": ["健康", "低滞销风险", "中滞销风险", "高滞销风险"],
                    "本周MSKU数": [store_current_metrics.get(stat, 0) for stat in
                                   ["健康", "低滞销风险", "中滞销风险", "高滞销风险"]],
                    "上周MSKU数": [store_last_week_metrics.get(stat, 0) for stat in
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

            # ========== 风险汇总表 ==========
            if df is not None and not df.empty and selected_store:
                # 获取当前周全量数据并过滤年份品
                current_week_full_data = get_week_data_year_product(df, selected_date)
                current_week_store_data = None
                if current_week_full_data is not None and not current_week_full_data.empty:
                    current_week_store_data = current_week_full_data[
                        current_week_full_data["店铺"] == selected_store].copy()
                    # 只保留年份品
                    current_week_store_data = current_week_store_data[current_week_store_data[
                                                                          "是否年份品"] == True].copy() if not current_week_store_data.empty else None

                # 获取上周全量数据并过滤年份品
                previous_week_full_data = get_previous_week_data(df, selected_date)
                previous_week_store_data = None
                if previous_week_full_data is not None and not previous_week_full_data.empty:
                    previous_week_store_data = previous_week_full_data[
                        previous_week_full_data["店铺"] == selected_store].copy()
                    # 只保留年份品
                    previous_week_store_data = previous_week_store_data[previous_week_store_data[
                                                                            "是否年份品"] == True].copy() if not previous_week_store_data.empty else None

                # 生成风险汇总表
                store_summary_df = create_risk_summary_table(current_week_store_data, previous_week_store_data)
                render_risk_summary_table(store_summary_df)

            # ========== 新增：周转状态专用辅助函数 ==========
            # ========== 优化：周转状态专用辅助函数（关联上周数据） ==========
            def calculate_turnover_metrics(data, prev_data=None):
                """
                计算周转状态分布指标（全量商品）
                新增：prev_data 传入上周数据，支持环比
                """
                # 基础指标（本周）
                base_metrics = {"总MSKU数": 0, "库存周转健康": 0, "轻度滞销风险": 0, "中度滞销风险": 0,
                                "严重滞销风险": 0, "数据异常": 0}
                if data is not None and not data.empty:
                    total = len(data)
                    status_counts = data["库存周转状态判断"].value_counts().to_dict()
                    base_metrics = {"总MSKU数": total}
                    for status in ["库存周转健康", "轻度滞销风险", "中度滞销风险", "严重滞销风险", "数据异常"]:
                        base_metrics[status] = status_counts.get(status, 0)
                    # 计算周转滞销库存总量
                    base_metrics["周转滞销库存总量"] = data[
                        "周转天数超过100天的滞销数量"].sum() if "周转天数超过100天的滞销数量" in data.columns else 0

                # 环比指标（本周 vs 上周）
                compare_metrics = {}
                if prev_data is not None and not prev_data.empty:
                    # 计算上周基础指标
                    prev_total = len(prev_data)
                    prev_status_counts = prev_data["库存周转状态判断"].value_counts().to_dict()
                    prev_metrics = {"总MSKU数": prev_total}
                    for status in ["库存周转健康", "轻度滞销风险", "中度滞销风险", "严重滞销风险", "数据异常"]:
                        prev_metrics[status] = prev_status_counts.get(status, 0)
                    prev_metrics["周转滞销库存总量"] = prev_data[
                        "周转天数超过100天的滞销数量"].sum() if "周转天数超过100天的滞销数量" in prev_data.columns else 0

                    # 生成环比对比
                    compare_metrics = compare_turnover_metrics(base_metrics, prev_metrics)

                # 合并结果：本周指标 + 环比对比
                base_metrics["环比数据"] = compare_metrics
                # 新增：周转状态变化（改善/不变/恶化）
                base_metrics["状态变化"] = calculate_turnover_status_change(data,
                                                                            prev_data) if prev_data is not None else {}
                # 新增：周转滞销库存环比
                base_metrics["周转滞销库存总量_上周"] = prev_data["周转天数超过100天的滞销数量"].sum() if (
                            prev_data is not None and not prev_data.empty and "周转天数超过100天的滞销数量" in prev_data.columns) else 0

                return base_metrics

            def get_turnover_compare_text(current_overstock, last_week_overstock, status=None):
                """周转滞销库存对比文本生成"""
                current = round(float(current_overstock), 2)
                last_week = round(float(last_week_overstock), 2)
                if last_week == 0:
                    return f"<br><span style='color:#666; font-size:0.8em;'>{status + ' ' if status else ''}周转滞销库存: {current:.2f}</span>"
                diff = current - last_week
                trend = "↑" if diff > 0 else "↓" if diff < 0 else "→"
                color = "#DC143C" if diff > 0 else "#2E8B57" if diff < 0 else "#666"
                pct = (diff / last_week) * 100 if last_week != 0 else 0.0
                pct_text = f"{abs(pct):.2f}%"
                return f"<br><span style='color:{color}; font-size:0.8em;'>{status + ' ' if status else ''}周转滞销库存: {current:.2f} ({trend}{abs(diff):.2f} {pct_text})</span>"

            def get_turnover_status_change_text(status, turnover_status_change):
                """周转状态变化文本生成"""
                changes = turnover_status_change[status]
                total = changes["改善"] + changes["不变"] + changes["恶化"]
                if total == 0:
                    return "<br><span style='color:#666; font-size:0.8em;'>状态变化: 无数据</span>"
                return f"""<br>
                <span style='color:#2E8B57; font-size:0.8em;'>改善: {changes['改善']}</span> | 
                <span style='color:#666; font-size:0.8em;'>不变: {changes['不变']}</span> | 
                <span style='color:#DC143C; font-size:0.8em;'>恶化: {changes['恶化']}</span>
                """

            def get_turnover_compare_text_metric(metric_data, metric_name):
                """周转指标环比文本生成"""
                if metric_data["last_week"] == 0:
                    return "<br><span style='color:#666; font-size:0.8em;'>无上周数据</span>"
                trend = "↑" if metric_data["diff"] > 0 else "↓" if metric_data["diff"] < 0 else "→"
                color = "#DC143C" if metric_data["diff"] > 0 else "#2E8B57" if metric_data["diff"] < 0 else "#666"
                pct_text = f"{abs(metric_data['pct']):.2f}%"
                if metric_name == "总MSKU数":
                    return f"<br><span style='color:{color}; font-size:0.8em;'>{trend} 上周{metric_data['last_week']}，变化{metric_data['diff']} ({pct_text})</span>"
                else:
                    status = "上升" if metric_data["diff"] > 0 else "下降" if metric_data["diff"] < 0 else "无变化"
                    return f"<br><span style='color:{color}; font-size:0.8em;'>{trend} 上周{metric_data['last_week']}，{status}{abs(metric_data['diff'])} ({pct_text})</span>"

            # ========== 新增：2 全量商品库存周转分析 ==========
            st.subheader("所有品库存周转分析")

            # 1. 全量商品数据准备（包含年份品+非年份品）
            turnover_current_data = store_current_data_all.copy()  # 全量商品数据
            turnover_current_metrics = calculate_turnover_metrics(turnover_current_data)

            # 2. 上周周转数据处理
            def get_store_last_week_turnover_metrics():
                from datetime import timedelta
                if turnover_current_data is None or turnover_current_data.empty:
                    return {
                        "总MSKU数": 0, "库存周转健康": 0, "轻度滞销风险": 0, "中度滞销风险": 0, "严重滞销风险": 0,
                        "数据异常": 0,
                        "周转滞销库存总量": 0
                    }, None

                current_date = pd.to_datetime(turnover_current_data["记录时间"].iloc[0])
                last_week_start = current_date - timedelta(days=14)
                last_week_end = current_date - timedelta(days=7)

                if prev_data_full is not None and not prev_data_full.empty:
                    prev_data_filtered = prev_data_full[prev_data_full["店铺"] == selected_store].copy()
                    prev_data_filtered['记录时间'] = pd.to_datetime(prev_data_filtered['记录时间'])
                    last_week_data = prev_data_filtered[
                        (prev_data_filtered['记录时间'] >= last_week_start) &
                        (prev_data_filtered['记录时间'] <= last_week_end)
                        ]
                    # 全量商品（不过滤年份品）
                    if not last_week_data.empty:
                        metrics = calculate_turnover_metrics(last_week_data)
                        metrics["周转滞销库存总量"] = last_week_data[
                            "周转天数超过100天的滞销数量"].sum() if "周转天数超过100天的滞销数量" in last_week_data.columns else 0
                        return metrics, last_week_data

                return {
                    "总MSKU数": 0, "库存周转健康": 0, "轻度滞销风险": 0, "中度滞销风险": 0, "严重滞销风险": 0,
                    "数据异常": 0,
                    "周转滞销库存总量": 0
                }, None

            turnover_last_week_metrics, turnover_last_week_data = get_store_last_week_turnover_metrics()

            # 3. 周转状态变化计算
            turnover_status_change = {
                "库存周转健康": {"改善": 0, "不变": 0, "恶化": 0},
                "轻度滞销风险": {"改善": 0, "不变": 0, "恶化": 0},
                "中度滞销风险": {"改善": 0, "不变": 0, "恶化": 0},
                "严重滞销风险": {"改善": 0, "不变": 0, "恶化": 0},
                "数据异常": {"改善": 0, "不变": 0, "恶化": 0}
            }
            turnover_status_severity = {"库存周转健康": 0, "轻度滞销风险": 1, "中度滞销风险": 2, "严重滞销风险": 3,
                                        "数据异常": 4}

            if turnover_last_week_data is not None and not turnover_last_week_data.empty and "MSKU" in turnover_current_data.columns:
                merged_turnover_data = pd.merge(
                    turnover_current_data[["MSKU", "库存周转状态判断"]],
                    turnover_last_week_data[["MSKU", "库存周转状态判断"]],
                    on="MSKU",
                    suffixes=("_current", "_prev"),
                    how="inner"
                )
                for _, row in merged_turnover_data.iterrows():
                    current_status = row["库存周转状态判断_current"]
                    prev_status = row["库存周转状态判断_prev"]
                    if current_status not in turnover_status_severity or prev_status not in turnover_status_severity:
                        continue
                    if current_status == prev_status:
                        turnover_status_change[current_status]["不变"] += 1
                    elif turnover_status_severity[current_status] < turnover_status_severity[prev_status]:
                        turnover_status_change[current_status]["改善"] += 1
                    else:
                        turnover_status_change[current_status]["恶化"] += 1

            # 4. 周转指标计算
            turnover_metrics = {}
            for metric in ["总MSKU数", "库存周转健康", "轻度滞销风险", "中度滞销风险", "严重滞销风险", "数据异常"]:
                current = int(turnover_current_metrics.get(metric, 0))
                last_week = int(turnover_last_week_metrics.get(metric, 0))
                diff = current - last_week
                pct = (diff / last_week) * 100 if last_week != 0 else 0.0
                turnover_metrics[metric] = {
                    "current": current,
                    "last_week": last_week,
                    "diff": diff,
                    "pct": round(pct, 2)
                }

            # 5. 周转指标卡片
            cols_turnover = st.columns(6)
            with cols_turnover[0]:
                data = turnover_metrics["总MSKU数"]
                compare_text = get_turnover_compare_text_metric(data, "总MSKU数")
                total_turnover_overstock = turnover_current_data["周转天数超过100天的滞销数量"].sum() if (
                        turnover_current_data is not None and "周转天数超过100天的滞销数量" in turnover_current_data.columns) else 0
                last_week_total_turnover_overstock = turnover_last_week_metrics.get("周转滞销库存总量", 0)
                overstock_text = get_turnover_compare_text(total_turnover_overstock, last_week_total_turnover_overstock)
                render_metric_card(
                    f"{selected_store} 全量商品总数{compare_text}{overstock_text}",
                    data["current"],
                    data["diff"],
                    data["pct"],
                    "#000000"
                )

            with cols_turnover[1]:
                data = turnover_metrics["库存周转健康"]
                compare_text = get_turnover_compare_text_metric(data, "库存周转健康")
                healthy_turnover_overstock = \
                turnover_current_data[turnover_current_data["库存周转状态判断"] == "库存周转健康"][
                    "周转天数超过100天的滞销数量"].sum() if (
                        turnover_current_data is not None and "库存周转状态判断" in turnover_current_data.columns and "周转天数超过100天的滞销数量" in turnover_current_data.columns) else 0
                last_week_healthy_turnover_overstock = \
                turnover_last_week_data[turnover_last_week_data["库存周转状态判断"] == "库存周转健康"][
                    "周转天数超过100天的滞销数量"].sum() if (
                        turnover_last_week_data is not None and "库存周转状态判断" in turnover_last_week_data.columns and "周转天数超过100天的滞销数量" in turnover_last_week_data.columns) else 0
                overstock_text = get_turnover_compare_text(healthy_turnover_overstock,
                                                           last_week_healthy_turnover_overstock,
                                                           status="周转健康")
                change_text = get_turnover_status_change_text("库存周转健康", turnover_status_change)
                render_metric_card(
                    f"{selected_store} 周转健康{compare_text}{overstock_text}{change_text}",
                    data["current"],
                    data["diff"],
                    data["pct"],
                    TURNOVER_STATUS_COLORS["库存周转健康"]
                )

            with cols_turnover[2]:
                data = turnover_metrics["轻度滞销风险"]
                compare_text = get_turnover_compare_text_metric(data, "轻度滞销风险")
                low_turnover_overstock = \
                turnover_current_data[turnover_current_data["库存周转状态判断"] == "轻度滞销风险"][
                    "周转天数超过100天的滞销数量"].sum() if (
                        turnover_current_data is not None and "库存周转状态判断" in turnover_current_data.columns and "周转天数超过100天的滞销数量" in turnover_current_data.columns) else 0
                last_week_low_turnover_overstock = \
                turnover_last_week_data[turnover_last_week_data["库存周转状态判断"] == "轻度滞销风险"][
                    "周转天数超过100天的滞销数量"].sum() if (
                        turnover_last_week_data is not None and "库存周转状态判断" in turnover_last_week_data.columns and "周转天数超过100天的滞销数量" in turnover_last_week_data.columns) else 0
                overstock_text = get_turnover_compare_text(low_turnover_overstock, last_week_low_turnover_overstock,
                                                           status="轻度滞销")
                change_text = get_turnover_status_change_text("轻度滞销风险", turnover_status_change)
                render_metric_card(
                    f"{selected_store} 轻度滞销风险{compare_text}{overstock_text}{change_text}",
                    data["current"],
                    data["diff"],
                    data["pct"],
                    TURNOVER_STATUS_COLORS["轻度滞销风险"]
                )

            with cols_turnover[3]:
                data = turnover_metrics["中度滞销风险"]
                compare_text = get_turnover_compare_text_metric(data, "中度滞销风险")
                mid_turnover_overstock = \
                turnover_current_data[turnover_current_data["库存周转状态判断"] == "中度滞销风险"][
                    "周转天数超过100天的滞销数量"].sum() if (
                        turnover_current_data is not None and "库存周转状态判断" in turnover_current_data.columns and "周转天数超过100天的滞销数量" in turnover_current_data.columns) else 0
                last_week_mid_turnover_overstock = \
                turnover_last_week_data[turnover_last_week_data["库存周转状态判断"] == "中度滞销风险"][
                    "周转天数超过100天的滞销数量"].sum() if (
                        turnover_last_week_data is not None and "库存周转状态判断" in turnover_last_week_data.columns and "周转天数超过100天的滞销数量" in turnover_last_week_data.columns) else 0
                overstock_text = get_turnover_compare_text(mid_turnover_overstock, last_week_mid_turnover_overstock,
                                                           status="中度滞销")
                change_text = get_turnover_status_change_text("中度滞销风险", turnover_status_change)
                render_metric_card(
                    f"{selected_store} 中度滞销风险{compare_text}{overstock_text}{change_text}",
                    data["current"],
                    data["diff"],
                    data["pct"],
                    TURNOVER_STATUS_COLORS["中度滞销风险"]
                )

            with cols_turnover[4]:
                data = turnover_metrics["严重滞销风险"]
                compare_text = get_turnover_compare_text_metric(data, "严重滞销风险")
                high_turnover_overstock = \
                turnover_current_data[turnover_current_data["库存周转状态判断"] == "严重滞销风险"][
                    "周转天数超过100天的滞销数量"].sum() if (
                        turnover_current_data is not None and "库存周转状态判断" in turnover_current_data.columns and "周转天数超过100天的滞销数量" in turnover_current_data.columns) else 0
                last_week_high_turnover_overstock = \
                turnover_last_week_data[turnover_last_week_data["库存周转状态判断"] == "严重滞销风险"][
                    "周转天数超过100天的滞销数量"].sum() if (
                        turnover_last_week_data is not None and "库存周转状态判断" in turnover_last_week_data.columns and "周转天数超过100天的滞销数量" in turnover_last_week_data.columns) else 0
                overstock_text = get_turnover_compare_text(high_turnover_overstock, last_week_high_turnover_overstock,
                                                           status="严重滞销")
                change_text = get_turnover_status_change_text("严重滞销风险", turnover_status_change)
                render_metric_card(
                    f"{selected_store} 严重滞销风险{compare_text}{overstock_text}{change_text}",
                    data["current"],
                    data["diff"],
                    data["pct"],
                    TURNOVER_STATUS_COLORS["严重滞销风险"]
                )

            with cols_turnover[5]:
                data = turnover_metrics["数据异常"]
                compare_text = get_turnover_compare_text_metric(data, "数据异常")
                render_metric_card(
                    f"{selected_store} 数据异常{compare_text}",
                    data["current"],
                    data["diff"],
                    data["pct"],
                    TURNOVER_STATUS_COLORS["数据异常"]
                )

            # 6. 周转状态图表
            col1_turnover, col2_turnover, col3_turnover = st.columns(3)
            # 6.1 周转状态分布柱状图
            with col1_turnover:
                turnover_status_data = pd.DataFrame({
                    "状态": ["库存周转健康", "轻度滞销风险", "中度滞销风险", "严重滞销风险", "数据异常"],
                    "MSKU数": [turnover_current_metrics.get(stat, 0) for stat in
                               ["库存周转健康", "轻度滞销风险", "中度滞销风险", "严重滞销风险", "数据异常"]]
                })
                fig_turnover_status = px.bar(
                    turnover_status_data,
                    x="状态",
                    y="MSKU数",
                    color="状态",
                    color_discrete_map=TURNOVER_STATUS_COLORS,
                    title=f"{selected_store} 库存周转状态分布",
                    text="MSKU数",
                    height=400
                )
                fig_turnover_status.update_traces(
                    textposition="outside",
                    textfont=dict(size=12, weight="bold"),
                    marker=dict(line=dict(color="#fff", width=1))
                )
                fig_turnover_status.update_layout(
                    xaxis_title="周转状态",
                    yaxis_title="MSKU数量",
                    showlegend=True,
                    plot_bgcolor="#f8f9fa",
                    margin=dict(t=50, b=20, l=20, r=20)
                )
                st.plotly_chart(fig_turnover_status, use_container_width=True)

            # 6.2 周转状态占比饼图
            with col2_turnover:
                turnover_pie_data = pd.DataFrame({
                    "状态": ["库存周转健康", "轻度滞销风险", "中度滞销风险", "严重滞销风险", "数据异常"],
                    "MSKU数": [turnover_current_metrics.get(stat, 0) for stat in
                               ["库存周转健康", "轻度滞销风险", "中度滞销风险", "严重滞销风险", "数据异常"]]
                })
                total_turnover_msku = turnover_pie_data["MSKU数"].sum()
                turnover_pie_data["占比(%)"] = turnover_pie_data["MSKU数"].apply(
                    lambda x: round((x / total_turnover_msku) * 100, 1) if total_turnover_msku != 0 else 0.0
                )
                turnover_pie_data["自定义标签"] = turnover_pie_data.apply(
                    lambda row: f"{row['状态']}<br>{row['MSKU数']}个<br>({row['占比(%)']}%)",
                    axis=1
                )
                fig_turnover_pie = px.pie(
                    turnover_pie_data,
                    values="MSKU数",
                    names="状态",
                    color="状态",
                    color_discrete_map=TURNOVER_STATUS_COLORS,
                    title=f"{selected_store} 库存周转状态占比",
                    height=400,
                    labels={"MSKU数": "MSKU数量"}
                )
                fig_turnover_pie.update_traces(
                    text=turnover_pie_data["自定义标签"],
                    textinfo="text",
                    textfont=dict(size=10, weight="bold"),
                    hovertemplate="%{label}: %{value}个 (%{percent:.1%})"
                )
                fig_turnover_pie.update_layout(
                    showlegend=True,
                    legend_title="周转状态",
                    plot_bgcolor="#f8f9fa",
                    margin=dict(t=50, b=20, l=20, r=20)
                )
                st.plotly_chart(fig_turnover_pie, use_container_width=True)

            # 6.3 周转状态环比对比图
            with col3_turnover:
                turnover_change_data = pd.DataFrame({
                    "状态": ["库存周转健康", "轻度滞销风险", "中度滞销风险", "严重滞销风险", "数据异常"],
                    "本周MSKU数": [turnover_current_metrics.get(stat, 0) for stat in
                                   ["库存周转健康", "轻度滞销风险", "中度滞销风险", "严重滞销风险", "数据异常"]],
                    "上周MSKU数": [turnover_last_week_metrics.get(stat, 0) for stat in
                                   ["库存周转健康", "轻度滞销风险", "中度滞销风险", "严重滞销风险", "数据异常"]]
                })
                turnover_change_data_long = pd.melt(
                    turnover_change_data,
                    id_vars="状态",
                    value_vars=["本周MSKU数", "上周MSKU数"],
                    var_name="周期",
                    value_name="MSKU数"
                )
                fig_turnover_change = px.bar(
                    turnover_change_data_long,
                    x="状态",
                    y="MSKU数",
                    color="周期",
                    barmode="group",
                    color_discrete_map={"本周MSKU数": "#2E86AB", "上周MSKU数": "#A23B72"},
                    title=f"{selected_store} 周转状态变化对比",
                    height=400,
                    text="MSKU数"
                )
                fig_turnover_change.update_traces(
                    textposition="outside",
                    textfont=dict(size=10, weight="bold"),
                    marker=dict(line=dict(color="#fff", width=1))
                )
                fig_turnover_change.update_layout(
                    xaxis_title="周转状态",
                    yaxis_title="MSKU数量",
                    showlegend=True,
                    legend_title="周期",
                    plot_bgcolor="#f8f9fa",
                    margin=dict(t=50, b=20, l=20, r=20)
                )
                st.plotly_chart(fig_turnover_change, use_container_width=True)

            # ===================== 新增：全量商品周转风险汇总表核心函数 =====================
            # ===================== 新增：全量商品周转风险汇总表（匹配目标样式） =====================
            def create_turnover_summary_table(current_week_store_data, previous_week_store_data):
                """生成和目标样式一致的全量商品周转风险汇总表"""
                # 1. 基础数据初始化
                summary_data = []
                # 定义周转状态（对应目标表的风险等级）
                turnover_status_mapping = {
                    "库存周转健康": "健康",
                    "轻度滞销风险": "低滞销风险",
                    "中度滞销风险": "中滞销风险",
                    "严重滞销风险": "高滞销风险",
                    "数据异常": "数据异常"
                }
                # 组合风险维度（和目标表一致）
                combine_dimensions = [
                    ("低滞销风险+中滞销风险+高滞销风险", ["轻度滞销风险", "中度滞销风险", "严重滞销风险"]),
                    ("中滞销风险+高滞销风险", ["中度滞销风险", "严重滞销风险"])
                ]

                # 2. 计算本周/上周全量基础值（用于占比计算）
                # 本周全量
                current_total_msku = len(current_week_store_data) if (
                            current_week_store_data is not None and not current_week_store_data.empty) else 0
                current_total_stock = current_week_store_data["周转天数超过100天的滞销数量"].sum() if (
                            current_week_store_data is not None and not current_week_store_data.empty) else 0
                # 上周全量
                previous_total_msku = len(previous_week_store_data) if (
                            previous_week_store_data is not None and not previous_week_store_data.empty) else 0
                previous_total_stock = previous_week_store_data["周转天数超过100天的滞销数量"].sum() if (
                            previous_week_store_data is not None and not previous_week_store_data.empty) else 0

                # 3. 单状态维度统计（健康/低/中/高/数据异常）
                for status_key, status_name in turnover_status_mapping.items():
                    # 本周数据
                    current_msku = 0
                    current_stock = 0
                    if current_week_store_data is not None and not current_week_store_data.empty:
                        current_filter = current_week_store_data["库存周转状态判断"] == status_key
                        current_msku = len(current_week_store_data[current_filter])
                        current_stock = current_week_store_data[current_filter]["周转天数超过100天的滞销数量"].sum()

                    # 上周数据
                    previous_msku = 0
                    previous_stock = 0
                    if previous_week_store_data is not None and not previous_week_store_data.empty:
                        previous_filter = previous_week_store_data["库存周转状态判断"] == status_key
                        previous_msku = len(previous_week_store_data[previous_filter])
                        previous_stock = previous_week_store_data[previous_filter]["周转天数超过100天的滞销数量"].sum()

                    # 计算占比
                    msku_ratio = (current_msku / current_total_msku * 100) if current_total_msku != 0 else 0.0
                    stock_ratio = (current_stock / current_total_stock * 100) if current_total_stock != 0 else 0.0

                    # 计算环比变化（MSKU）
                    msku_diff = current_msku - previous_msku
                    msku_diff_pct = (msku_diff / previous_msku * 100) if previous_msku != 0 else 0.0
                    msku_change = f"{msku_diff} ({msku_diff_pct:.1f}%)" if previous_msku != 0 else f"{msku_diff} (0.0%)"

                    # 计算环比变化（库存）
                    stock_diff = current_stock - previous_stock
                    stock_diff_pct = (stock_diff / previous_stock * 100) if previous_stock != 0 else 0.0
                    stock_change = f"{stock_diff} ({stock_diff_pct:.1f}%)" if previous_stock != 0 else f"{stock_diff} (0.0%)"

                    summary_data.append({
                        "库存周转状态判断": status_name,
                        "MSKU数": current_msku,
                        "MSKU占比": f"{msku_ratio:.1f}%",
                        "MSKU环比变化": msku_change,
                        "周转天数超过100天的滞销数量": current_stock,
                        "周转天数超过100天的滞销数量占比": f"{stock_ratio:.1f}%",
                        "库存环比变化": stock_change
                    })

                # 4. 组合风险维度统计（低+中+高、中+高）
                for combine_name, combine_status in combine_dimensions:
                    # 本周数据
                    current_msku = 0
                    current_stock = 0
                    if current_week_store_data is not None and not current_week_store_data.empty:
                        current_filter = current_week_store_data["库存周转状态判断"].isin(combine_status)
                        current_msku = len(current_week_store_data[current_filter])
                        current_stock = current_week_store_data[current_filter]["周转天数超过100天的滞销数量"].sum()

                    # 上周数据
                    previous_msku = 0
                    previous_stock = 0
                    if previous_week_store_data is not None and not previous_week_store_data.empty:
                        previous_filter = previous_week_store_data["库存周转状态判断"].isin(combine_status)
                        previous_msku = len(previous_week_store_data[previous_filter])
                        previous_stock = previous_week_store_data[previous_filter]["周转天数超过100天的滞销数量"].sum()

                    # 计算占比
                    msku_ratio = (current_msku / current_total_msku * 100) if current_total_msku != 0 else 0.0
                    stock_ratio = (current_stock / current_total_stock * 100) if current_total_stock != 0 else 0.0

                    # 计算环比变化（MSKU）
                    msku_diff = current_msku - previous_msku
                    msku_diff_pct = (msku_diff / previous_msku * 100) if previous_msku != 0 else 0.0
                    msku_change = f"{msku_diff} ({msku_diff_pct:.1f}%)" if previous_msku != 0 else f"{msku_diff} (0.0%)"

                    # 计算环比变化（库存）
                    stock_diff = current_stock - previous_stock
                    stock_diff_pct = (stock_diff / previous_stock * 100) if previous_stock != 0 else 0.0
                    stock_change = f"{stock_diff} ({stock_diff_pct:.1f}%)" if previous_stock != 0 else f"{stock_diff} (0.0%)"

                    summary_data.append({
                        "库存周转状态判断": combine_name,
                        "MSKU数": current_msku,
                        "MSKU占比": f"{msku_ratio:.1f}%",
                        "MSKU环比变化": msku_change,
                        "周转天数超过100天的滞销数量": current_stock,
                        "周转天数超过100天的滞销数量占比": f"{stock_ratio:.1f}%",
                        "库存环比变化": stock_change
                    })

                # 转换为DataFrame
                summary_df = pd.DataFrame(summary_data)
                return summary_df

            def render_turnover_summary_table(summary_df):
                """渲染和目标样式一致的周转风险汇总表"""
                if summary_df is None or summary_df.empty:
                    st.warning("暂无全量商品周转风险数据")
                    return

                # 复刻目标表格样式（颜色/字体/边框）
                st.markdown("""
                    <style>
                    .risk-summary-table {
                        font-size: 14px;
                        width: 100%;
                        border-collapse: collapse;
                        font-family: Arial, sans-serif;
                    }
                    .risk-summary-table th {
                        background-color: #f5f5f5;
                        padding: 10px;
                        text-align: center;
                        border: 1px solid #dddddd;
                        font-weight: normal;
                    }
                    .risk-summary-table td {
                        padding: 10px;
                        text-align: center;
                        border: 1px solid #dddddd;
                    }
                    /* 风险等级颜色（匹配目标表） */
                    .low-risk { color: #f7b500; }
                    .mid-risk { color: #f7941d; }
                    .high-risk { color: #e63946; }
                    </style>
                """, unsafe_allow_html=True)

                # 构建带颜色的HTML表格
                html_table = "<table class='risk-summary-table'><thead><tr>"
                # 表头（和目标表完全一致）
                headers = ["库存周转状态判断", "MSKU数", "MSKU占比", "MSKU环比变化", "周转天数超过100天的滞销数量", "周转天数超过100天的滞销数量占比",
                           "库存环比变化"]
                for header in headers:
                    html_table += f"<th>{header}</th>"
                html_table += "</tr></thead><tbody>"

                # 表体（按风险等级加颜色）
                for _, row in summary_df.iterrows():
                    risk_name = row["库存周转状态判断"]
                    # 风险等级颜色匹配
                    color_class = ""
                    if "低滞销风险" in risk_name:
                        color_class = "low-risk"
                    elif "中滞销风险" in risk_name and "低" not in risk_name:
                        color_class = "mid-risk"
                    elif "高滞销风险" in risk_name and "低" not in risk_name and "中" not in risk_name:
                        color_class = "high-risk"

                    html_table += f"<tr {'class=' + color_class if color_class else ''}>"
                    for col in headers:
                        value = row[col]
                        html_table += f"<td>{value}</td>"
                    html_table += "</tr>"
                html_table += "</tbody></table>"

                st.markdown(html_table, unsafe_allow_html=True)

            st.subheader("库存周转状态判断汇总表")

            # 获取当前周全量商品数据
            current_week_turnover_data = get_week_data(df, selected_date)
            current_week_turnover_store = None
            if current_week_turnover_data is not None and not current_week_turnover_data.empty:
                current_week_turnover_store = current_week_turnover_data[
                    current_week_turnover_data["店铺"] == selected_store].copy()

            # 获取上周全量商品数据
            previous_week_turnover_data = get_previous_week_turnover_data(df, selected_date)
            previous_week_turnover_store = None
            if previous_week_turnover_data is not None and not previous_week_turnover_data.empty:
                previous_week_turnover_store = previous_week_turnover_data[
                    previous_week_turnover_data["店铺"] == selected_store].copy()

            # 生成并渲染匹配样式的周转风险汇总表
            turnover_summary_df = create_turnover_summary_table(current_week_turnover_store,
                                                                previous_week_turnover_store)
            render_turnover_summary_table(turnover_summary_df)

            # ========== 库存消耗天数组合图 ==========
            st.subheader(f"{selected_store} 库存消耗天数分布（MSKU数+总滞销库存）")
            if not store_current_data_all.empty:
                today = pd.to_datetime(store_current_data_all["记录时间"].iloc[0])
                days_to_target = (TARGET_DATE - today).days

                # 只统计年份品的库存消耗天数
                valid_days = store_current_data["预计总库存需要消耗天数"].clip(lower=0) if (
                            store_current_data is not None and not store_current_data.empty) else pd.Series()
                max_days = valid_days.max() if not valid_days.empty else 0
                bin_width = 20
                num_bins = int((max_days + bin_width - 1) // bin_width)
                bins = [i * bin_width for i in range(num_bins + 1)]
                bin_labels = [f"{bins[i]}-{bins[i + 1]}" for i in range(len(bins) - 1)]

                msku_count = pd.Series()
                if not valid_days.empty:
                    msku_count = pd.cut(
                        valid_days,
                        bins=bins,
                        labels=bin_labels,
                        include_lowest=True
                    ).value_counts().sort_index()

                temp_df = store_current_data[["预计总库存需要消耗天数", "总滞销库存"]].copy() if (
                            store_current_data is not None and not store_current_data.empty) else pd.DataFrame()
                if not temp_df.empty:
                    temp_df["预计总库存需要消耗天数"] = temp_df["预计总库存需要消耗天数"].clip(lower=0)
                    temp_df["天数区间"] = pd.cut(
                        temp_df["预计总库存需要消耗天数"],
                        bins=bins,
                        labels=bin_labels,
                        include_lowest=True
                    )
                    overstock_sum = temp_df.groupby("天数区间")["总滞销库存"].sum().sort_index()
                else:
                    overstock_sum = pd.Series()

                combined_data = pd.DataFrame({
                    "天数区间": bin_labels,
                    "MSKU数量": [msku_count.get(label, 0) for label in bin_labels],
                    "总滞销库存": [overstock_sum.get(label, 0.0) for label in bin_labels]
                })

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
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    plot_bgcolor="#f8f9fa",
                    margin=dict(t=50, b=80, l=20, r=20)
                )
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

            # ========== 产品列表（全量数据） ==========
            st.subheader(f"{selected_store} 产品列表（年份品+非年份品）")
            display_columns = [
                "店铺", "MSKU", "品名", "记录时间",
                "日均", "7天日均", "14天日均", "28天日均",
                "FBA+AWD+在途库存", "本地可用", "全部总库存", "预计FBA+AWD+在途用完时间",
                "预计总库存用完", "库存周转状态判断", "总库存周转天数100天内达标日均","周转天数超过100天的滞销数量",
                "年份品清仓风险", "预计清完FBA+AWD+在途需要的日均", "清库存的目标日均", "FBA+AWD+在途滞销数量",
                "本地滞销数量", "总滞销库存",
                "预计总库存需要消耗天数", "预计用完时间比目标时间多出来的天数", "环比上周库年份品滞销风险变化",
                "是否年份品"  # 新增列，方便查看是否年份品
            ]
            # 用全量数据渲染产品列表
            render_product_detail_table(
                store_current_data_all,  # 核心：全量数据（年份品+非年份品）
                prev_data_full[prev_data_full["店铺"] == selected_store] if (
                            prev_data_full is not None and not prev_data_full.empty) else None,
                page=st.session_state.current_page,
                page_size=30,
                table_id=f"store_{selected_store}"
            )

            # ========== 下载数据（全量） ==========
            if not store_current_data_all.empty:
                existing_cols = [col for col in display_columns if col in store_current_data_all.columns]
                download_data = store_current_data_all[existing_cols].copy()
                date_cols = ["记录时间", "预计FBA+AWD+在途用完时间", "预计总库存用完"]
                for col in date_cols:
                    if col in download_data.columns:
                        download_data[col] = pd.to_datetime(download_data[col]).dt.strftime("%Y-%m-%d")
                csv = download_data.to_csv(index=False, encoding='utf-8-sig')
                today_str = pd.to_datetime(store_current_data_all["记录时间"].iloc[0]).strftime("%Y%m%d")
                file_name = f"{selected_store}_产品列表_全量_{today_str}.csv"
                st.download_button(
                    label="下载全量产品列表（年份品+非年份品）",
                    data=csv,
                    file_name=file_name,
                    mime="text/csv",
                    key=f"download_{selected_store}_all"
                )
    else:
        st.warning("无店铺数据可分析")

    # ========== 单个MSKU分析（全量数据） ==========
    st.subheader("单个MSKU分析（支持年份品+非年份品）")
    if current_data_full is not None and not current_data_full.empty:
        # 从全量数据中获取所有MSKU
        msku_list = sorted(current_data_full["MSKU"].unique())

        # MSKU查询框
        col1, col2 = st.columns([3, 1])
        with col1:
            msku_query = st.text_input(
                "输入MSKU查询（支持模糊搜索）",
                placeholder="例如：ABC123 或 直接输入非年份品MSKU",
                key="msku_query"
            )

        # 过滤MSKU列表
        filtered_mskus = []
        if msku_query:
            filtered_mskus = [msku for msku in msku_list if msku_query.strip().lower() in msku.lower()]
            if not filtered_mskus:
                st.warning(f"未找到包含 '{msku_query}' 的MSKU，请检查输入")
                filtered_mskus = msku_list
        else:
            filtered_mskus = msku_list

        with col2:
            selected_msku = st.selectbox("或从列表选择MSKU", options=filtered_mskus, key="msku_select")

        # 显示选中的MSKU详情
        if selected_msku:
            # 从全量数据中获取产品信息
            product_data = current_data_full[current_data_full["MSKU"] == selected_msku].copy()
            if not product_data.empty:
                product_info = product_data.iloc[0].to_dict()
                st.subheader(f"MSKU详情：{selected_msku}")

                # 显示详情列
                display_cols = [
                    "MSKU", "品名", "店铺", "是否年份品",
                    "日均", "7天日均", "14天日均", "28天日均",
                    "FBA+AWD+在途库存", "本地可用", "全部总库存", "预计FBA+AWD+在途用完时间", "预计总库存用完",
                    "库存周转状态判断", "总库存周转天数100天内达标日均","周转天数超过100天的滞销数量",
                    "年份品清仓风险", "预计清完FBA+AWD+在途需要的日均", "清库存的目标日均", "FBA+AWD+在途滞销数量",
                    "本地滞销数量", "总滞销库存",
                    "预计总库存需要消耗天数", "预计用完时间比目标时间多出来的天数", "环比上周库年份品滞销风险变化"
                ]
                valid_display_cols = [col for col in display_cols if col in product_data.columns]
                info_df = product_data[valid_display_cols].copy()

                # 格式化日期列
                date_cols = ["预计FBA+AWD+在途用完时间", "预计总库存用完"]
                for col in date_cols:
                    if col in info_df.columns:
                        info_df[col] = pd.to_datetime(info_df[col]).dt.strftime("%Y-%m-%d")

                # 状态颜色渲染（加默认值）
                if "年份品清仓风险" in info_df.columns:
                    info_df["年份品清仓风险"] = info_df["年份品清仓风险"].apply(
                        lambda x: f"<span style='color:{STATUS_COLORS.get(x, '#808080')}; font-weight:bold;'>{x}</span>"
                    )

                # 系数列处理（修复语法错误）
                coefficient_cols = [
                    "10月16-11月15日系数",
                    "11月16-30日系数",
                    "12月1-31日系数"
                ]
                for col in coefficient_cols:
                    if col in info_df.columns:
                        info_df[col] = info_df[col].round(2)

                # 显示表格
                st.markdown(info_df.to_html(escape=False, index=False), unsafe_allow_html=True)

                # 库存预测图
                forecast_fig = render_stock_forecast_chart(product_data, selected_msku)
                st.plotly_chart(forecast_fig, use_container_width=True)
            else:
                st.warning(f"未找到MSKU {selected_msku} 的数据")
    else:
        st.warning("无产品数据可分析")

    # 第二部分：趋势与变化分析
    st.header("2 近一个月的趋势与变化分析")
    # 2.1 三周状态变化趋势
    st.subheader("2.1.1 近一个月状态变化趋势（年份品清仓风险）")
    trend_fig = render_four_week_status_chart(df, all_dates)
    st.plotly_chart(trend_fig, use_container_width=True)
    # 新增：全量品库存周转状态近一个月趋势
    st.subheader("2.1.2 近一个月状态变化趋势（全量品库存周转）")
    turnover_trend_fig = render_turnover_four_week_status_chart(df, all_dates)
    # 原有：年份品店铺清仓风险趋势
    # 2.2 店铺周变化情况
    # 原有：年份品店铺每周清仓风险变化
    st.subheader("2.2.1 各店铺年份品每周清仓风险变化")
    render_store_weekly_changes(df, all_dates)

    # 新增：全量品店铺每周库存周转状态变化
    st.subheader("2.2.2 各店铺全量品每周库存周转状态变化")
    render_turnover_store_weekly_changes(df, all_dates)
    # 2.2 店铺周趋势变化情况
    st.subheader("2.3.1 各店铺年份品清仓风险趋势")
    render_store_trend_charts(df, all_dates)
    # 新增：全量品店铺库存周转趋势
    st.subheader("2.3.2 各店铺全量品库存周转状态趋势")
    render_turnover_store_trend_charts(df, all_dates)
    # 2.4 店铺与状态变化联合分析
    st.subheader("2.4 店铺与状态变化联合分析")
    if df is not None and not df.empty:
        all_stores = sorted(df["店铺"].unique())
        selected_analysis_store = st.selectbox(
            "选择店铺进行联合分析",
            options=["全部"] + all_stores
        )
        analysis_data = df.copy()
        if selected_analysis_store != "全部":
            analysis_data = analysis_data[analysis_data["店铺"] == selected_analysis_store]
        analysis_data = analysis_data.sort_values(by=["店铺", "MSKU"])
        display_columns = [
            "MSKU", "品名", "店铺",
            "日均", "7天日均", "14天日均", "28天日均",
            # "10月16-11月15日系数", "10月16-11月15日调整后日均",
            # "11月16-30日系数", "11月16-30日调整后日均",
            # "12月1-31日系数", "12月1-31日调整后日均",
            "FBA+AWD+在途库存", "本地可用",
            "全部总库存", "预计FBA+AWD+在途用完时间",
            "预计总库存用完", "库存周转状态判断", "总库存周转天数100天内达标日均","周转天数超过100天的滞销数量",
            "年份品清仓风险",  "预计清完FBA+AWD+在途需要的日均","清库存的目标日均",
            "FBA+AWD+在途滞销数量", "本地滞销数量", "总滞销库存",
            "预计总库存需要消耗天数", "预计用完时间比目标时间多出来的天数",
            "环比上周库年份品滞销风险变化"
        ]
        render_status_change_table(
            analysis_data,
            page=st.session_state.current_status_page,
            page_size=30
        )
        if not analysis_data.empty:
            expected_columns = [
                "MSKU", "品名", "店铺", "记录时间",
                "日均", "7天日均", "14天日均", "28天日均",
                # "10月16-11月15日系数", "10月16-11月15日调整后日均",
                # "11月16-30日系数", "11月16-30日调整后日均",
                # "12月1-31日系数", "12月1-31日调整后日均",
                "FBA+AWD+在途库存", "本地可用",
                "全部总库存", "预计FBA+AWD+在途用完时间",
                "预计总库存用完", "库存周转状态判断", "总库存周转天数100天内达标日均","周转天数超过100天的滞销数量",
                "年份品清仓风险", "预计清完FBA+AWD+在途需要的日均", "清库存的目标日均",
                "FBA+AWD+在途滞销数量", "本地滞销数量", "总滞销库存",
                "预计总库存需要消耗天数", "预计用完时间比目标时间多出来的天数",
                "环比上周库年份品滞销风险变化"
            ]
            filtered_data = analysis_data.copy()
            actual_columns = filtered_data.columns.tolist()
            valid_columns = [col for col in expected_columns if col in actual_columns]
            missing_columns = [col for col in expected_columns if col not in actual_columns]
            if missing_columns:
                st.warning(f"数据中缺少以下列，已自动跳过：{', '.join(missing_columns)}")
            if valid_columns:
                download_data = filtered_data[valid_columns]
            else:
                st.error("没有找到有效的列用于生成下载数据，请检查数据格式是否正确")
                download_data = pd.DataFrame()
            if "记录时间" in download_data.columns:
                download_data["记录时间"] = pd.to_datetime(download_data["记录时间"]).dt.strftime("%Y-%m-%d")
            csv = download_data.to_csv(index=False, encoding='utf-8-sig')
            store_part = selected_analysis_store if selected_analysis_store != "全部" else "所有店铺"
            file_name = f"店铺状态变化联合分析_{store_part}.csv"
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
        all_mskus = sorted(df["MSKU"].unique())
        search_term = st.text_input(
            "搜索产品（MSKU或品名）",
            placeholder="输入关键词搜索..."
        )
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
        selected_analysis_msku = st.selectbox(
            "选择产品进行详细分析",
            options=filtered_mskus
        )
        if selected_analysis_msku:
            product_history_data = df[df["MSKU"] == selected_analysis_msku].sort_values("记录时间", ascending=False)
            display_cols = [
                "MSKU", "品名", "店铺", "记录时间",
                "日均", "7天日均", "14天日均", "28天日均",
                # "10月16-11月15日系数", "10月16-11月15日调整后日均",
                # "11月16-30日系数", "11月16-30日调整后日均",
                # "12月1-31日系数", "12月1-31日调整后日均",
                "FBA+AWD+在途库存", "本地可用", "全部总库存", "预计FBA+AWD+在途用完时间", "预计总库存用完",
                "库存周转状态判断", "总库存周转天数100天内达标日均","周转天数超过100天的滞销数量",
                "年份品清仓风险",  "预计清完FBA+AWD+在途需要的日均","清库存的目标日均", "FBA+AWD+在途滞销数量", "本地滞销数量", "总滞销库存",
                "预计总库存需要消耗天数", "预计用完时间比目标时间多出来的天数", "环比上周库年份品滞销风险变化"
            ]
            valid_display_cols = [col for col in display_cols if col in product_history_data.columns]
            if not valid_display_cols:
                st.error("⚠️ 没有可展示的有效列（所有指定列均不存在）")
                table_data = pd.DataFrame()
            else:
                table_data = product_history_data[valid_display_cols].copy()
            date_cols = ["记录时间", "预计FBA+AWD+在途用完时间", "预计总库存用完"]
            for col in date_cols:
                if col in table_data.columns:
                    table_data[col] = pd.to_datetime(table_data[col]).dt.strftime("%Y-%m-%d")
            if "年份品清仓风险" in table_data.columns:
                table_data["年份品清仓风险"] = table_data["年份品清仓风险"].apply(
                    lambda x: f"<span style='color:{STATUS_COLORS[x]}; font-weight:bold;'>{x}</span>"
                )
            if "环比上周库年份品滞销风险变化" in table_data.columns:
                def color_status_change(x):
                    if x == "改善":
                        return f"<span style='color:#2E8B57; font-weight:bold;'>{x}</span>"
                    elif x == "恶化":
                        return f"<span style='color:#DC143C; font-weight:bold;'>{x}</span>"
                    else:
                        return f"<span style='color:#000000; font-weight:bold;'>{x}</span>"
                table_data["环比上周库年份品滞销风险变化"] = table_data["环比上周库年份品滞销风险变化"].apply(
                    color_status_change)
            st.subheader("产品历史数据")
            st.markdown(table_data.to_html(escape=False, index=False), unsafe_allow_html=True)
            forecast_chart = render_product_detail_chart(df, selected_analysis_msku)
            st.plotly_chart(forecast_chart, use_container_width=True)
    else:
        st.warning("无产品数据可进行详细分析")
if __name__ == "__main__":
    main()