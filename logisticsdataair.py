import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# ---------------------- 页面基础配置 ----------------------
st.set_page_config(
    page_title="空派物流交期分析看板",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------------------- 数据读取与预处理 ----------------------
@st.cache_data
def load_data():
    """读取空派数据并预处理"""
    # 读取指定sheet
    url = "https://github.com/Jane-zzz-123/Logistics/raw/main/Logisticsdata.xlsx"
    df_air = pd.read_excel(url, sheet_name="上架完成-空运")

    # 指定需要分析的列
    target_cols = [
        "FBA号", "店铺", "仓库", "货代", "异常备注",
        "发货-起飞", "到港-提取", "提取-签收", "签收-完成上架",
        "发货-签收", "发货-完成上架","清关耗时", "到货年月",
        "签收-发货时间", "上架完成-发货时间",
        "预计物流时效-实际物流时效差值(绝对值)",
        "预计物流时效-实际物流时效差值", "提前/延期"
        "预计物流时效-实际物流时效差值（货代）",
        "提前/延期（货代）", "提前/延期（仓库）"
    ]

    # 确保只保留目标列（处理列名可能的空格/大小写问题）
    df_air = df_air[[col for col in target_cols if col in df_air.columns]]

    # 数据类型处理
    df_air["到货年月"] = pd.to_datetime(df_air["到货年月"], errors='coerce').dt.strftime("%Y-%m")
    df_air = df_air.dropna(subset=["到货年月"])  # 去除到货年月为空的数据

    # 数值列处理
    numeric_cols = [
        "签收-发货时间", "上架完成-发货时间",
        "预计物流时效-实际物流时效差值(绝对值)",
        "预计物流时效-实际物流时效差值",
        "预计物流时效-实际物流时效差值（货代）"
    ]
    for col in numeric_cols:
        if col in df_air.columns:
            df_air[col] = pd.to_numeric(df_air[col], errors='coerce').fillna(0)

    return df_air


# 加载数据
df_air = load_data()


# ---------------------- 工具函数 ----------------------
def get_prev_month(current_month):
    """获取上个月的年月字符串（格式：YYYY-MM）"""
    try:
        current = datetime.strptime(current_month, "%Y-%m")
        prev_month = current.replace(day=1) - pd.Timedelta(days=1)
        return prev_month.strftime("%Y-%m")
    except:
        return ""


def calculate_percent_change(current, prev):
    """计算环比变化百分比"""
    try:
        if prev == 0:
            return 0 if current == 0 else 100
        return ((current - prev) / prev) * 100
    except:
        return 0


def highlight_large_cells(val, avg, col_name):
    """高亮大于平均值的单元格"""
    try:
        # 跳过非数值和平均值行
        if pd.isna(val) or val == "-" or str(val) == "平均值":
            return ""
        val_num = float(val)
        if val_num > avg:
            return "background-color: #ffcccc"  # 浅红色
    except:
        pass
    return ""


def highlight_change(val):
    """高亮环比变化（红升绿降）"""
    try:
        # 处理空值和非数值
        if pd.isna(val) or val == "-" or str(val).strip() == "":
            return ""

        # 提取数值
        val_str = str(val).replace('%', '').strip()
        val_num = float(val_str)

        # 设置颜色
        if val_num > 0:
            return "color: red"
        elif val_num < 0:
            return "color: green"
    except:
        pass
    return ""


# ---------------------- 主页面构建 ----------------------
st.title("📦 空派分析看板区域")
st.divider()

# ===================== 一、当月的情况 =====================
st.subheader("🔍 当月空派分析")

# 时间筛选器（到货年月，最新的在最上方）
month_options = sorted(df_air["到货年月"].unique(), reverse=True) if len(df_air["到货年月"].unique()) > 0 else []
selected_month = st.selectbox(
    "选择到货年月",
    options=month_options,
    index=0 if month_options else None,
    key="month_selector_current"
) if month_options else st.write("⚠️ 暂无可用的到货年月数据")

# 筛选当月数据
if month_options and selected_month:
    df_current = df_air[df_air["到货年月"] == selected_month].copy()
    # 获取上月数据
    prev_month = get_prev_month(selected_month)
    df_prev = df_air[
        df_air["到货年月"] == prev_month].copy() if prev_month and prev_month in month_options else pd.DataFrame()

    # ---------------------- ① 核心指标卡片 ----------------------
    st.markdown("### 核心指标")

    # ---------------------- ① 核心指标卡片 ----------------------
    st.markdown("### 核心指标")

    # 计算核心指标
    # 1. FBA单数
    current_fba = len(df_current)
    prev_fba = len(df_prev) if not df_prev.empty else 0
    fba_change = current_fba - prev_fba
    fba_change_text = f"{'↑' if fba_change > 0 else '↓' if fba_change < 0 else '—'} {abs(fba_change)} (上月: {prev_fba})"
    fba_change_color = "red" if fba_change > 0 else "green" if fba_change < 0 else "gray"

    # 2. 提前/准时数
    current_on_time = len(
        df_current[df_current["提前/延期"] == "提前/准时"]) if "提前/延期" in df_current.columns else 0
    prev_on_time = len(
        df_prev[df_prev["提前/延期"] == "提前/准时"]) if not df_prev.empty and "提前/延期" in df_prev.columns else 0
    on_time_change = current_on_time - prev_on_time
    on_time_change_text = f"{'↑' if on_time_change > 0 else '↓' if on_time_change < 0 else '—'} {abs(on_time_change)} (上月: {prev_on_time})"
    on_time_change_color = "red" if on_time_change > 0 else "green" if on_time_change < 0 else "gray"

    # 3. 延期数
    current_delay = len(df_current[df_current["提前/延期"] == "延期"]) if "提前/延期" in df_current.columns else 0
    prev_delay = len(
        df_prev[df_prev["提前/延期"] == "延期"]) if not df_prev.empty and "提前/延期" in df_prev.columns else 0
    delay_change = current_delay - prev_delay
    delay_change_text = f"{'↑' if delay_change > 0 else '↓' if delay_change < 0 else '—'} {abs(delay_change)} (上月: {prev_delay})"
    delay_change_color = "red" if delay_change > 0 else "green" if delay_change < 0 else "gray"

    # 4. 绝对值差值平均值（将百分比改为差值）
    abs_col = "预计物流时效-实际物流时效差值(绝对值)"
    current_abs_avg = df_current[abs_col].mean() if abs_col in df_current.columns and len(df_current) > 0 else 0
    prev_abs_avg = df_prev[abs_col].mean() if not df_prev.empty and abs_col in df_prev.columns and len(
        df_prev) > 0 else 0
    abs_change = current_abs_avg - prev_abs_avg  # 差值计算（替换百分比）
    abs_change_text = f"{'↑' if abs_change > 0 else '↓' if abs_change < 0 else '—'} {abs(abs_change):.2f} (上月: {prev_abs_avg:.2f})"
    abs_change_color = "red" if abs_change > 0 else "green" if abs_change < 0 else "gray"

    # 5. 实际差值平均值
    diff_col = "预计物流时效-实际物流时效差值"
    current_diff_avg = df_current[diff_col].mean() if diff_col in df_current.columns and len(df_current) > 0 else 0
    prev_diff_avg = df_prev[diff_col].mean() if not df_prev.empty and diff_col in df_prev.columns and len(
        df_prev) > 0 else 0
    diff_change = current_diff_avg - prev_diff_avg
    diff_change_text = f"{'↑' if diff_change > 0 else '↓' if diff_change < 0 else '—'} {abs(diff_change):.2f} (上月: {prev_diff_avg:.2f})"
    diff_change_color = "red" if diff_change > 0 else "green" if diff_change < 0 else "gray"
    # ========== 新增：6. 准时率（核心修改1） ==========
    # 当月准时率（提前/准时数 ÷ 总FBA数 × 100%）
    current_on_time_rate = (current_on_time / current_fba * 100) if current_fba > 0 else 0.0
    # 上月准时率
    prev_on_time_rate = (prev_on_time / prev_fba * 100) if prev_fba > 0 else 0.0
    # 准时率环比变化（百分点）
    on_time_rate_change = current_on_time_rate - prev_on_time_rate
    # 准时率变化文本（和其他指标样式统一）
    on_time_rate_change_text = f"{'↑' if on_time_rate_change > 0 else '↓' if on_time_rate_change < 0 else '—'} {abs(on_time_rate_change):.1f}% (上月: {prev_on_time_rate:.1f}%)"
    # 准时率变化颜色（红升绿降）
    on_time_rate_change_color = "red" if on_time_rate_change > 0 else "green" if on_time_rate_change < 0 else "gray"

    # 显示卡片（一行六列）- 改用HTML自定义样式（核心修改2：从5列改为6列）
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.markdown(f"""
        <div style='background-color: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center;'>
            <h5 style='margin: 0; color: #333;'>FBA单</h5>
            <p style='font-size: 24px; margin: 8px 0; font-weight: bold;'>{current_fba}</p>
            <p style='font-size: 14px; color: {fba_change_color}; margin: 0;'>{fba_change_text}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style='background-color: #f0f8f0; padding: 15px; border-radius: 8px; text-align: center;'>
            <h5 style='margin: 0; color: green;'>提前/准时数</h5>
            <p style='font-size: 24px; margin: 8px 0; font-weight: bold;'>{current_on_time}</p>
            <p style='font-size: 14px; color: {on_time_change_color}; margin: 0;'>{on_time_change_text}</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div style='background-color: #fff0f0; padding: 15px; border-radius: 8px; text-align: center;'>
            <h5 style='margin: 0; color: red;'>延期数</h5>
            <p style='font-size: 24px; margin: 8px 0; font-weight: bold;'>{current_delay}</p>
            <p style='font-size: 14px; color: {delay_change_color}; margin: 0;'>{delay_change_text}</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div style='background-color: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center;'>
            <h5 style='margin: 0; color: #333;'>绝对值差值均值</h5>
            <p style='font-size: 24px; margin: 8px 0; font-weight: bold;'>{current_abs_avg:.2f}</p>
            <p style='font-size: 14px; color: {abs_change_color}; margin: 0;'>{abs_change_text}</p>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
        <div style='background-color: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center;'>
            <h5 style='margin: 0; color: #333;'>实际差值均值</h5>
            <p style='font-size: 24px; margin: 8px 0; font-weight: bold;'>{current_diff_avg:.2f}</p>
            <p style='font-size: 14px; color: {diff_change_color}; margin: 0;'>{diff_change_text}</p>
        </div>
        """, unsafe_allow_html=True)

    # ========== 新增：第6列 准时率卡片（核心修改3） ==========
    with col6:
        st.markdown(f"""
        <div style='background-color: #e8f4f8; padding: 15px; border-radius: 8px; text-align: center;'>
            <h5 style='margin: 0; color: #2196f3;'>准时率</h5>
            <p style='font-size: 24px; margin: 8px 0; font-weight: bold;'>{current_on_time_rate:.1f}%</p>
            <p style='font-size: 14px; color: {on_time_rate_change_color}; margin: 0;'>{on_time_rate_change_text}</p>
        </div>
        """, unsafe_allow_html=True)

    # 生成总结文字
    summary_text = f"""
    {selected_month.replace('-', '年')}月物流时效情况：本月的FBA单有：{current_fba}单，与上个月对比{'增加' if fba_change > 0 else '减少' if fba_change < 0 else '持平'} {abs(fba_change)}单，
    其中提前/准时单有：{current_on_time}单，与上个月对比{'增加' if on_time_change > 0 else '减少' if on_time_change < 0 else '持平'} {abs(on_time_change)}单，
    延期单有：{current_delay}单，与上个月对比{'增加' if delay_change > 0 else '减少' if delay_change < 0 else '持平'} {abs(delay_change)}单，
    预计物流时效-实际物流时效差异（绝对值）为：{current_abs_avg:.2f}，与上个月对比{'增加' if abs_change > 0 else '减少' if abs_change < 0 else '持平'} {abs(abs_change):.2f}，
    预计物流时效-实际物流时效差异为：{current_diff_avg:.2f}，与上个月对比{'增加' if diff_change > 0 else '减少' if diff_change < 0 else '持平'} {abs(diff_change):.2f}。
    """

    # 差异判断
    if current_diff_avg > 0:
        summary_text += "虽然有延迟，但延迟情况不严重，整体提前！"
    else:
        summary_text += "虽然有提前，但延迟更严重，整体还是延迟的！"

    st.markdown(f"> {summary_text}")
    st.divider()

    # ---------------------- ② 当月准时率与时效偏差 ----------------------
    st.markdown("### 准时率与时效偏差分布")
    col1, col2 = st.columns(2)

    # 左：饼图（提前/准时 vs 延期）
    with col1:
        if "提前/延期" in df_current.columns and len(df_current) > 0:
            pie_data = df_current["提前/延期"].value_counts()

            # 确保颜色映射严格生效（显式指定颜色列表）
            # 提取类别并按顺序映射颜色
            categories = pie_data.index.tolist()
            colors = []
            for cat in categories:
                if cat == "提前/准时":
                    colors.append("green")
                elif cat == "延期":
                    colors.append("red")
                else:
                    colors.append("gray")  # 处理意外类别

            fig_pie = px.pie(
                values=pie_data.values,
                names=pie_data.index,
                title=f"{selected_month} 空派准时率分布",
                color=pie_data.index,  # 显式指定颜色依据
                color_discrete_sequence=colors  # 使用顺序颜色列表确保对应关系
            )
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.write("⚠️ 暂无准时率数据")

    # 右：文本直方图（提前/准时 和 延期）
    with col2:
        if diff_col in df_current.columns and len(df_current) > 0:
            # 提取并处理数据
            diff_data = df_current[diff_col].dropna()
            diff_data = diff_data.round().astype(int)  # 转换为整数天数

            # 分离提前/准时（>=0）和延期（<0）数据
            early_data = diff_data[diff_data >= 0]  # 包含0天（准时）
            delay_data = diff_data[diff_data < 0]  # 延期数据

            # 统计各天数出现次数
            early_counts = early_data.value_counts().sort_index(ascending=False)  # 从大到小排序
            delay_counts = delay_data.value_counts().sort_index()  # 从小到大排序（-7, -6...）

            # 计算最大计数（用于归一化显示长度）
            max_count = max(
                early_counts.max() if not early_counts.empty else 0,
                delay_counts.max() if not delay_counts.empty else 0
            )
            max_display_length = 20  # 最大显示字符数

            # 生成文本直方图（使用HTML设置颜色，与饼图保持一致）
            st.markdown("#### 提前/准时区间分布")
            if not early_counts.empty:
                for day, count in early_counts.items():
                    # 计算显示长度（按比例缩放）
                    display_length = int((count / max_count) * max_display_length) if max_count > 0 else 0
                    bar = "█" * display_length
                    day_label = f"+{day}天" if day > 0 else "0天"  # 0天特殊处理
                    # 绿色显示（与饼图提前/准时颜色一致）
                    st.markdown(
                        f"<div style='font-family: monospace;'><span style='display: inline-block; width: 60px;'>{day_label}</span>"
                        f"<span style='color: green;'>{bar}</span> <span> ({count})</span></div>",
                        unsafe_allow_html=True
                    )
            else:
                st.text("暂无提前/准时数据")

            st.markdown("#### 延迟区间分布")
            if not delay_counts.empty:
                for day, count in delay_counts.items():
                    display_length = int((count / max_count) * max_display_length) if max_count > 0 else 0
                    bar = "█" * display_length
                    # 红色显示（与饼图延期颜色一致）
                    st.markdown(
                        f"<div style='font-family: monospace;'><span style='display: inline-block; width: 60px;'>{day}天</span>"
                        f"<span style='color: red;'>{bar}</span> <span> ({count})</span></div>",
                        unsafe_allow_html=True
                    )
            else:
                st.text("暂无延迟数据")
        else:
            st.write("⚠️ 暂无时效偏差数据")

    st.divider()

    # ---------------------- ③ 当月空派明细表格 ----------------------
    st.markdown("### 空派明细（含平均值）")

    # 准备明细数据
    detail_cols = [
        "到货年月", "提前/延期", "FBA号", "店铺", "仓库", "货代",
        # 新增的物流阶段列（加在货代右边）
        "发货-起飞", "到港-提取", "提取-签收", "清关耗时","签收-完成上架",
        "签收-发货时间", "上架完成-发货时间",
        abs_col, diff_col
    ]
    # 过滤存在的列
    detail_cols = [col for col in detail_cols if col in df_current.columns]
    df_detail = df_current[detail_cols].copy() if len(detail_cols) > 0 else pd.DataFrame()

    if len(df_detail) > 0:
        # 按时效差值升序排序
        if diff_col in df_detail.columns:
            df_detail = df_detail.sort_values(diff_col, ascending=True)

        # 定义需要显示为整数的列
        int_cols = [
            "发货-起飞", "到港-提取", "提取-签收", "签收-完成上架","清关耗时",
            "签收-发货时间", "上架完成-发货时间"
        ]
        # 过滤存在的整数列
        int_cols = [col for col in int_cols if col in df_detail.columns]

        # 将整数列转换为无小数点格式（空值填充为0）
        for col in int_cols:
            df_detail[col] = pd.to_numeric(df_detail[col], errors='coerce').fillna(0).astype(int)

        # 计算平均值行
        avg_row = {}
        for col in detail_cols:
            if col in ["到货年月"]:
                avg_row[col] = "平均值"
            elif col in ["提前/延期", "FBA号", "店铺", "仓库", "货代"]:
                avg_row[col] = "-"
            elif col in int_cols:
                # 整数列的平均值保留两位小数
                avg_val = df_detail[col].mean()
                avg_row[col] = round(avg_val, 2)
            else:
                # 其他数值列保留两位小数
                avg_val = df_detail[col].mean() if len(df_detail) > 0 else 0
                avg_row[col] = round(avg_val, 2)


        # 格式化函数
        def format_value(val, col):
            """格式化单元格值"""
            try:
                if val == "平均值" or val == "-":
                    return val
                if col in int_cols:
                    if isinstance(val, (int, float)):
                        if val == int(val):
                            return f"{int(val)}"
                        else:
                            return f"{val:.2f}"
                elif col in [abs_col, diff_col]:
                    return f"{val:.2f}"
                return str(val)
            except:
                return str(val)


        # === 1. 解决列名不完整：换行/自适应宽度 ===
        # 处理长列名（换行显示）
        def format_colname(col):
            """列名换行处理，避免截断"""
            if len(col) > 8:
                # 按特殊字符拆分长列名
                if "-" in col:
                    return col.replace("-", "<br>-")
                elif "（" in col:
                    return col.replace("（", "<br>（")
                else:
                    # 手动换行
                    return col[:8] + "<br>" + col[8:]
            return col


        # === 2. 生成带固定行的表格（列名完整） ===
        html_content = f"""
        <style>
        /* 容器样式 */
        .table-container {{
            height: 400px;
            overflow-y: auto;
            overflow-x: auto;  /* 横向滚动，避免列名截断 */
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            margin: 10px 0;
        }}

        /* 核心：单表格 + sticky固定行 */
        .data-table {{
            width: 100%;
            min-width: max-content;  /* 确保列名完整显示 */
            border-collapse: collapse;
        }}

        /* 表头固定 + 列名完整显示 */
        .data-table thead th {{
            position: sticky;
            top: 0;
            background-color: #f8f9fa;
            font-weight: bold;
            z-index: 2;
            padding: 8px 4px;  /* 减小内边距，增加显示空间 */
            white-space: normal;  /* 允许列名换行 */
            line-height: 1.2;     /* 行高适配换行 */
            text-align: center;   /* 列名居中，更易读 */
        }}

        /* 平均值行固定（紧跟表头） */
        .avg-row td {{
            position: sticky;
            top: 60px; /* 适配换行后的表头高度 */
            background-color: #fff3cd;
            font-weight: 500;
            z-index: 1;
            text-align: center;
        }}

        /* 通用单元格样式 */
        .data-table th, .data-table td {{
            padding: 8px;
            border: 1px solid #e0e0e0;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}

        /* 数据行左对齐 */
        .data-table tbody tr td {{
            text-align: left;
        }}

        /* 高亮样式 */
        .highlight {{
            background-color: #ffcccc !important;
        }}
        </style>

        <div class="table-container">
            <table class="data-table">
                <!-- 表头（列名换行处理） -->
                <thead>
                    <tr>
                        {''.join([f'<th>{format_colname(col)}</th>' for col in detail_cols])}
                    </tr>
                </thead>
                <tbody>
                    <!-- 平均值行 -->
                    <tr class="avg-row">
                        {''.join([f'<td>{format_value(avg_row[col], col)}</td>' for col in detail_cols])}
                    </tr>
                    <!-- 数据行 -->
                    {''.join([
            '<tr>' + ''.join([
                f'<td class={"highlight" if (
                        col in (int_cols + [abs_col, diff_col])
                        and avg_row[col] not in ["-", "平均值"]
                        and pd.notna(row[col])
                        and float(row[col]) > float(avg_row[col])
                ) else ""}>{format_value(row[col], col)}</td>'
                for col in detail_cols
            ]) + '</tr>'
            for _, row in df_detail.iterrows()
        ])}
                </tbody>
            </table>
        </div>
        """

        # 渲染表格
        st.markdown(html_content, unsafe_allow_html=True)

        # === 3. 添加表格下载功能 ===
        import pandas as pd
        from io import BytesIO
        import base64

        # 构建带平均值的完整数据（用于下载）
        df_download = pd.concat([pd.DataFrame([avg_row]), df_detail], ignore_index=True)


        # 定义下载函数
        def get_table_download_link(df, filename, text):
            """生成表格下载链接"""
            # 保存为Excel（保留格式）
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='空派明细')
            output.seek(0)
            b64 = base64.b64encode(output.read()).decode()

            # 生成下载链接
            href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">{text}</a>'
            return href


        # 显示下载按钮
        st.markdown(
            get_table_download_link(
                df_download,
                f"空派明细_{selected_month}.xlsx",
                "📥 下载空派明细表格（Excel格式）"
            ),
            unsafe_allow_html=True
        )

    else:
        st.write("⚠️ 暂无明细数据")

    st.divider()

    # ---------------------- ④ 当月货代准时情况 ----------------------
    # ---------------------- 货代准时情况分析（独立版：发货-签收环节，无仓库关联） ----------------------
    st.markdown("### 货代准时情况分析（发货-签收环节）")

    # ========== 列名映射字典（根据你的实际列名修改！）==========
    COLUMN_MAPPING = {
        "货代列名": "货代",  # 改成你数据中实际的货代列名
        "货代提前延期列名": "提前/延期（货代）",  # 改成你实际的货代提前/延期列名
        "货代时效差值列名": "预计物流时效-实际物流时效差值（货代）"  # 改成你实际的货代时效差值列名
    }

    # 筛选有效数据（仅保留有货代信息的行）
    df_freight_valid = df_current[
        df_current[COLUMN_MAPPING["货代列名"]].notna() &
        (df_current[COLUMN_MAPPING["货代列名"]] != "")
        ].copy()

    if len(df_freight_valid) == 0:
        st.warning(f"{selected_month}月暂无货代相关数据")
    else:
        # ===== 列名校验：避免KeyError =====
        required_cols = [COLUMN_MAPPING["货代列名"], COLUMN_MAPPING["货代提前延期列名"],
                         COLUMN_MAPPING["货代时效差值列名"]]
        missing_cols = [col for col in required_cols if col not in df_freight_valid.columns]
        if missing_cols:
            st.error(f"缺少货代分析必要列：{missing_cols}，请检查列名是否正确！")
            st.stop()

        # ===== 1. 货代核心指标计算 =====
        freight_stats = df_freight_valid.groupby(COLUMN_MAPPING["货代列名"]).agg(
            总订单数=(COLUMN_MAPPING["货代列名"], "count"),
            提前准时订单数=(COLUMN_MAPPING["货代提前延期列名"], lambda x: len(x[x == "提前/准时"])),
            延期订单数=(COLUMN_MAPPING["货代提前延期列名"], lambda x: len(x[x == "延期"])),
            时效差值均值=(COLUMN_MAPPING["货代时效差值列名"], "mean"),
            最大延期天数=(COLUMN_MAPPING["货代时效差值列名"], lambda x: min(x.min(), 0)),  # 仅取延期负数
            最大提前天数=(COLUMN_MAPPING["货代时效差值列名"], lambda x: max(x.max(), 0))  # 仅取提前正数
        ).reset_index()

        # 重命名货代列，方便后续使用
        freight_stats.rename(columns={COLUMN_MAPPING["货代列名"]: "货代"}, inplace=True)

        # 计算衍生指标（核心）- 统一保留2位小数
        freight_stats["准时率(%)"] = round(freight_stats["提前准时订单数"] / freight_stats["总订单数"] * 100, 2)
        freight_stats["订单量占比(%)"] = round(freight_stats["总订单数"] / len(df_freight_valid) * 100, 2)
        freight_stats["延期率(%)"] = round(100 - freight_stats["准时率(%)"], 2)

        # ===== 2. 计算上月货代准时率（调整为“准时率差值”）=====
        prev_freight_valid = df_prev[
            df_prev[COLUMN_MAPPING["货代列名"]].notna() &
            (df_prev[COLUMN_MAPPING["货代列名"]] != "")
            ].copy() if not df_prev.empty else pd.DataFrame()

        if len(prev_freight_valid) > 0:
            prev_freight_stats = prev_freight_valid.groupby(COLUMN_MAPPING["货代列名"]).agg(
                上月提前准时订单数=(COLUMN_MAPPING["货代提前延期列名"], lambda x: len(x[x == "提前/准时"])),
                上月总订单数=(COLUMN_MAPPING["货代列名"], "count")
            ).reset_index()
            prev_freight_stats.rename(columns={COLUMN_MAPPING["货代列名"]: "货代"}, inplace=True)
            prev_freight_stats["上月准时率(%)"] = round(
                prev_freight_stats["上月提前准时订单数"] / prev_freight_stats["上月总订单数"] * 100, 2)
            # 合并本月&上月数据
            freight_stats = pd.merge(freight_stats, prev_freight_stats[["货代", "上月准时率(%)"]], on="货代",
                                     how="left")
            freight_stats["准时率差值(%)"] = round(
                freight_stats["准时率(%)"] - freight_stats["上月准时率(%)"].fillna(0), 2)
        else:
            freight_stats["上月准时率(%)"] = None  # 无数据时显示空
            freight_stats["准时率差值(%)"] = None

        # ===== 3. 可视化展示（双轴图 + 所有货代迷你卡片）=====
        col1, col2 = st.columns([2, 1])
        # 3.1 左：货代订单量占比 + 准时率 双轴图（核心趋势）
        with col1:
            import plotly.graph_objects as go

            fig = go.Figure()
            # 订单量占比-柱状图
            fig.add_trace(go.Bar(
                x=freight_stats["货代"],
                y=freight_stats["订单量占比(%)"],
                name="订单量占比(%)",
                yaxis="y1",
                marker_color="#4299e1",
                opacity=0.8,
                text=freight_stats["订单量占比(%)"].apply(lambda x: f"{x:.2f}%"),  # 显示2位小数
                textposition="auto"
            ))
            # 准时率-折线图
            fig.add_trace(go.Scatter(
                x=freight_stats["货代"],
                y=freight_stats["准时率(%)"],
                name="准时率(%)",
                yaxis="y2",
                marker_color="#e53e3e",
                mode="lines+markers+text",
                line=dict(width=3),
                marker=dict(size=8),
                text=freight_stats["准时率(%)"].apply(lambda x: f"{x:.2f}%"),  # 显示2位小数
                textposition="top center"
            ))
            # 图表样式配置
            fig.update_layout(
                title=f"{selected_month} 货代订单量占比 & 准时率对比",
                yaxis=dict(title="订单量占比(%)", side="left", range=[0, 100], color="#4299e1"),
                yaxis2=dict(title="准时率(%)", side="right", overlaying="y", range=[0, 100], color="#e53e3e"),
                xaxis=dict(title="货代名称", tickangle=0),
                legend=dict(x=0.02, y=0.98, bordercolor="#eee", borderwidth=1),
                height=400,
                plot_bgcolor="#ffffff"
            )
            st.plotly_chart(fig, use_container_width=True)

        # 3.2 右：所有货代核心表现迷你卡片（适配3-4个货代，颜色分级）
        with col2:
            st.markdown("#### 货代核心表现")
            for _, row in freight_stats.iterrows():
                # 准时率颜色分级：优质≥90% | 合格80-90% | 异常<80%
                if row["准时率(%)"] >= 90:
                    card_bg = "#f0f8f0"
                    rate_color = "#2e7d32"
                    tag = "优质"
                elif row["准时率(%)"] >= 80:
                    card_bg = "#fff8e1"
                    rate_color = "#ff9800"
                    tag = "合格"
                else:
                    card_bg = "#fff0f0"
                    rate_color = "#c62828"
                    tag = "异常"
                # 准时率差值样式
                diff_val = row["准时率差值(%)"]
                if pd.notna(diff_val):
                    if diff_val > 0:
                        diff_text = f"↑{diff_val:.2f}%"
                        diff_color = "#2e7d32"
                    elif diff_val < 0:
                        diff_text = f"↓{abs(diff_val):.2f}%"
                        diff_color = "#c62828"
                    else:
                        diff_text = "—"
                        diff_color = "#757575"
                    # 上月准时率显示（无数据时隐藏）
                    prev_rate_text = f"（上月{row['上月准时率(%)']:.2f}%）" if pd.notna(row["上月准时率(%)"]) else ""
                else:
                    diff_text = "—"
                    diff_color = "#757575"
                    prev_rate_text = ""
                # 生成货代迷你卡片
                st.markdown(f"""
                <div style='background-color: {card_bg}; padding: 12px; border-radius: 8px; margin-bottom: 12px; border-left: 4px solid {rate_color};'>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <p style='margin: 0; font-weight: bold; font-size: 16px;'>{row['货代']}</p>
                        <span style='font-size: 12px; padding: 2px 6px; border-radius: 12px; background: {rate_color}; color: white;'>{tag}</span>
                    </div>
                    <p style='margin: 6px 0 0; font-size: 14px;'>
                        准时率：<span style='color: {rate_color}; font-weight: bold; font-size: 18px;'>{row['准时率(%)']:.2f}%</span>
                    </p>
                    <p style='margin: 4px 0 0; font-size: 12px; color: #666;'>订单：{row['总订单数']}单（{row['订单量占比(%)']:.2f}%）</p>
                    <p style='margin: 4px 0 0; font-size: 12px; color: #666;'>差值：<span style='color: {diff_color}; font-weight: bold;'>{diff_text}</span> {prev_rate_text}</p>
                    <p style='margin: 4px 0 0; font-size: 12px; color: #666;'>最大延期：{abs(row['最大延期天数'])}天</p>
                </div>
                """, unsafe_allow_html=True)

        # ===== 4. 货代详细时效指标表（带上月差值对比+兼容Streamlit样式）=====
        st.markdown("#### 货代详细时效指标表")

        # ---------------------- 计算上月货代订单类指标 ----------------------
        prev_order_stats = pd.DataFrame()
        if len(prev_freight_valid) > 0:
            prev_order_stats = prev_freight_valid.groupby(COLUMN_MAPPING["货代列名"]).agg(
                上月总订单数=(COLUMN_MAPPING["货代列名"], "count"),
                上月提前准时订单数=(COLUMN_MAPPING["货代提前延期列名"], lambda x: len(x[x == "提前/准时"])),
                上月延期订单数=(COLUMN_MAPPING["货代提前延期列名"], lambda x: len(x[x == "延期"]))
            ).reset_index()
            prev_order_stats.rename(columns={COLUMN_MAPPING["货代列名"]: "货代"}, inplace=True)
            freight_stats = pd.merge(freight_stats, prev_order_stats, on="货代", how="left")
        else:
            freight_stats["上月总订单数"] = None
            freight_stats["上月提前准时订单数"] = None
            freight_stats["上月延期订单数"] = None

        # ---------------------- 格式化订单数列（纯文本兼容版） ----------------------
        display_cols = [
            "货代", "总订单数", "订单量占比(%)", "提前准时订单数", "延期订单数", "延期率(%)",
            "准时率(%)", "上月准时率(%)", "准时率差值(%)",
            "时效差值均值", "最大提前天数", "最大延期天数"
        ]
        freight_display = freight_stats[display_cols].copy()


        # 自定义格式化函数（纯文本，用[]包裹上月信息，视觉区分）
        def format_order_col(current_val, prev_val):
            """
            纯文本格式化：本月数 [差值 上月数]
            - 上月信息用[]包裹，视觉上弱化
            - 差值带正负号，无上月数据时只显示本月数
            """
            if pd.notna(prev_val):
                diff = current_val - prev_val
                diff_sign = "+" if diff > 0 else "" if diff == 0 else "-"
                diff_abs = abs(diff)
                # 用[]包裹上月信息，通过空格/符号实现视觉层次
                return f"{current_val}  [{diff_sign}{diff_abs} 上月{prev_val}]"
            else:
                return f"{current_val}"


        # 应用格式化（直接操作freight_stats的原始数值）
        freight_display["总订单数"] = freight_stats.apply(
            lambda x: format_order_col(x["总订单数"], x["上月总订单数"]), axis=1
        )
        freight_display["提前准时订单数"] = freight_stats.apply(
            lambda x: format_order_col(x["提前准时订单数"], x["上月提前准时订单数"]), axis=1
        )
        freight_display["延期订单数"] = freight_stats.apply(
            lambda x: format_order_col(x["延期订单数"], x["上月延期订单数"]), axis=1
        )

        # 其他数值格式化
        freight_display["时效差值均值"] = round(freight_display["时效差值均值"], 2)
        freight_display["最大延期天数"] = freight_display["最大延期天数"].apply(
            lambda x: f"{abs(x)}天" if x < 0 else "0天")
        freight_display["最大提前天数"] = freight_display["最大提前天数"].apply(lambda x: f"{x}天" if x > 0 else "0天")

        # 百分比列格式化
        for col in ["订单量占比(%)", "延期率(%)", "准时率(%)", "上月准时率(%)", "准时率差值(%)"]:
            freight_display[col] = freight_display[col].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "")


        # ---------------------- 表格高亮规则 ----------------------
        def highlight_freight(row):
            styles = [""] * len(row)
            # 准时率差值为负标红
            if row["准时率差值(%)"] and isinstance(row["准时率差值(%)"], str) and float(
                    row["准时率差值(%)"].replace("%", "")) < 0:
                styles[display_cols.index(
                    "准时率差值(%)")] = "background-color: #fff5f5; color: #c62828; font-weight: bold;"
            # 延期率>20%标红
            if row["延期率(%)"] and isinstance(row["延期率(%)"], str) and float(row["延期率(%)"].replace("%", "")) > 20:
                styles[
                    display_cols.index("延期率(%)")] = "background-color: #fff5f5; color: #c62828; font-weight: bold;"
            # 准时率<80%标红
            if row["准时率(%)"] and isinstance(row["准时率(%)"], str) and float(row["准时率(%)"].replace("%", "")) < 80:
                styles[
                    display_cols.index("准时率(%)")] = "background-color: #fff5f5; color: #c62828; font-weight: bold;"
            return styles


        # ---------------------- 展示表格（移除unsafe_allow_html，兼容Streamlit） ----------------------
        styled_table = freight_display.style.apply(highlight_freight, axis=1)
        st.dataframe(
            styled_table,
            use_container_width=True,
            hide_index=True  # 移除unsafe_allow_html参数，避免TypeError
        )

        # ===== 5. 数据下载功能 =====
        # 下载数据保留原始数值（非格式化）
        download_data = freight_stats.copy()
        csv_data = download_data.to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            label="📥 下载货代分析完整数据",
            data=csv_data,
            file_name=f"{selected_month}_货代准时率分析数据.csv",
            mime="text/csv",
            key="freight_data_download"
        )
    # ===== 6. 货代当月表现总结文字（修复重复问题） =====
    st.markdown("### 货代当月表现总结")

    # 每次运行都重新创建空列表（避免追加重复内容）
    summary_paragraphs = []
    for _, row in freight_stats.iterrows():
        # 基础信息提取
        freight_name = row["货代"]
        order_count = row["总订单数"]
        order_ratio = row["订单量占比(%)"]
        on_time_rate = row["准时率(%)"]
        max_delay = abs(row["最大延期天数"])
        prev_rate = row["上月准时率(%)"]
        diff_val = row["准时率差值(%)"]

        # 评级判断+颜色
        if on_time_rate >= 90:
            level_tag = "【优质】"
            level_color = "#2e7d32"
            level_desc = "准时率表现优秀"
        elif on_time_rate >= 80:
            level_tag = "【合格】"
            level_color = "#ff9800"
            level_desc = "准时率表现达标"
        else:
            level_tag = "【异常】"
            level_color = "#c62828"
            level_desc = "准时率表现不达标，需重点关注"

        # 差值描述（修复无上月数据）
        if pd.notna(prev_rate):
            if diff_val > 0:
                diff_desc = f"较上月提升{diff_val:.2f}个百分点"
            elif diff_val < 0:
                diff_desc = f"较上月下降{abs(diff_val):.2f}个百分点"
            else:
                diff_desc = "与上月持平"
        else:
            diff_desc = "无上月数据对比"

        # 延期描述
        delay_desc = "全程无延期订单" if max_delay == 0 else f"最大延期天数为{max_delay}天"

        # 生成单条总结（精简HTML，避免冗余标签）
        summary = f"""
        - <b>{freight_name} <span style='color:{level_color};'>{level_tag}</span></b>：
          本月承接{order_count}单（占总订单量{order_ratio:.2f}%），{level_desc}，准时率为{on_time_rate:.2f}%，{diff_desc}，{delay_desc}。
        """
        summary_paragraphs.append(summary)

    # 清空重复内容后，只渲染一次
    st.markdown("\n".join(summary_paragraphs), unsafe_allow_html=True)

    # ---------------------- ⑤ 当月仓库准时情况 ----------------------
    # ---------------------- 仓库准时情况分析（签收-完成上架环节） ----------------------
    st.markdown("### 仓库准时情况分析（签收-完成上架环节）")

    # ========== 列名映射字典（根据你的实际列名修改！）==========
    WAREHOUSE_COLUMN_MAPPING = {
        "仓库列名": "仓库",  # 改成你数据中实际的仓库列名
        "签收上架时长列名": "签收-完成上架",  # 改成你实际的「签收-完成上架」时长列名
        # 注：「提前/延期（仓库）」列会自动计算，无需手动映射
    }

    # 筛选有效数据（仅保留有仓库信息+签收上架时长的行）
    df_warehouse_valid = df_current[
        (df_current[WAREHOUSE_COLUMN_MAPPING["仓库列名"]].notna() &
         (df_current[WAREHOUSE_COLUMN_MAPPING["仓库列名"]] != "")) &
        (df_current[WAREHOUSE_COLUMN_MAPPING["签收上架时长列名"]].notna())
        ].copy()

    if len(df_warehouse_valid) == 0:
        st.warning(f"{selected_month}月暂无仓库相关数据")
    else:
        # ===== 列名校验：避免KeyError =====
        required_cols = [WAREHOUSE_COLUMN_MAPPING["仓库列名"], WAREHOUSE_COLUMN_MAPPING["签收上架时长列名"]]
        missing_cols = [col for col in required_cols if col not in df_warehouse_valid.columns]
        if missing_cols:
            st.error(f"缺少仓库分析必要列：{missing_cols}，请检查列名是否正确！")
            st.stop()

        # ===== 1. 核心计算：自动生成「提前/延期（仓库）」列 =====
        # 规则：时长≤3天=提前/准时，>3天=延期
        df_warehouse_valid["提前/延期（仓库）"] = df_warehouse_valid[WAREHOUSE_COLUMN_MAPPING["签收上架时长列名"]].apply(
            lambda x: "提前/准时" if x <= 3 else "延期"
        )

        # ===== 2. 仓库核心指标计算 =====
        warehouse_stats = df_warehouse_valid.groupby(WAREHOUSE_COLUMN_MAPPING["仓库列名"]).agg(
            总订单数=(WAREHOUSE_COLUMN_MAPPING["仓库列名"], "count"),
            提前准时订单数=("提前/延期（仓库）", lambda x: len(x[x == "提前/准时"])),
            延期订单数=("提前/延期（仓库）", lambda x: len(x[x == "延期"])),
            签收上架时长均值=(WAREHOUSE_COLUMN_MAPPING["签收上架时长列名"], "mean"),
            签收上架时长中位数=(WAREHOUSE_COLUMN_MAPPING["签收上架时长列名"], "median"),
            最长上架时长=(WAREHOUSE_COLUMN_MAPPING["签收上架时长列名"], "max"),
            最短上架时长=(WAREHOUSE_COLUMN_MAPPING["签收上架时长列名"], "min")
        ).reset_index()

        # 重命名仓库列，方便后续使用
        warehouse_stats.rename(columns={WAREHOUSE_COLUMN_MAPPING["仓库列名"]: "仓库"}, inplace=True)

        # 计算衍生指标（核心）- 统一保留2位小数
        warehouse_stats["准时率(%)"] = round(warehouse_stats["提前准时订单数"] / warehouse_stats["总订单数"] * 100, 2)
        warehouse_stats["订单量占比(%)"] = round(warehouse_stats["总订单数"] / len(df_warehouse_valid) * 100, 2)
        warehouse_stats["延期率(%)"] = round(100 - warehouse_stats["准时率(%)"], 2)

        # ===== 3. 计算上月仓库指标（环比/差值分析）=====
        # 处理上月数据
        prev_warehouse_valid = df_prev[
            (df_prev[WAREHOUSE_COLUMN_MAPPING["仓库列名"]].notna() &
             (df_prev[WAREHOUSE_COLUMN_MAPPING["仓库列名"]] != "")) &
            (df_prev[WAREHOUSE_COLUMN_MAPPING["签收上架时长列名"]].notna())
            ].copy() if not df_prev.empty else pd.DataFrame()

        if len(prev_warehouse_valid) > 0:
            # 上月数据生成「提前/延期（仓库）」列
            prev_warehouse_valid["提前/延期（仓库）"] = prev_warehouse_valid[
                WAREHOUSE_COLUMN_MAPPING["签收上架时长列名"]].apply(
                lambda x: "提前/准时" if x <= 3 else "延期"
            )
            # 计算上月仓库核心指标
            prev_warehouse_stats = prev_warehouse_valid.groupby(WAREHOUSE_COLUMN_MAPPING["仓库列名"]).agg(
                上月提前准时订单数=("提前/延期（仓库）", lambda x: len(x[x == "提前/准时"])),
                上月总订单数=(WAREHOUSE_COLUMN_MAPPING["仓库列名"], "count")
            ).reset_index()
            prev_warehouse_stats.rename(columns={WAREHOUSE_COLUMN_MAPPING["仓库列名"]: "仓库"}, inplace=True)
            prev_warehouse_stats["上月准时率(%)"] = round(
                prev_warehouse_stats["上月提前准时订单数"] / prev_warehouse_stats["上月总订单数"] * 100, 2)

            # 合并本月&上月数据
            warehouse_stats = pd.merge(warehouse_stats, prev_warehouse_stats[["仓库", "上月准时率(%)"]], on="仓库",
                                       how="left")
            warehouse_stats["准时率差值(%)"] = round(
                warehouse_stats["准时率(%)"] - warehouse_stats["上月准时率(%)"].fillna(0), 2)

            # 计算上月订单数（用于表格差值展示）
            prev_order_stats = prev_warehouse_valid.groupby(WAREHOUSE_COLUMN_MAPPING["仓库列名"]).agg(
                上月总订单数=(WAREHOUSE_COLUMN_MAPPING["仓库列名"], "count"),
                上月提前准时订单数=("提前/延期（仓库）", lambda x: len(x[x == "提前/准时"])),
                上月延期订单数=("提前/延期（仓库）", lambda x: len(x[x == "延期"]))
            ).reset_index()
            prev_order_stats.rename(columns={WAREHOUSE_COLUMN_MAPPING["仓库列名"]: "仓库"}, inplace=True)
            warehouse_stats = pd.merge(warehouse_stats, prev_order_stats, on="仓库", how="left")
        else:
            # 无上月数据时填充空值
            warehouse_stats["上月准时率(%)"] = None
            warehouse_stats["准时率差值(%)"] = None
            warehouse_stats["上月总订单数"] = None
            warehouse_stats["上月提前准时订单数"] = None
            warehouse_stats["上月延期订单数"] = None

        # ===== 4. 可视化展示（双轴图 + 所有仓库迷你卡片）=====
        col1, col2 = st.columns([2, 1])
        # 4.1 左：仓库订单量占比 + 准时率 双轴图
        with col1:
            import plotly.graph_objects as go

            fig = go.Figure()
            # 订单量占比-柱状图
            fig.add_trace(go.Bar(
                x=warehouse_stats["仓库"],
                y=warehouse_stats["订单量占比(%)"],
                name="订单量占比(%)",
                yaxis="y1",
                marker_color="#9f7aea",  # 紫色（和货代的蓝色区分）
                opacity=0.8,
                text=warehouse_stats["订单量占比(%)"].apply(lambda x: f"{x:.2f}%"),
                textposition="auto"
            ))
            # 准时率-折线图
            fig.add_trace(go.Scatter(
                x=warehouse_stats["仓库"],
                y=warehouse_stats["准时率(%)"],
                name="准时率(%)",
                yaxis="y2",
                marker_color="#38b2ac",  # 青绿色（和货代的红色区分）
                mode="lines+markers+text",
                line=dict(width=3),
                marker=dict(size=8),
                text=warehouse_stats["准时率(%)"].apply(lambda x: f"{x:.2f}%"),
                textposition="top center"
            ))
            # 图表样式配置
            fig.update_layout(
                title=f"{selected_month} 仓库订单量占比 & 准时率对比（签收-上架）",
                yaxis=dict(title="订单量占比(%)", side="left", range=[0, 100], color="#9f7aea"),
                yaxis2=dict(title="准时率(%)", side="right", overlaying="y", range=[0, 100], color="#38b2ac"),
                xaxis=dict(title="仓库名称", tickangle=0),
                legend=dict(x=0.02, y=0.98, bordercolor="#eee", borderwidth=1),
                height=400,
                plot_bgcolor="#ffffff"
            )
            st.plotly_chart(fig, use_container_width=True)

        # 4.2 右：所有仓库核心表现迷你卡片
        with col2:
            st.markdown("#### 仓库核心表现")
            for _, row in warehouse_stats.iterrows():
                # 准时率颜色分级：优质≥90% | 合格80-90% | 异常<80%
                if row["准时率(%)"] >= 90:
                    card_bg = "#f0f8f0"
                    rate_color = "#2e7d32"
                    tag = "优质"
                    level_desc = "准时率表现优秀"
                elif row["准时率(%)"] >= 80:
                    card_bg = "#fff8e1"
                    rate_color = "#ff9800"
                    tag = "合格"
                    level_desc = "准时率表现达标"
                else:
                    card_bg = "#fff0f0"
                    rate_color = "#c62828"
                    tag = "异常"
                    level_desc = "准时率表现不达标，需重点关注"

                # 准时率差值样式
                diff_val = row["准时率差值(%)"]
                prev_rate = row["上月准时率(%)"]
                if pd.notna(prev_rate):
                    if diff_val > 0:
                        diff_text = f"↑{diff_val:.2f}%"
                        diff_color = "#2e7d32"
                    elif diff_val < 0:
                        diff_text = f"↓{abs(diff_val):.2f}%"
                        diff_color = "#c62828"
                    else:
                        diff_text = "—"
                        diff_color = "#757575"
                    prev_rate_text = f"（上月{prev_rate:.2f}%）"
                else:
                    diff_text = "—"
                    diff_color = "#757575"
                    prev_rate_text = ""

                # 生成仓库迷你卡片
                st.markdown(f"""
                <div style='background-color: {card_bg}; padding: 12px; border-radius: 8px; margin-bottom: 12px; border-left: 4px solid {rate_color};'>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <p style='margin: 0; font-weight: bold; font-size: 16px;'>{row['仓库']}</p>
                        <span style='font-size: 12px; padding: 2px 6px; border-radius: 12px; background: {rate_color}; color: white;'>{tag}</span>
                    </div>
                    <p style='margin: 6px 0 0; font-size: 14px;'>
                        准时率：<span style='color: {rate_color}; font-weight: bold; font-size: 18px;'>{row['准时率(%)']:.2f}%</span>
                    </p>
                    <p style='margin: 4px 0 0; font-size: 12px; color: #666;'>订单：{row['总订单数']}单（{row['订单量占比(%)']:.2f}%）</p>
                    <p style='margin: 4px 0 0; font-size: 12px; color: #666;'>差值：<span style='color: {diff_color}; font-weight: bold;'>{diff_text}</span> {prev_rate_text}</p>
                    <p style='margin: 4px 0 0; font-size: 12px; color: #666;'>最长上架时长：{row['最长上架时长']:.1f}天</p>
                </div>
                """, unsafe_allow_html=True)

        # ===== 5. 仓库详细时效指标表（带上月差值对比）=====
        st.markdown("#### 仓库详细时效指标表")
        display_cols = [
            "仓库", "总订单数", "订单量占比(%)", "提前准时订单数", "延期订单数", "延期率(%)",
            "准时率(%)", "上月准时率(%)", "准时率差值(%)",
            "签收上架时长均值", "最短上架时长", "最长上架时长"
        ]
        warehouse_display = warehouse_stats[display_cols].copy()


        # 自定义格式化函数（纯文本，[]包裹上月信息）
        def format_order_col(current_val, prev_val):
            if pd.notna(prev_val):
                diff = current_val - prev_val
                diff_sign = "+" if diff > 0 else "" if diff == 0 else "-"
                diff_abs = abs(diff)
                return f"{current_val}  [{diff_sign}{diff_abs} 上月{prev_val}]"
            else:
                return f"{current_val}"


        # 应用订单数列格式化
        warehouse_display["总订单数"] = warehouse_stats.apply(
            lambda x: format_order_col(x["总订单数"], x["上月总订单数"]), axis=1
        )
        warehouse_display["提前准时订单数"] = warehouse_stats.apply(
            lambda x: format_order_col(x["提前准时订单数"], x["上月提前准时订单数"]), axis=1
        )
        warehouse_display["延期订单数"] = warehouse_stats.apply(
            lambda x: format_order_col(x["延期订单数"], x["上月延期订单数"]), axis=1
        )

        # 其他数值格式化
        warehouse_display["签收上架时长均值"] = round(warehouse_display["签收上架时长均值"], 2)
        warehouse_display["最短上架时长"] = warehouse_display["最短上架时长"].apply(lambda x: f"{x:.1f}天")
        warehouse_display["最长上架时长"] = warehouse_display["最长上架时长"].apply(lambda x: f"{x:.1f}天")

        # 百分比列格式化
        for col in ["订单量占比(%)", "延期率(%)", "准时率(%)", "上月准时率(%)", "准时率差值(%)"]:
            warehouse_display[col] = warehouse_display[col].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "")


        # 表格高亮规则
        def highlight_warehouse(row):
            styles = [""] * len(row)
            # 准时率差值为负标红
            if row["准时率差值(%)"] and isinstance(row["准时率差值(%)"], str) and float(
                    row["准时率差值(%)"].replace("%", "")) < 0:
                styles[display_cols.index(
                    "准时率差值(%)")] = "background-color: #fff5f5; color: #c62828; font-weight: bold;"
            # 延期率>20%标红
            if row["延期率(%)"] and isinstance(row["延期率(%)"], str) and float(row["延期率(%)"].replace("%", "")) > 20:
                styles[
                    display_cols.index("延期率(%)")] = "background-color: #fff5f5; color: #c62828; font-weight: bold;"
            # 准时率<80%标红
            if row["准时率(%)"] and isinstance(row["准时率(%)"], str) and float(row["准时率(%)"].replace("%", "")) < 80:
                styles[
                    display_cols.index("准时率(%)")] = "background-color: #fff5f5; color: #c62828; font-weight: bold;"
            return styles


        # 展示表格
        styled_table = warehouse_display.style.apply(highlight_warehouse, axis=1)
        st.dataframe(
            styled_table,
            use_container_width=True,
            hide_index=True
        )

        # ===== 6. 仓库当月表现总结文字（含评级+颜色）=====
        st.markdown("### 仓库当月表现总结")
        summary_paragraphs = []
        for _, row in warehouse_stats.iterrows():
            # 基础信息
            warehouse_name = row["仓库"]
            order_count = row["总订单数"]
            order_ratio = row["订单量占比(%)"]
            on_time_rate = row["准时率(%)"]
            max_duration = row["最长上架时长"]
            prev_rate = row["上月准时率(%)"]
            diff_val = row["准时率差值(%)"]

            # 评级+颜色
            if on_time_rate >= 90:
                level_tag = "【优质】"
                level_color = "#2e7d32"
                level_desc = "准时率表现优秀"
            elif on_time_rate >= 80:
                level_tag = "【合格】"
                level_color = "#ff9800"
                level_desc = "准时率表现达标"
            else:
                level_tag = "【异常】"
                level_color = "#c62828"
                level_desc = "准时率表现不达标，需重点关注"

            # 差值描述
            if pd.notna(prev_rate):
                if diff_val > 0:
                    diff_desc = f"较上月提升{diff_val:.2f}个百分点"
                elif diff_val < 0:
                    diff_desc = f"较上月下降{abs(diff_val):.2f}个百分点"
                else:
                    diff_desc = "与上月持平"
            else:
                diff_desc = "无上月数据对比"

            # 上架时长描述
            duration_desc = f"最长签收-上架时长为{max_duration:.1f}天"

            # 生成总结
            summary = f"""
            - <b>{warehouse_name} <span style='color:{level_color};'>{level_tag}</span></b>：
              本月承接{order_count}单（占总订单量{order_ratio:.2f}%），{level_desc}，准时率为{on_time_rate:.2f}%，{diff_desc}，{duration_desc}。
            """
            summary_paragraphs.append(summary)

        st.markdown("\n".join(summary_paragraphs), unsafe_allow_html=True)

        # ===== 7. 数据下载功能 =====
        csv_data = warehouse_stats.to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            label="📥 下载仓库分析完整数据",
            data=csv_data,
            file_name=f"{selected_month}_仓库准时率分析数据.csv",
            mime="text/csv",
            key="warehouse_data_download"
        )

    # ---------------------- 不同月份整体趋势分析（总订单+准时率） ----------------------
    # ---------------------- 不同月份整体趋势分析（总订单+准时率） ----------------------
    st.markdown("## 📈 不同月份整体趋势分析")
    st.divider()

    # ===== 1. 数据预处理（按到货年月聚合）=====
    required_cols = ["到货年月", "FBA号", "提前/延期"]
    missing_cols = [col for col in required_cols if col not in df_air.columns]
    if missing_cols:
        st.error(f"缺少月度分析必要列：{missing_cols}，请检查数据列名！")
    else:
        # 按到货年月分组计算核心指标
        monthly_stats = df_air.groupby("到货年月").agg(
            总订单数=("FBA号", "count"),
            提前准时订单数=("提前/延期", lambda x: len(x[x == "提前/准时"])),
            延期订单数=("提前/延期", lambda x: len(x[x == "延期"]))
        ).reset_index()

        # 计算准时率（保留2位小数）
        monthly_stats["准时率(%)"] = round(monthly_stats["提前准时订单数"] / monthly_stats["总订单数"] * 100, 2)

        # 生成中文月份标签（如：2026年1月）
        monthly_stats["年月排序"] = pd.to_datetime(monthly_stats["到货年月"] + "-01")
        monthly_stats["中文月份"] = monthly_stats["年月排序"].dt.strftime("%Y年%m月")
        # 按时间正序排序（图表从左到右时间递增）
        monthly_stats = monthly_stats.sort_values("年月排序", ascending=True).reset_index(drop=True)

        # 计算环比变化（总订单数、准时率）
        monthly_stats["总订单数环比变化"] = monthly_stats["总订单数"].diff(1)
        monthly_stats["准时率环比变化(百分点)"] = monthly_stats["准时率(%)"].diff(1)
        # 填充空值
        monthly_stats["总订单数环比变化"] = monthly_stats["总订单数环比变化"].fillna(0)
        monthly_stats["准时率环比变化(百分点)"] = monthly_stats["准时率环比变化(百分点)"].fillna(0)

        if len(monthly_stats) == 0:
            st.warning("暂无跨月份数据可分析")
        else:
            # ===== 2. 筛选器：双下拉框时间范围选择 =====
            st.markdown("### 筛选条件")
            col_start, col_end = st.columns(2)
            with col_start:
                start_month = st.selectbox(
                    "开始月份",
                    options=monthly_stats["中文月份"].tolist(),
                    index=0,
                    key="start_month"
                )
            with col_end:
                end_month = st.selectbox(
                    "结束月份",
                    options=monthly_stats["中文月份"].tolist(),
                    index=len(monthly_stats) - 1,
                    key="end_month"
                )

            # 转换回原始年月格式用于筛选
            start_ym = monthly_stats[monthly_stats["中文月份"] == start_month]["到货年月"].iloc[0]
            end_ym = monthly_stats[monthly_stats["中文月份"] == end_month]["到货年月"].iloc[0]

            # 筛选数据并保持时间正序
            df_filtered = monthly_stats[
                (monthly_stats["到货年月"] >= start_ym) &
                (monthly_stats["到货年月"] <= end_ym)
                ].copy()
            df_filtered = df_filtered.sort_values("年月排序", ascending=True).reset_index(drop=True)

            # ===== 3. 计算平均准时率（用于红色虚线）=====
            avg_on_time_rate = df_filtered["准时率(%)"].mean()

            # ===== 4. 双轴趋势图（中文X轴+平均准时率虚线）=====
            st.markdown("### 月度订单数&准时率趋势")
            import plotly.graph_objects as go

            fig = go.Figure()

            # 左轴：柱状图（总订单数、提前准时订单数、延期订单数）
            fig.add_trace(go.Bar(
                x=df_filtered["中文月份"],
                y=df_filtered["总订单数"],
                name="总订单数",
                yaxis="y1",
                marker_color="#4299e1",
                opacity=0.8
            ))
            fig.add_trace(go.Bar(
                x=df_filtered["中文月份"],
                y=df_filtered["提前准时订单数"],
                name="提前/准时订单数",
                yaxis="y1",
                marker_color="#48bb78",
                opacity=0.8
            ))
            fig.add_trace(go.Bar(
                x=df_filtered["中文月份"],
                y=df_filtered["延期订单数"],
                name="延期订单数",
                yaxis="y1",
                marker_color="#e53e3e",
                opacity=0.8
            ))

            # 右轴：折线图（准时率）
            fig.add_trace(go.Scatter(
                x=df_filtered["中文月份"],
                y=df_filtered["准时率(%)"],
                name="准时率(%)",
                yaxis="y2",
                marker_color="#9f7aea",
                mode="lines+markers+text",
                line=dict(width=3),
                marker=dict(size=8),
                text=df_filtered["准时率(%)"].apply(lambda x: f"{x:.2f}%"),
                textposition="top center"
            ))

            # 新增：平均准时率红色虚线
            fig.add_trace(go.Scatter(
                x=df_filtered["中文月份"],
                y=[avg_on_time_rate] * len(df_filtered),
                name=f"平均准时率: {avg_on_time_rate:.2f}%",
                yaxis="y2",
                mode="lines",
                line=dict(color="#ff0000", dash="dash", width=2),
                hoverinfo="name+y"
            ))

            # 图表配置
            fig.update_layout(
                title="月度总订单数/提前准时订单数/延期订单数 & 准时率趋势",
                yaxis=dict(title="订单数", side="left", range=[0, max(df_filtered["总订单数"]) * 1.2]),
                yaxis2=dict(title="准时率(%)", side="right", overlaying="y", range=[0, 100]),
                xaxis=dict(title="到货年月", tickangle=45),
                legend=dict(x=0.02, y=0.98, bordercolor="#eee", borderwidth=1),
                height=450,
                plot_bgcolor="#ffffff",
                barmode="group"
            )
            st.plotly_chart(fig, use_container_width=True)

            # ===== 5. 月度明细表格（倒序排列）=====
            st.markdown("### 月度核心指标明细（倒序排列）")
            # 按倒序展示表格（最新月份在最前）
            df_display = df_filtered.sort_values("年月排序", ascending=False).reset_index(drop=True)
            display_cols = [
                "中文月份", "总订单数", "总订单数环比变化", "提前准时订单数", "延期订单数",
                "准时率(%)", "准时率环比变化(百分点)"
            ]
            df_display = df_display[display_cols].copy()

            # 格式化环比变化（带正负号）
            df_display["总订单数环比变化"] = df_display["总订单数环比变化"].apply(
                lambda x: f"+{int(x)}" if x > 0 else f"{int(x)}" if x < 0 else "0"
            )
            df_display["准时率环比变化(百分点)"] = df_display["准时率环比变化(百分点)"].apply(
                lambda x: f"+{x:.2f}" if x > 0 else f"{x:.2f}" if x < 0 else "0.00"
            )


            # 表格高亮规则（准时率<80%标红，环比下降标红）
            def highlight_monthly(row):
                styles = [""] * len(row)
                # 准时率<80%标红
                if float(row["准时率(%)"]) < 80:
                    styles[display_cols.index(
                        "准时率(%)")] = "background-color: #fff5f5; color: #c62828; font-weight: bold;"
                # 总订单数环比下降标红
                if row["总订单数环比变化"].startswith("-"):
                    styles[display_cols.index(
                        "总订单数环比变化")] = "background-color: #fff5f5; color: #c62828; font-weight: bold;"
                # 准时率环比下降标红
                if row["准时率环比变化(百分点)"].startswith("-"):
                    styles[display_cols.index(
                        "准时率环比变化(百分点)")] = "background-color: #fff5f5; color: #c62828; font-weight: bold;"
                return styles


            # 展示表格
            styled_table = df_display.style.apply(highlight_monthly, axis=1)
            st.dataframe(
                styled_table,
                use_container_width=True,
                hide_index=True
            )

            # ===== 6. 数据下载 =====
            csv_data = monthly_stats.to_csv(index=False, encoding="utf-8-sig")
            st.download_button(
                label="📥 下载所有月度整体数据",
                data=csv_data,
                file_name="月度整体趋势分析数据.csv",
                mime="text/csv",
                key="monthly_trend_download"
            )

            # ===== 7. 整体趋势总结 =====
            st.markdown("### 整体趋势总结")
            latest_month = df_filtered.iloc[-1]["中文月份"] if len(df_filtered) > 0 else ""
            if latest_month:
                latest_total = df_filtered.iloc[-1]["总订单数"]
                latest_on_time = df_filtered.iloc[-1]["提前准时订单数"]
                latest_delay = df_filtered.iloc[-1]["延期订单数"]
                latest_rate = df_filtered.iloc[-1]["准时率(%)"]
                prev_month = df_filtered.iloc[-2]["中文月份"] if len(df_filtered) > 1 else None

                summary = f"最新{latest_month}整体表现：总订单数{latest_total}单，其中提前/准时订单{latest_on_time}单，延期订单{latest_delay}单，准时率{latest_rate:.2f}%。"

                if prev_month:
                    prev_total = df_filtered.iloc[-2]["总订单数"]
                    prev_rate = df_filtered.iloc[-2]["准时率(%)"]
                    total_change = latest_total - prev_total
                    rate_change = latest_rate - prev_rate
                    summary += f" 与{prev_month}相比，总订单数{'增加' if total_change > 0 else '减少' if total_change < 0 else '持平'} {abs(total_change)}单，准时率{'提升' if rate_change > 0 else '下降' if rate_change < 0 else '持平'} {abs(rate_change):.2f}个百分点。"

                # 趋势判断
                if len(df_filtered) >= 3:
                    rate_trend = df_filtered["准时率(%)"].tail(3).tolist()
                    if rate_trend[2] > rate_trend[1] > rate_trend[0]:
                        summary += f" 近{len(df_filtered)}个月准时率呈上升趋势，整体表现向好！"
                    elif rate_trend[2] < rate_trend[1] < rate_trend[0]:
                        summary += f" 近{len(df_filtered)}个月准时率呈下降趋势，需重点关注延期问题！"
                    else:
                        summary += f" 近{len(df_filtered)}个月准时率波动较小，整体表现稳定。"

                summary += f" 所选时间范围平均准时率为：{avg_on_time_rate:.2f}%。"
                st.markdown(f"> {summary}")

    # ---------------------- 货代不同月份趋势分析 ----------------------
    st.markdown("## 🚢 货代不同月份趋势分析")
    st.divider()

    # ===== 1. 数据预处理 & 列名校验 =====
    FREIGHT_MONTH_COLUMN_MAPPING = {
        "货代列名": "货代",  # 替换为你实际的货代列名
        "到货年月列名": "到货年月",  # 替换为你实际的到货年月列名
        "提前延期列名": "提前/延期（货代）"  # 替换为你实际的提前/延期列名
    }
    required_cols = [
        FREIGHT_MONTH_COLUMN_MAPPING["货代列名"],
        FREIGHT_MONTH_COLUMN_MAPPING["到货年月列名"],
        FREIGHT_MONTH_COLUMN_MAPPING["提前延期列名"],
        "FBA号"  # 用于统计订单数
    ]
    missing_cols = [col for col in required_cols if col not in df_air.columns]
    if missing_cols:
        st.error(f"缺少货代月度分析必要列：{missing_cols}，请检查数据列名！")
    else:
        # 筛选有效数据
        df_freight_month_valid = df_air[
            (df_air[FREIGHT_MONTH_COLUMN_MAPPING["货代列名"]].notna()) &
            (df_air[FREIGHT_MONTH_COLUMN_MAPPING["到货年月列名"]].notna())
            ].copy()

        if len(df_freight_month_valid) == 0:
            st.warning("暂无货代跨月份数据可分析")
        else:
            # ===== 2. 按「到货年月+货代」聚合核心指标 =====
            freight_month_stats = df_freight_month_valid.groupby(
                [FREIGHT_MONTH_COLUMN_MAPPING["到货年月列名"], FREIGHT_MONTH_COLUMN_MAPPING["货代列名"]]
            ).agg(
                总订单数=("FBA号", "count"),
                提前准时订单数=(FREIGHT_MONTH_COLUMN_MAPPING["提前延期列名"], lambda x: len(x[x == "提前/准时"])),
                延期订单数=(FREIGHT_MONTH_COLUMN_MAPPING["提前延期列名"], lambda x: len(x[x == "延期"]))
            ).reset_index()

            # 重命名列方便后续使用
            freight_month_stats.rename(columns={
                FREIGHT_MONTH_COLUMN_MAPPING["到货年月列名"]: "到货年月",
                FREIGHT_MONTH_COLUMN_MAPPING["货代列名"]: "货代"
            }, inplace=True)

            # 计算准时率（修复列名：确保列名是「准时率(%)」，无多余空格）
            freight_month_stats["准时率(%)"] = round(
                freight_month_stats["提前准时订单数"] / freight_month_stats["总订单数"] * 100, 2
            )


            # ===== 3. 货代归类（优质/合格/异常 + 颜色标记）=====
            def get_freight_category(rate):
                """根据准时率返回归类标签和颜色"""
                if rate >= 90:
                    return "优质", "#2e7d32"  # 绿色
                elif rate >= 80:
                    return "合格", "#ff9800"  # 黄色/橙色
                else:
                    return "异常", "#c62828"  # 红色


            # 新增归类列
            freight_month_stats["货代归类"] = freight_month_stats["准时率(%)"].apply(
                lambda x: get_freight_category(x)[0])
            freight_month_stats["归类颜色"] = freight_month_stats["准时率(%)"].apply(
                lambda x: get_freight_category(x)[1])

            # ===== 4. 双下拉框时间范围筛选 =====
            st.markdown("### 筛选条件")
            # 生成中文月份列表（用于下拉框）
            freight_month_stats["年月排序"] = pd.to_datetime(freight_month_stats["到货年月"] + "-01")
            freight_month_stats["中文月份"] = freight_month_stats["年月排序"].dt.strftime("%Y年%m月")

            # 提取唯一的中文月份（正序）
            unique_months = freight_month_stats.sort_values("年月排序")["中文月份"].unique().tolist()
            unique_ym = freight_month_stats.sort_values("年月排序")["到货年月"].unique().tolist()

            # 双下拉框选择开始/结束月份
            col_start, col_end = st.columns(2)
            with col_start:
                start_month_cn = st.selectbox(
                    "开始月份",
                    options=unique_months,
                    index=0,
                    key="freight_start_month"
                )
            with col_end:
                end_month_cn = st.selectbox(
                    "结束月份",
                    options=unique_months,
                    index=len(unique_months) - 1,
                    key="freight_end_month"
                )

            # 转换为原始年月格式
            start_ym = freight_month_stats[freight_month_stats["中文月份"] == start_month_cn]["到货年月"].iloc[0]
            end_ym = freight_month_stats[freight_month_stats["中文月份"] == end_month_cn]["到货年月"].iloc[0]

            # 筛选时间范围内的数据
            df_freight_filtered = freight_month_stats[
                (freight_month_stats["到货年月"] >= start_ym) &
                (freight_month_stats["到货年月"] <= end_ym)
                ].copy()

            # 按「到货年月降序 + 总订单数降序」排序
            df_freight_filtered["年月排序"] = pd.to_datetime(df_freight_filtered["到货年月"] + "-01")
            df_freight_filtered = df_freight_filtered.sort_values(
                by=["年月排序", "总订单数"],
                ascending=[False, False]
            ).reset_index(drop=True)

            if len(df_freight_filtered) == 0:
                st.warning("所选时间范围内无货代数据")
            else:
                # ===== 5. 货代月度明细表格（带颜色归类）=====
                st.markdown("### 货代月度核心指标明细（到货年月降序+订单数降序）")

                # 准备展示列
                display_cols = [
                    "中文月份", "货代", "总订单数", "提前准时订单数", "延期订单数", "准时率(%)", "货代归类"
                ]
                df_freight_display = df_freight_filtered[display_cols].copy()

                # 表格样式：归类列按颜色标记
                def highlight_freight_category(row):
                    styles = [""] * len(row)
                    # 获取归类颜色
                    color = df_freight_filtered.loc[row.name, "归类颜色"]
                    # 给货代归类列上色
                    styles[
                        display_cols.index("货代归类")] = f"background-color: {color}; color: white; font-weight: bold;"
                    # 准时率<80%标红
                    if row["准时率(%)"] < 80:
                        styles[display_cols.index(
                            "准时率(%)")] = "background-color: #fff5f5; color: #c62828; font-weight: bold;"
                    return styles
                # 3. 核心修复：格式化准时率为2位小数（去掉多余0）
                styled_freight_table = df_freight_display.style.apply(highlight_freight_category, axis=1)
                # 关键：强制格式化准时率为2位小数，自动去除末尾无意义的0
                styled_freight_table = styled_freight_table.format({
                    "准时率(%)": lambda x:
                    # 先保留2位小数，再去掉末尾的0和小数点（如果需要）
                    f"{x:.2f}".rstrip('0').rstrip('.') if '.' in f"{x:.2f}" else f"{x:.2f}"
                })
                st.dataframe(
                    styled_freight_table,
                    use_container_width=True,
                    hide_index=True
                )

                # ===== 6. 货代归类结果汇总表（修复KeyError核心点）=====
                st.markdown("### 货代归类结果汇总（所选时间范围）")

                # 按货代+归类汇总（列名无空格，和前面保持一致）
                freight_category_summary = df_freight_filtered.groupby(["货代", "货代归类"]).agg(
                    涉及月份数=("到货年月", "nunique"),
                    累计订单数=("总订单数", "sum"),
                    平均准时率=("准时率(%)", "mean")  # 修复：去掉列名中的多余空格
                ).reset_index()

                # 格式化平均准时率（保留2位小数）
                freight_category_summary["平均准时率"] = round(freight_category_summary["平均准时率"], 2)
                # 重命名列（可选：添加%符号，更直观）
                freight_category_summary.rename(columns={"平均准时率": "平均准时率(%)"}, inplace=True)


                # 汇总表样式
                def highlight_summary_category(row):
                    styles = [""] * len(row)
                    # 获取归类颜色
                    if row["货代归类"] == "优质":
                        color = "#2e7d32"
                    elif row["货代归类"] == "合格":
                        color = "#ff9800"
                    else:
                        color = "#c62828"
                    cate_col_idx = freight_category_summary.columns.get_loc("货代归类")
                    styles[cate_col_idx] = f"background-color: {color}; color: white; font-weight: bold;"
                    return styles


                styled_summary_table = freight_category_summary.style.apply(highlight_summary_category, axis=1)
                st.dataframe(
                    styled_summary_table,
                    use_container_width=True,
                    hide_index=True
                )

                # ===== 7. 货代月度趋势图（货代筛选器+双轴图）=====
                st.markdown("### 货代月度趋势分析（按货代筛选）")

                # 货代筛选器
                unique_freights = df_freight_filtered["货代"].unique().tolist()
                selected_freight = st.selectbox(
                    "选择货代查看趋势",
                    options=unique_freights,
                    index=0,
                    key="selected_freight"
                )

                # 筛选所选货代的数据（按时间正序）
                df_freight_trend = df_freight_filtered[
                    df_freight_filtered["货代"] == selected_freight
                    ].sort_values("年月排序", ascending=True).reset_index(drop=True)

                if len(df_freight_trend) == 0:
                    st.warning(f"所选时间范围内无{selected_freight}的相关数据")
                else:
                    # 计算该货代的平均准时率（用于虚线）
                    avg_freight_rate = df_freight_trend["准时率(%)"].mean()

                    # 绘制双轴趋势图
                    import plotly.graph_objects as go

                    fig_freight = go.Figure()

                    # 左轴：柱状图（总订单数、提前准时订单数、延期订单数）
                    fig_freight.add_trace(go.Bar(
                        x=df_freight_trend["中文月份"],
                        y=df_freight_trend["总订单数"],
                        name="总订单数",
                        yaxis="y1",
                        marker_color="#4299e1",
                        opacity=0.8
                    ))
                    fig_freight.add_trace(go.Bar(
                        x=df_freight_trend["中文月份"],
                        y=df_freight_trend["提前准时订单数"],
                        name="提前/准时订单数",
                        yaxis="y1",
                        marker_color="#48bb78",
                        opacity=0.8
                    ))
                    fig_freight.add_trace(go.Bar(
                        x=df_freight_trend["中文月份"],
                        y=df_freight_trend["延期订单数"],
                        name="延期订单数",
                        yaxis="y1",
                        marker_color="#e53e3e",
                        opacity=0.8
                    ))

                    # 右轴：折线图（准时率）
                    fig_freight.add_trace(go.Scatter(
                        x=df_freight_trend["中文月份"],
                        y=df_freight_trend["准时率(%)"],
                        name="准时率(%)",
                        yaxis="y2",
                        marker_color="#9f7aea",
                        mode="lines+markers+text",
                        line=dict(width=3),
                        marker=dict(size=8),
                        text=df_freight_trend["准时率(%)"].apply(lambda x: f"{x:.2f}%"),
                        textposition="top center"
                    ))

                    # 平均准时率红色虚线
                    fig_freight.add_trace(go.Scatter(
                        x=df_freight_trend["中文月份"],
                        y=[avg_freight_rate] * len(df_freight_trend),
                        name=f"平均准时率: {avg_freight_rate:.2f}%",
                        yaxis="y2",
                        mode="lines",
                        line=dict(color="#ff0000", dash="dash", width=2),
                        hoverinfo="name+y"
                    ))

                    # 图表配置
                    fig_freight.update_layout(
                        title=f"{selected_freight} - 月度订单数&准时率趋势",
                        yaxis=dict(title="订单数", side="left", range=[0, max(df_freight_trend["总订单数"]) * 1.2]),
                        yaxis2=dict(title="准时率(%)", side="right", overlaying="y", range=[0, 100]),
                        xaxis=dict(title="到货年月", tickangle=45),
                        legend=dict(x=0.02, y=0.98, bordercolor="#eee", borderwidth=1),
                        height=450,
                        plot_bgcolor="#ffffff",
                        barmode="group"
                    )
                    st.plotly_chart(fig_freight, use_container_width=True)

                # ===== 优化后的货代月度表现总结 =====
                st.markdown("### 货代月度表现总结（综合版）")

                # ---------------------- 第一步：整体汇总 & 核心指标计算 ----------------------
                total_months = df_freight_filtered["中文月份"].nunique()
                total_freights = df_freight_filtered["货代"].nunique()
                total_orders = df_freight_filtered["总订单数"].sum()
                avg_overall_rate = round(df_freight_filtered["准时率(%)"].mean(), 2)

                st.markdown(
                    f"> **整体汇总**：所选时间范围共涵盖{total_months}个月份，涉及{total_freights}个货代，累计订单数{total_orders}单，整体平均准时率{avg_overall_rate}%。")

                # --- 核心优化1：先修复最新月份取值逻辑（关键！避免显示2026年2月） ---
                # 对筛选后的数据按年月排序（正序）
                df_filtered_sorted = df_freight_filtered.sort_values("年月排序", ascending=True)
                # 提取筛选范围内的所有有效月份
                valid_months = df_filtered_sorted["中文月份"].unique()
                # 确定筛选范围内的最新月份（避免取到筛选外的月份）
                latest_month = valid_months[-1] if len(valid_months) > 0 else "无数据"


                # --- 核心优化：计算综合评分和评级 ---
                def calculate_comprehensive_score(freight_data):
                    """
                    根据货代数据计算综合评分和评级。
                    综合考虑：订单量、出现频次、加权平均准时率。
                    """
                    # 1. 基础数据
                    total_orders = freight_data["总订单数"].sum()
                    total_months = len(freight_data)
                    # 加权平均准时率：按订单量加权
                    weighted_avg_rate = (freight_data["准时率(%)"] * freight_data[
                        "总订单数"]).sum() / total_orders if total_orders > 0 else 0

                    # 2. 设定门槛（可根据业务调整）
                    MIN_ORDERS = 10  # 最低订单量门槛
                    MIN_MONTHS = 2  # 最低出现月份门槛

                    # 3. 评级逻辑
                    if total_orders < MIN_ORDERS or total_months < MIN_MONTHS:
                        return "样本不足", weighted_avg_rate, total_orders, total_months
                    elif weighted_avg_rate >= 90:
                        return "优质", weighted_avg_rate, total_orders, total_months
                    elif weighted_avg_rate >= 80:
                        return "合格", weighted_avg_rate, total_orders, total_months
                    else:
                        return "异常", weighted_avg_rate, total_orders, total_months


                # --- 为每个货代计算综合评级 ---
                comprehensive_summary = []
                for freight in df_freight_filtered["货代"].unique():
                    freight_data = df_freight_filtered[df_freight_filtered["货代"] == freight].copy()
                    # 修复：避免货代只有1条数据时iloc[0]报错
                    if len(freight_data) > 0:
                        # 按年月降序排序，取最新月份的表现
                        freight_data_sorted = freight_data.sort_values("年月排序", ascending=False)
                        latest_perf = freight_data_sorted.iloc[0]["货代归类"]
                    else:
                        latest_perf = "无数据"

                    rating, avg_rate, total_ord, total_mth = calculate_comprehensive_score(freight_data)
                    comprehensive_summary.append({
                        "货代": freight,
                        "综合评级": rating,
                        "加权平均准时率": round(avg_rate, 2),
                        "累计订单数": total_ord,
                        "出现月份数": total_mth,
                        "最新月份表现": latest_perf
                    })

                df_comprehensive = pd.DataFrame(comprehensive_summary)

                # --- 按综合评级统计 ---
                category_count = df_comprehensive["综合评级"].value_counts()
                cate_summary = []
                if "优质" in category_count:
                    cate_summary.append(f"- **优质货代**：共{category_count['优质']}个，主要表现为加权平均准时率≥90%。")
                if "合格" in category_count:
                    cate_summary.append(
                        f"- **合格货代**：共{category_count['合格']}个，主要表现为加权平均准时率≥80%且<90%。")
                if "异常" in category_count:
                    cate_summary.append(f"- **异常货代**：共{category_count['异常']}个，主要表现为加权平均准时率<80%。")
                if "样本不足" in category_count:
                    cate_summary.append(
                        f"- **样本不足货代**：共{category_count['样本不足']}个，因订单量或出现频次过低，暂不评级。")

                st.markdown("\n".join(cate_summary))

                # --- 核心货代点评 ---
                # 排除样本不足的货代
                valid_freights = df_comprehensive[df_comprehensive["综合评级"] != "样本不足"]
                if not valid_freights.empty:
                    top_freight = valid_freights.sort_values("累计订单数", ascending=False).iloc[0]
                    st.markdown(
                        f">- **核心货代{top_freight['货代']}**：累计订单数最多（{top_freight['累计订单数']}单），加权平均准时率{top_freight['加权平均准时率']}%，综合评级为{top_freight['综合评级']}。")

                # --- 异常提醒 ---
                abnormal_freights = df_comprehensive[df_comprehensive["综合评级"] == "异常"]["货代"].tolist()
                if abnormal_freights:
                    st.markdown(
                        f">- **异常提醒**：{','.join(abnormal_freights)}等货代加权平均准时率低于80%，且满足样本量要求，需重点关注并推动时效优化。")

                st.markdown("#### 2. 各货代详细表现（综合评级）")
                for _, row in df_comprehensive.iterrows():
                    freight = row["货代"]
                    rating = row["综合评级"]
                    avg_rate = row["加权平均准时率"]
                    total_ord = row["累计订单数"]
                    total_mth = row["出现月份数"]
                    latest_perf = row["最新月份表现"]

                    # 归类样式和描述
                    if rating == "优质":
                        color = "#2e7d32"
                        desc = "综合表现优秀，长期稳定可靠。"
                    elif rating == "合格":
                        color = "#ff9800"
                        desc = "综合表现达标，仍有优化空间。"
                    elif rating == "异常":
                        color = "#c62828"
                        desc = "综合表现不佳，存在较大风险。"
                    else:
                        color = "#718096"
                        desc = f"样本不足（订单{total_ord}单/月份{total_mth}个），建议持续观察。"

                    # 生成货代卡片（修复HTML渲染问题）
                    st.markdown(f"""
                    <div style='border:1px solid #e2e8f0; border-radius:6px; padding:15px; margin:10px 0; border-left:4px solid {color};'>
                      <strong style='font-size:16px; color:#1a202c;'>{freight}</strong>
                      <p style='margin:5px 0; color:{color};'>{rating} | {desc}</p>
                      <p style='margin:2px 0; font-size:14px; color:#4a5568;'>📊 加权平均准时率：{avg_rate}% | 📦 累计订单：{total_ord}单 | 📅 出现月份：{total_mth}个月</p>
                      <p style='margin:2px 0; font-size:14px; color:#4a5568;'>🔍 最新月份（{latest_month}）表现：{latest_perf}</p>
                    </div>
                    """, unsafe_allow_html=True)

                # ===== 9. 数据下载 =====
                # 明细数据下载
                freight_detail_csv = df_freight_display.to_csv(index=False, encoding="utf-8-sig")
                st.download_button(
                    label="📥 下载货代月度明细数据",
                    data=freight_detail_csv,
                    file_name="货代月度明细数据.csv",
                    mime="text/csv",
                    key="freight_detail_download"
                )
                # 汇总数据下载
                freight_summary_csv = freight_category_summary.to_csv(index=False, encoding="utf-8-sig")
                st.download_button(
                    label="📥 下载货代归类汇总数据",
                    data=freight_summary_csv,
                    file_name="货代归类汇总数据.csv",
                    mime="text/csv",
                    key="freight_summary_download"
                )
    # ---------------------- 仓库不同月份趋势分析（终极修复版 - 无IndexError） ----------------------
    st.markdown("## 🏠 仓库不同月份趋势分析")
    st.divider()

    # ===== 1. 数据预处理 & 列名校验 =====
    WAREHOUSE_MONTH_COLUMN_MAPPING = {
        "仓库列名": "仓库",  # 替换为你实际的仓库列名
        "到货年月列名": "到货年月",  # 替换为你实际的到货年月列名
        "提前延期列名": "提前/延期（仓库）"  # 替换为你实际的提前/延期列名
    }
    required_warehouse_cols = [
        WAREHOUSE_MONTH_COLUMN_MAPPING["仓库列名"],
        WAREHOUSE_MONTH_COLUMN_MAPPING["到货年月列名"],
        WAREHOUSE_MONTH_COLUMN_MAPPING["提前延期列名"],
        "FBA号"
    ]
    missing_warehouse_cols = [col for col in required_warehouse_cols if col not in df_air.columns]
    if missing_warehouse_cols:
        st.error(f"缺少仓库月度分析必要列：{missing_warehouse_cols}，请检查数据列名！")
    else:
        # 筛选有效数据
        df_warehouse_month_valid = df_air[
            (df_air[WAREHOUSE_MONTH_COLUMN_MAPPING["仓库列名"]].notna()) &
            (df_air[WAREHOUSE_MONTH_COLUMN_MAPPING["到货年月列名"]].notna())
            ].copy()

        if len(df_warehouse_month_valid) == 0:
            st.warning("暂无仓库跨月份数据可分析")
        else:
            # ===== 2. 聚合核心指标 =====
            warehouse_month_stats = df_warehouse_month_valid.groupby(
                [WAREHOUSE_MONTH_COLUMN_MAPPING["到货年月列名"], WAREHOUSE_MONTH_COLUMN_MAPPING["仓库列名"]]
            ).agg(
                总订单数=("FBA号", "count"),
                提前准时订单数=(WAREHOUSE_MONTH_COLUMN_MAPPING["提前延期列名"], lambda x: len(x[x == "提前/准时"])),
                延期订单数=(WAREHOUSE_MONTH_COLUMN_MAPPING["提前延期列名"], lambda x: len(x[x == "延期"]))
            ).reset_index()

            # 重命名列
            warehouse_month_stats.rename(columns={
                WAREHOUSE_MONTH_COLUMN_MAPPING["到货年月列名"]: "到货年月",
                WAREHOUSE_MONTH_COLUMN_MAPPING["仓库列名"]: "仓库"
            }, inplace=True)

            # 计算准时率
            warehouse_month_stats["准时率(%)"] = round(
                warehouse_month_stats["提前准时订单数"] / warehouse_month_stats["总订单数"] * 100, 2
            )


            # ===== 3. 仓库归类 =====
            def get_warehouse_category(rate):
                if rate >= 90:
                    return "优质", "#2e7d32"
                elif rate >= 80:
                    return "合格", "#ff9800"
                else:
                    return "异常", "#c62828"


            warehouse_month_stats["仓库归类"] = warehouse_month_stats["准时率(%)"].apply(
                lambda x: get_warehouse_category(x)[0])
            warehouse_month_stats["归类颜色"] = warehouse_month_stats["准时率(%)"].apply(
                lambda x: get_warehouse_category(x)[1])

            # ===== 4. 时间筛选（终极修复：仅基于年月排序筛选，放弃反向匹配） =====
            st.markdown("### 筛选条件")


            # 核心修改1：生成可靠的年月排序（仅用于筛选，不反向匹配）
            def safe_parse_ym(ym):
                """安全解析到货年月为datetime"""
                try:
                    # 处理常见格式：202509、2025-09、2025年09月等
                    ym_str = str(ym).replace("年", "").replace("月", "").replace("-", "").strip()
                    if len(ym_str) == 6:  # 202509
                        return pd.to_datetime(f"{ym_str[:4]}-{ym_str[4:]}-01")
                    elif len(ym_str) == 8:  # 20250901
                        return pd.to_datetime(ym_str)
                    else:
                        return pd.NaT
                except:
                    return pd.NaT


            # 生成可靠的年月排序列
            warehouse_month_stats["年月排序"] = warehouse_month_stats["到货年月"].apply(safe_parse_ym)
            # 过滤无效日期
            warehouse_month_stats = warehouse_month_stats[warehouse_month_stats["年月排序"].notna()].copy()

            if len(warehouse_month_stats) == 0:
                st.warning("无有效仓库月份数据可分析")
            else:
                # 生成展示用的中文月份（仅用于下拉框展示）
                warehouse_month_stats["中文月份"] = warehouse_month_stats["年月排序"].dt.strftime("%Y年%m月")
                # 获取唯一的中文月份（按时间正序）
                unique_months = sorted(warehouse_month_stats["中文月份"].unique())

                # 核心修改2：下拉框选择中文月份，但筛选时直接用年月排序
                col_start, col_end = st.columns(2)
                with col_start:
                    start_month_cn = st.selectbox("开始月份", options=unique_months, index=0, key="warehouse_start")
                with col_end:
                    end_month_cn = st.selectbox("结束月份", options=unique_months, index=len(unique_months) - 1,
                                                key="warehouse_end")

                # 核心修改3：将选中的中文月份转回datetime，直接筛选年月排序（无反向匹配）
                start_dt = pd.to_datetime(start_month_cn + "-01", format="%Y年%m月-%d")
                end_dt = pd.to_datetime(end_month_cn + "-01", format="%Y年%m月-%d")

                # 直接筛选时间范围（彻底避免反向匹配）
                df_warehouse_filtered = warehouse_month_stats[
                    (warehouse_month_stats["年月排序"] >= start_dt) &
                    (warehouse_month_stats["年月排序"] <= end_dt)
                    ].copy()

                # 排序
                df_warehouse_filtered = df_warehouse_filtered.sort_values(
                    by=["年月排序", "总订单数"], ascending=[False, False]
                ).reset_index(drop=True)

                if len(df_warehouse_filtered) == 0:
                    st.warning("所选时间范围内无仓库数据")
                else:
                    # ===== 5. 月度明细表格 =====
                    st.markdown("### 仓库月度核心指标明细")
                    display_cols = ["中文月份", "仓库", "总订单数", "提前准时订单数", "延期订单数", "准时率(%)",
                                    "仓库归类"]
                    df_warehouse_display = df_warehouse_filtered[display_cols].copy()


                    # 表格样式
                    def highlight_warehouse_category(row):
                        styles = [""] * len(row)
                        color = df_warehouse_filtered.loc[row.name, "归类颜色"]
                        styles[display_cols.index(
                            "仓库归类")] = f"background-color: {color}; color: white; font-weight: bold;"
                        if row["准时率(%)"] < 80:
                            styles[display_cols.index(
                                "准时率(%)")] = "background-color: #fff5f5; color: #c62828; font-weight: bold;"
                        return styles


                    styled_table = df_warehouse_display.style.apply(highlight_warehouse_category, axis=1)
                    styled_table = styled_table.format({"准时率(%)": lambda x: f"{x:.2f}".rstrip('0').rstrip(
                        '.') if '.' in f"{x:.2f}" else f"{x:.2f}"})
                    st.dataframe(styled_table, use_container_width=True, hide_index=True)

                    # ===== 6. 归类汇总表 =====
                    st.markdown("### 仓库归类结果汇总")
                    warehouse_category_summary = df_warehouse_filtered.groupby(["仓库", "仓库归类"]).agg(
                        涉及月份数=("到货年月", "nunique"),
                        累计订单数=("总订单数", "sum"),
                        平均准时率=("准时率(%)", "mean")
                    ).reset_index()
                    warehouse_category_summary["平均准时率"] = round(warehouse_category_summary["平均准时率"], 2)
                    warehouse_category_summary.rename(columns={"平均准时率": "平均准时率(%)"}, inplace=True)


                    # 汇总表样式
                    def highlight_summary(row):
                        styles = [""] * len(row)
                        color = "#2e7d32" if row["仓库归类"] == "优质" else "#ff9800" if row[
                                                                                             "仓库归类"] == "合格" else "#c62828"
                        cate_idx = warehouse_category_summary.columns.get_loc("仓库归类")
                        styles[cate_idx] = f"background-color: {color}; color: white; font-weight: bold;"
                        return styles


                    styled_summary = warehouse_category_summary.style.apply(highlight_summary, axis=1)
                    st.dataframe(styled_summary, use_container_width=True, hide_index=True)

                    # ===== 7. 趋势图 =====
                    st.markdown("### 仓库月度趋势分析")
                    unique_warehouses = df_warehouse_filtered["仓库"].unique().tolist()
                    selected_warehouse = st.selectbox("选择仓库", options=unique_warehouses, index=0,
                                                      key="selected_warehouse")

                    df_trend = df_warehouse_filtered[df_warehouse_filtered["仓库"] == selected_warehouse].sort_values(
                        "年月排序", ascending=True).copy()
                    if len(df_trend) == 0:
                        st.warning(f"无{selected_warehouse}的趋势数据")
                    else:
                        import plotly.graph_objects as go

                        avg_rate = df_trend["准时率(%)"].mean()

                        fig = go.Figure()
                        # 柱状图
                        fig.add_trace(
                            go.Bar(x=df_trend["中文月份"], y=df_trend["总订单数"], name="总订单数", yaxis="y1",
                                   marker_color="#4299e1"))
                        fig.add_trace(
                            go.Bar(x=df_trend["中文月份"], y=df_trend["提前准时订单数"], name="提前/准时订单数",
                                   yaxis="y1", marker_color="#48bb78"))
                        fig.add_trace(
                            go.Bar(x=df_trend["中文月份"], y=df_trend["延期订单数"], name="延期订单数", yaxis="y1",
                                   marker_color="#e53e3e"))
                        # 折线图
                        fig.add_trace(
                            go.Scatter(x=df_trend["中文月份"], y=df_trend["准时率(%)"], name="准时率(%)", yaxis="y2",
                                       marker_color="#9f7aea", mode="lines+markers+text",
                                       text=[f"{x:.2f}%" for x in df_trend["准时率(%)"]]))
                        # 平均线
                        fig.add_trace(go.Scatter(x=df_trend["中文月份"], y=[avg_rate] * len(df_trend),
                                                 name=f"平均准时率: {avg_rate:.2f}%",
                                                 yaxis="y2", mode="lines", line=dict(color="#ff0000", dash="dash")))

                        fig.update_layout(
                            title=f"{selected_warehouse} 月度趋势",
                            yaxis=dict(title="订单数", side="left", range=[0, max(df_trend["总订单数"]) * 1.2]),
                            yaxis2=dict(title="准时率(%)", side="right", overlaying="y", range=[0, 100]),
                            xaxis=dict(title="年月", tickangle=45),
                            height=450, barmode="group"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    # ===== 8. 综合总结 =====
                    st.markdown("### 仓库月度表现总结")
                    total_months = df_warehouse_filtered["中文月份"].nunique()
                    total_warehouses = df_warehouse_filtered["仓库"].nunique()
                    total_orders = df_warehouse_filtered["总订单数"].sum()
                    avg_overall = round(df_warehouse_filtered["准时率(%)"].mean(), 2)

                    st.markdown(
                        f"> 所选范围涵盖{total_months}个月，{total_warehouses}个仓库，累计{total_orders}单，整体平均准时率{avg_overall}%。")


                    # 综合评级
                    def calc_score(warehouse_data):
                        total_ord = warehouse_data["总订单数"].sum()
                        total_mth = len(warehouse_data)
                        weighted_rate = (warehouse_data["准时率(%)"] * warehouse_data[
                            "总订单数"]).sum() / total_ord if total_ord > 0 else 0
                        if total_ord < 10 or total_mth < 2:
                            return "样本不足", weighted_rate
                        elif weighted_rate >= 90:
                            return "优质", weighted_rate
                        elif weighted_rate >= 80:
                            return "合格", weighted_rate
                        else:
                            return "异常", weighted_rate


                    summary_list = []
                    for warehouse in df_warehouse_filtered["仓库"].unique():
                        wh_data = df_warehouse_filtered[df_warehouse_filtered["仓库"] == warehouse].copy()
                        latest_perf = wh_data.sort_values("年月排序", ascending=False).iloc[0]["仓库归类"] if len(
                            wh_data) > 0 else "无数据"
                        rating, avg_rate = calc_score(wh_data)
                        summary_list.append({
                            "仓库": warehouse,
                            "综合评级": rating,
                            "加权平均准时率": round(avg_rate, 2),
                            "累计订单数": wh_data["总订单数"].sum(),
                            "出现月份数": len(wh_data),
                            "最新表现": latest_perf
                        })

                    df_summary = pd.DataFrame(summary_list)
                    cate_count = df_summary["综合评级"].value_counts()
                    cate_text = []
                    if "优质" in cate_count: cate_text.append(f"- 优质仓库：{cate_count['优质']}个（≥90%）")
                    if "合格" in cate_count: cate_text.append(f"- 合格仓库：{cate_count['合格']}个（80%-90%）")
                    if "异常" in cate_count: cate_text.append(f"- 异常仓库：{cate_count['异常']}个（<80%）")
                    if "样本不足" in cate_count: cate_text.append(f"- 样本不足：{cate_count['样本不足']}个")
                    st.markdown("\n".join(cate_text))

                    # 仓库卡片
                    st.markdown("#### 各仓库详细表现")
                    for _, row in df_summary.iterrows():
                        color = "#2e7d32" if row["综合评级"] == "优质" else "#ff9800" if row[
                                                                                             "综合评级"] == "合格" else "#c62828" if \
                        row["综合评级"] == "异常" else "#718096"
                        desc = "优秀稳定" if row["综合评级"] == "优质" else "达标待优化" if row[
                                                                                                "综合评级"] == "合格" else "风险较高" if \
                        row["综合评级"] == "异常" else "需持续观察"
                        st.markdown(f"""
                        <div style='border:1px solid #e2e8f0; border-radius:6px; padding:15px; margin:10px 0; border-left:4px solid {color};'>
                          <strong style='font-size:16px;'>{row['仓库']}</strong>
                          <p style='color:{color};'>{row['综合评级']} | {desc}</p>
                          <p>📊 加权准时率：{row['加权平均准时率']}% | 📦 累计订单：{row['累计订单数']}单 | 📅 出现月份：{row['出现月份数']}个</p>
                          <p>🔍 最新表现：{row['最新表现']}</p>
                        </div>
                        """, unsafe_allow_html=True)

                    # ===== 9. 数据下载 =====
                    st.markdown("### 数据下载")
                    col1, col2 = st.columns(2)
                    with col1:
                        csv_detail = df_warehouse_display.to_csv(index=False, encoding="utf-8-sig")
                        st.download_button("下载明细数据", data=csv_detail, file_name="仓库月度明细.csv",
                                           mime="text/csv")
                    with col2:
                        csv_summary = warehouse_category_summary.to_csv(index=False, encoding="utf-8-sig")
                        st.download_button("下载汇总数据", data=csv_summary, file_name="仓库归类汇总.csv",
                                           mime="text/csv")


    # ===================== 三、数据源 =====================
    st.subheader("📋 数据源筛选")

    # ---------------------- 筛选器（单选+默认“全部”） ----------------------
    col1, col2, col3, col4 = st.columns(4)

    # 1. 到货年月筛选器（单选+默认“全部”）
    with col1:
        month_unique = df_air["到货年月"].dropna().unique()
        month_options_filter = ["全部"] + sorted(month_unique, reverse=True) if len(month_unique) > 0 else ["全部"]
        selected_month_filter = st.selectbox(
            "到货年月",
            options=month_options_filter,
            index=0,  # 默认选中“全部”
            key="filter_month_single"
        )

    # 2. 仓库筛选器（单选+默认“全部”）
    with col2:
        warehouse_options_filter = ["全部"]
        if "仓库" in df_air.columns:
            warehouse_unique = df_air["仓库"].dropna().unique()
            if len(warehouse_unique) > 0:
                warehouse_options_filter += list(warehouse_unique)
        selected_warehouse_filter = st.selectbox(
            "仓库",
            options=warehouse_options_filter,
            index=0,  # 默认选中“全部”
            key="filter_warehouse_single"
        )

    # 3. 货代筛选器（单选+默认“全部”）
    with col3:
        freight_options_filter = ["全部"]
        if "货代" in df_air.columns:
            freight_unique = df_air["货代"].dropna().unique()
            if len(freight_unique) > 0:
                freight_options_filter += list(freight_unique)
        selected_freight_filter = st.selectbox(
            "货代",
            options=freight_options_filter,
            index=0,  # 默认选中“全部”
            key="filter_freight_single"
        )

    # 4. 提前/延期筛选器（单选+默认“全部”）
    with col4:
        status_options_filter = ["全部"]
        if "提前/延期" in df_air.columns:
            status_unique = df_air["提前/延期"].dropna().unique()
            if len(status_unique) > 0:
                status_options_filter += list(status_unique)
        selected_status_filter = st.selectbox(
            "提前/延期",
            options=status_options_filter,
            index=0,  # 默认选中“全部”
            key="filter_status_single"
        )

    # ---------------------- 应用筛选逻辑 ----------------------
    filter_conditions = pd.Series([True] * len(df_air))
    if selected_month_filter != "全部" and len(df_air) > 0:
        filter_conditions = filter_conditions & (df_air["到货年月"] == selected_month_filter)
    if "仓库" in df_air.columns and selected_warehouse_filter != "全部" and len(df_air) > 0:
        filter_conditions = filter_conditions & (df_air["仓库"] == selected_warehouse_filter)
    if "货代" in df_air.columns and selected_freight_filter != "全部" and len(df_air) > 0:
        filter_conditions = filter_conditions & (df_air["货代"] == selected_freight_filter)
    if "提前/延期" in df_air.columns and selected_status_filter != "全部" and len(df_air) > 0:
        filter_conditions = filter_conditions & (df_air["提前/延期"] == selected_status_filter)
    df_filtered = df_air[filter_conditions].copy()

    # ---------------------- 计算平均值 ----------------------
    avg_target_cols = [
        "发货-起飞", "到港-提取", "提取-签收", "签收-完成上架",
        "发货-签收", "发货-完成上架", "签收-发货时间", "上架完成-发货时间",
        "预计物流时效-实际物流时效差值(绝对值)", "预计物流时效-实际物流时效差值"
    ]
    display_cols = [
        "到货年月", "FBA号", "店铺", "仓库", "货代", "提前/延期",
        "异常备注", "发货-起飞", "到港-提取", "提取-签收","清关耗时", "签收-完成上架",
        "发货-签收", "发货-完成上架", "签收-发货时间", "上架完成-发货时间",
        "预计物流时效-实际物流时效差值(绝对值)", "预计物流时效-实际物流时效差值"
    ]
    display_cols = [col for col in display_cols if col in df_filtered.columns]

    # 初始化平均值
    avg_row = {col: "-" for col in display_cols}
    if len(df_filtered) > 0:
        for col in avg_target_cols:
            if col in display_cols:
                numeric_vals = pd.to_numeric(df_filtered[col], errors='coerce').dropna()
                avg_row[col] = round(numeric_vals.mean(), 2) if len(numeric_vals) > 0 else 0.00

    # 处理数据行
    df_display = df_filtered[display_cols].copy() if len(df_filtered) > 0 else pd.DataFrame(columns=display_cols)
    for col in avg_target_cols:
        if col in df_display.columns:
            df_display[col] = pd.to_numeric(df_display[col], errors='coerce')

    # ---------------------- 生成表格（修复样式语法） ----------------------
    st.markdown("### 原始数据（含筛选后平均值）")

    # 列宽配置（简化为单行字符串，避免语法错误）
    col_width_config = {
        "到货年月": "80px", "FBA号": "120px", "店铺": "80px", "仓库": "80px",
        "货代": "80px", "提前/延期": "80px", "异常备注": "100px", "发货-起飞": "80px",
        "到港-提取": "80px", "提取-签收": "80px", "签收-完成上架": "100px", "发货-签收": "80px",
        "发货-完成上架": "100px", "签收-发货时间": "100px", "上架完成-发货时间": "120px",
        "预计物流时效-实际物流时效差值(绝对值)": "150px", "预计物流时效-实际物流时效差值": "150px"
    }

    # 核心修复：CSS样式改为单行紧凑格式，避免换行导致的语法错误
    table_css = """
    <style>
    /* 全局表格样式重置 */
    .table-outer {
        width: 100%;
        border: 1px solid #dee2e6;
        margin: 10px 0;
        font-size: 14px;
    }
    /* 固定头部容器 */
    .table-fixed {
        position: sticky;
        top: 0;
        background: white;
        z-index: 99;
    }
    /* 表头样式 */
    .table-header th {
        width: var(--col-width);
        max-width: var(--col-width);
        min-width: var(--col-width);
        padding: 8px 12px;
        border: 1px solid #dee2e6;
        background: #e9ecef;
        font-weight: bold;
        text-align: left;
        white-space: normal;
        word-wrap: break-word;
        vertical-align: top;
    }
    /* 平均值行样式 */
    .table-avg td {
        width: var(--col-width);
        max-width: var(--col-width);
        min-width: var(--col-width);
        padding: 8px 12px;
        border: 1px solid #dee2e6;
        background: #fff3cd;
        font-weight: bold;
        text-align: left;
        white-space: normal;
        word-wrap: break-word;
        vertical-align: top;
    }
    /* 数据滚动容器 */
    .table-scroll {
        height: 400px;
        overflow-y: auto;
        overflow-x: hidden;
    }
    /* 数据行样式 */
    .table-data td {
        width: var(--col-width);
        max-width: var(--col-width);
        min-width: var(--col-width);
        padding: 8px 12px;
        border: 1px solid #dee2e6;
        text-align: left;
        white-space: normal;
        word-wrap: break-word;
        vertical-align: top;
    }
    /* 高亮单元格 */
    .highlight {
        background-color: #ffebee !important;
    }
    /* 表格布局统一 */
    .table-header, .table-avg, .table-data {
        width: 100%;
        table-layout: fixed;
        border-collapse: collapse;
        border-spacing: 0;
    }
    </style>
    """

    # 构建表头（使用CSS变量传递列宽，避免内联样式换行错误）
    header_html = "<table class='table-header'><tr>"
    for col in display_cols:
        width = col_width_config.get(col, "100px")
        header_html += f"<th style='--col-width: {width}'>{col}</th>"
    header_html += "</tr></table>"

    # 构建平均值行
    avg_html = "<table class='table-avg'><tr>"
    for col in display_cols:
        width = col_width_config.get(col, "100px")
        val = avg_row[col]
        if col in avg_target_cols and isinstance(val, (int, float)):
            val = f"{val:.2f}"
        avg_html += f"<td style='--col-width: {width}'>{val}</td>"
    avg_html += "</tr></table>"

    # 构建数据行
    data_html = "<table class='table-data'><tbody>"
    if len(df_display) > 0:
        for _, row in df_display.iterrows():
            data_html += "<tr>"
            for col in display_cols:
                width = col_width_config.get(col, "100px")
                val = row[col]
                highlight = "highlight" if (
                            col in avg_target_cols and pd.notna(val) and pd.notna(avg_row[col]) and isinstance(
                        avg_row[col], (int, float)) and float(val) > avg_row[col]) else ""
                display_val = f"{val:.2f}" if (col in avg_target_cols and isinstance(val, (int, float))) else (
                    "" if pd.isna(val) else str(val))
                data_html += f"<td style='--col-width: {width}' class='{highlight}'>{display_val}</td>"
            data_html += "</tr>"
    else:
        data_html += f"<tr><td colspan='{len(display_cols)}' style='text-align: center; padding: 20px;'>⚠️ 暂无符合筛选条件的数据</td></tr>"
    data_html += "</tbody></table>"

    # 拼接最终HTML（核心：使用CSS变量传递列宽，避免内联样式换行）
    final_html = f"""
    {table_css}
    <div class='table-outer'>
        <div class='table-fixed'>
            {header_html}
            {avg_html}
        </div>
        <div class='table-scroll'>
            {data_html}
        </div>
    </div>
    """

    st.markdown(final_html, unsafe_allow_html=True)

    # 数据量提示
    if len(df_filtered) > 0:
        st.caption(f"当前筛选结果共 {len(df_filtered)} 条数据 | 总数据量：{len(df_air)} 条")
    else:
        st.caption("⚠️ 暂无符合筛选条件的业务数据")