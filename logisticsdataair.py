import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from io import BytesIO
import base64

# 页面配置（完全保留）
st.set_page_config(
    page_title="FBA空派物流交期分析看板",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------- 工具函数（完全保留你的原有代码） ----------------------
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
        if pd.isna(val) or val == "-" or str(val).strip() == "":
            return ""
        val_str = str(val).replace('%', '').strip()
        val_num = float(val_str)
        if val_num > 0:
            return "color: red"
        elif val_num < 0:
            return "color: green"
    except:
        pass
    return ""

def get_table_download_link(df, filename, text):
    """生成表格下载链接"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='FBA空派明细')
    output.seek(0)
    b64 = base64.b64encode(output.read()).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">{text}</a>'
    return href

# ---------------------- 数据加载函数（两份数据逻辑） ----------------------
@st.cache_data
def load_data():
    url = "https://github.com/Jane-zzz-123/Logistics/raw/main/Logisticsdata.xlsx"
    try:
        df_all = pd.read_excel(url, sheet_name="上架完成-空运(FBA号）")  # 全部数据
    except Exception as e:
        st.error(f"读取数据失败：{str(e)}")
        return pd.DataFrame(), pd.DataFrame()

    # 处理「是否为异常数据」列
    abnormal_col = "是否为异常数据"
    if abnormal_col in df_all.columns:
        df_all[abnormal_col] = df_all[abnormal_col].str.strip().fillna("否")
        df_all[abnormal_col] = df_all[abnormal_col].replace({
            "异常数据": "是", "正常数据": "否", "异常": "是", "正常": "否"
        })
        df_clean = df_all[df_all[abnormal_col] == "否"].copy()  # 纯净数据
    else:
        df_all[abnormal_col] = "否"
        df_clean = df_all.copy()
        st.warning(f"未找到「{abnormal_col}」列，已默认全部为正常数据（否）")

    # 核心列筛选
    core_columns = [
        "货件单号", "FBA号", "物流方式", "店铺", "仓库", "区域","货代", "异常备注",
        "发货-起飞", "到港-提取", "提取-签收", "签收-完成上架",
        "到货年月", "上架完成-发货时间","发货-签收",
        "预计物流时效-实际物流时效差值(绝对值)",
        "预计物流时效-实际物流时效差值", "提前/延期(整体)",
        "预计物流时效-实际物流时效差值（货代）","发货-提取",
        "提前/延期（货代）", "提前/延期（仓库）", abnormal_col
    ]
    existing_columns = [col for col in core_columns if col in df_all.columns]
    missing_columns = [col for col in core_columns if col not in df_all.columns]
    if missing_columns:
        st.warning(f"以下列不存在，已忽略：{missing_columns}")
    df_all = df_all[existing_columns]
    df_clean = df_clean[existing_columns]

    # 统一到货年月格式
    df_all["到货年月"] = pd.to_datetime(df_all["到货年月"], errors='coerce').dt.strftime("%Y-%m")
    df_clean["到货年月"] = pd.to_datetime(df_clean["到货年月"], errors='coerce').dt.strftime("%Y-%m")
    df_all = df_all.dropna(subset=["到货年月"])
    df_clean = df_clean.dropna(subset=["到货年月"])

    # 清洗数值列
    abs_diff_col = "预计物流时效-实际物流时效差值(绝对值)"
    real_diff_col = "预计物流时效-实际物流时效差值"
    if abs_diff_col in df_all.columns:
        df_all[abs_diff_col] = pd.to_numeric(df_all[abs_diff_col], errors='coerce').fillna(0)
        df_clean[abs_diff_col] = pd.to_numeric(df_clean[abs_diff_col], errors='coerce').fillna(0)
    if real_diff_col in df_all.columns:
        df_all[real_diff_col] = pd.to_numeric(df_all[real_diff_col], errors='coerce').fillna(0)
        df_clean[real_diff_col] = pd.to_numeric(df_clean[real_diff_col], errors='coerce').fillna(0)

    return df_all, df_clean


# ---------------------- 主程序逻辑 ----------------------
# 1. 加载两份基础数据
df_all, df_clean = load_data()
if df_all.empty:
    st.error("暂无可用数据，请检查数据源或列名！")
    st.stop()

# 2. 顶部筛选按钮
st.header("FBA空派物流交期分析看板")
data_filter = st.radio(
    "📊 选择数据范围：",
    options=["全部数据", "纯净数据（剔除异常）"],
    index=0,
    horizontal=True,
    key="data_filter"
)

# 3. 核心：生成两套数据（完全满足你的需求）
if data_filter == "纯净数据（剔除异常）":
    # 仓库分析用：不去重，全部FBA号
    df_selected_FBA = df_clean.copy()

    # 非仓库分析用：按【货件单号】去重（保留第一条）
    df_selected = df_clean.drop_duplicates(subset=["货件单号"], keep="first").copy()

    exclude_count = len(df_all) - len(df_clean)
    st.success(
        f"✅ 已筛选为纯净数据，剔除 {exclude_count} 条异常数据（全局），当前共 {len(df_selected)} 条货件记录 | {len(df_selected_FBA)} 条FBA记录")
else:
    # 仓库分析用：不去重，全部FBA号
    df_selected_FBA = df_all.copy()

    # 非仓库分析用：按【货件单号】去重（保留第一条）
    df_selected = df_all.drop_duplicates(subset=["货件单号"], keep="first").copy()

    abnormal_count_total = len(df_all[df_all["是否为异常数据"] == "是"])
    st.info(
        f"ℹ️ 当前展示全部数据（全局），共 {len(df_selected)} 条货件记录 | {len(df_selected_FBA)} 条FBA记录（含 {abnormal_count_total} 条异常数据）")

# 5. 主看板区域
st.title("🚢 FBA空派分析看板区域")
st.divider()

# 6. 当月数据筛选（基于 df_selected，不会丢数据）
st.subheader("🔍 当月空派分析")
month_options = sorted(df_selected["到货年月"].unique(), reverse=True)
if not month_options:
    st.warning("⚠️ 暂无可用的到货年月数据")
    st.stop()

selected_month = st.selectbox(
    "选择到货年月",
    options=month_options,
    index=0,
    key="month_selector_current"
)
st.subheader("")  # 空行分隔，优化排版
# 获取所有物流方式选项（去重），并添加“全部”选项
logistics_methods = ['全部'] + list(df_selected['物流方式'].dropna().unique())
# 创建下拉筛选器，默认选中“全部”
selected_logistics = st.selectbox(
    "选择物流方式",
    options=logistics_methods,
    index=0,  # 默认选中第一个选项（全部）
    key="logistics_filter"  # 唯一key，避免streamlit缓存冲突
)

# -------------------------------------------
# 7. 当月数据【两套同步筛选】
# -------------------------------------------
# A. 货件去重（非仓库用）
df_current = df_selected[df_selected["到货年月"] == selected_month].copy()
if selected_logistics != '全部':
    df_current = df_current[df_current['物流方式'] == selected_logistics].copy()

# B. FBA不去重（仓库分析用）→ 新增
df_current_FBA = df_selected_FBA[df_selected_FBA["到货年月"] == selected_month].copy()
if selected_logistics != '全部':
    df_current_FBA = df_current_FBA[df_current_FBA['物流方式'] == selected_logistics].copy()

# -------------------------------------------
# 8. 上月数据【两套同步筛选】
# -------------------------------------------
prev_month = get_prev_month(selected_month)

# A. 货件去重（非仓库用）
df_prev = df_selected[df_selected["到货年月"] == prev_month].copy() if prev_month and prev_month in month_options else pd.DataFrame()
if selected_logistics != '全部' and not df_prev.empty:
    df_prev = df_prev[df_prev['物流方式'] == selected_logistics].copy()

# B. FBA不去重（仓库分析用）→ 新增
df_prev_FBA = df_selected_FBA[df_selected_FBA["到货年月"] == prev_month].copy() if prev_month and prev_month in month_options else pd.DataFrame()
if selected_logistics != '全部' and not df_prev_FBA.empty:
    df_prev_FBA = df_prev_FBA[df_prev_FBA['物流方式'] == selected_logistics].copy()

# -------------------------------------------
# 9. 当月异常数据统计（保持你原有逻辑不变）
# -------------------------------------------
abnormal_filter = (df_all["到货年月"] == selected_month) & (df_all["是否为异常数据"] == "是")
if selected_logistics != '全部':
    abnormal_filter = abnormal_filter & (df_all["物流方式"] == selected_logistics)
abnormal_current_month = len(df_all[abnormal_filter])

logistics_tip = f"，筛选物流方式：{selected_logistics}" if selected_logistics != "全部" else ""
if data_filter == "纯净数据（剔除异常）":
    st.info(f"📌 【{selected_month}】已筛选为纯净数据，剔除 {abnormal_current_month} 条异常数据{logistics_tip}，当前共 {len(df_current)} 条货件记录 | {len(df_current_FBA)} 条FBA记录")
else:
    st.info(f"📌 【{selected_month}】当前显示全部数据{logistics_tip}，共 {len(df_current)} 条货件记录 | {len(df_current_FBA)} 条FBA记录（含 {abnormal_current_month} 条异常数据）")

# ---------------------- 你的核心指标/可视化/表格代码（仅改数据源引用） ----------------------
# ---------------------- ① 核心指标卡片 ----------------------
st.markdown("### 核心指标")

# 计算核心指标
# 1. FBA单数
current_fba = len(df_current)
prev_fba = len(df_prev) if not df_prev.empty else 0
fba_change = current_fba - prev_fba
fba_change_text = f"{'↑' if fba_change > 0 else '↓' if fba_change < 0 else '—'} {abs(fba_change)} (上月: {prev_fba})"
fba_change_color = "red" if fba_change > 0 else "green" if fba_change < 0 else "gray"

# 2. 提前/准时数（修复：匹配实际数据中的值，比如可能是"提前"或"准时"分开存储）
# 兼容处理：如果数据中是"提前"和"准时"分开，合并统计
if "提前/延期(整体)" in df_current.columns:
    # 适配不同的数据值：支持"提前/准时"、"提前"、"准时"三种情况
    current_on_time = len(df_current[df_current["提前/延期(整体)"].isin(["提前/准时", "提前", "准时"])])
else:
    current_on_time = 0

if not df_prev.empty and "提前/延期(整体)" in df_prev.columns:
    prev_on_time = len(df_prev[df_prev["提前/延期(整体)"].isin(["提前/准时", "提前", "准时"])])
else:
    prev_on_time = 0

on_time_change = current_on_time - prev_on_time
on_time_change_text = f"{'↑' if on_time_change > 0 else '↓' if on_time_change < 0 else '—'} {abs(on_time_change)} (上月: {prev_on_time})"
on_time_change_color = "red" if on_time_change > 0 else "green" if on_time_change < 0 else "gray"

# 3. 延期数
current_delay = len(df_current[df_current["提前/延期(整体)"] == "延期"]) if "提前/延期(整体)" in df_current.columns else 0
prev_delay = len(
    df_prev[df_prev["提前/延期(整体)"] == "延期"]) if not df_prev.empty and "提前/延期(整体)" in df_prev.columns else 0
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

# 计算辅助指标（业务视角）
total_orders = current_fba
on_time_rate = (current_on_time / total_orders * 100) if total_orders > 0 else 0  # 准时率
delay_rate = (current_delay / total_orders * 100) if total_orders > 0 else 0  # 延期率
prev_on_time_rate = (prev_on_time / prev_fba * 100) if prev_fba > 0 else 0  # 上月准时率
on_time_rate_change = on_time_rate - prev_on_time_rate  # 准时率变化

# 核心结论（先给定性判断）
if on_time_rate >= 90:
    core_conclusion = f"{selected_month}空派物流整体表现优秀，准时率达{on_time_rate:.1f}%，远高于行业基准"
elif on_time_rate >= 80:
    core_conclusion = f"{selected_month}空派物流表现良好，准时率{on_time_rate:.1f}%，整体可控"
elif on_time_rate >= 70:
    core_conclusion = f"{selected_month}空派物流表现一般，准时率{on_time_rate:.1f}%，需关注延期问题"
else:
    core_conclusion = f"{selected_month}空派物流表现较差，准时率仅{on_time_rate:.1f}%，延期风险显著"

# 关键数据支撑（精简+业务化）
data_support = f"""
本月共处理订单{current_fba}单（环比{'+' if fba_change > 0 else ''}{fba_change}单）：
✅ 提前/准时单{current_on_time}单（准时率{on_time_rate:.1f}%，环比{'↑' if on_time_rate_change > 0 else '↓'}{abs(on_time_rate_change):.1f}个百分点）；
❌ 延期单{current_delay}单（延期率{delay_rate:.1f}%）；
📊 实际物流时效与预计的偏差均值为{current_diff_avg:.2f}天（绝对值均值{current_abs_avg:.2f}天），环比{'扩大' if abs_change > 0 else '收窄'}{abs(abs_change):.2f}天。
"""

# 风险/亮点提示（针对性分析）
tips = ""
# 1. 准时率大幅波动提示
if abs(on_time_rate_change) >= 5:
    if on_time_rate_change > 0:
        tips += f"💡 亮点：本月准时率环比提升{on_time_rate_change:.1f}个百分点，物流效率显著改善；"
    else:
        tips += f"⚠️ 风险：本月准时率环比下降{abs(on_time_rate_change):.1f}个百分点，需排查延期原因；"
# 2. 延期单占比过高提示
if delay_rate >= 30:
    tips += f"⚠️ 风险：延期单占比超30%，建议优先核查高频延期的货代/仓库；"
# 3. 时效偏差扩大提示
if abs_change >= 2:
    tips += f"⚠️ 风险：时效偏差绝对值环比扩大{abs_change:.2f}天，预计时效的准确性需优化；"
# 4. 无明显风险的正向提示
if not tips:
    tips = "💡 本月物流时效无显著异常，各维度表现稳定。"

# 整合最终总结
summary_text = f"""
### {selected_month}空派物流核心分析
{core_conclusion}

{data_support}

{tips}
"""

# 渲染总结（用markdown美化）
st.markdown(summary_text)

# ---------------------- ② 当月准时率与时效偏差 ----------------------
st.markdown("### 准时率与时效偏差分布")
col1, col2 = st.columns(2)

# 左：饼图（提前/准时 vs 延期）
with col1:
    if "提前/延期(整体)" in df_current.columns and len(df_current) > 0:
        # 兼容数据值：合并"提前/准时"、"提前"、"准时"为同一类别
        df_current["提前/延期(整体)_分类"] = df_current["提前/延期(整体)"].apply(
            lambda x: "提前/准时" if x in ["提前/准时", "提前", "准时"] else "延期" if x == "延期" else "其他"
        )
        pie_data = df_current["提前/延期(整体)_分类"].value_counts()

        # 确保颜色映射严格生效（显式指定颜色列表）
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
# ---------------------- ③ 当月FBA空派明细表格 ----------------------
st.markdown("### 空派明细（含平均值）")

# 准备明细数据
detail_cols = [
    "到货年月", "提前/延期(整体)", "FBA号", "物流方式", "店铺", "仓库", "货代",
    # 新增的物流阶段列（加在货代右边）
    "发货-起飞", "到港-提取", "提取-签收", "签收-完成上架","发货-提取",
    "发货-签收", "上架完成-发货时间", "提前/延期（货代）",
    "提前/延期（仓库）",
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
        "发货-起飞", "到港-提取", "提取-签收", "签收-完成上架","发货-提取",
        "发货-签收", "上架完成-发货时间"
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
        elif col in ["提前/延期(整体)", "FBA号", "店铺", "仓库", "货代", "物流方式", "提前/延期（货代）",
                     "提前/延期（仓库）"]:
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
    # 构建带平均值的完整数据（用于下载）
    df_download = pd.concat([pd.DataFrame([avg_row]), df_detail], ignore_index=True)

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
# --------------------------
# 1. 数据预处理 & 字段定义（核心：匹配你的业务逻辑）
# --------------------------
st.subheader("📝 延期订单深度归因分析")

# 请确认以下字段名与你的数据完全一致！
main_delay_col = "提前/延期(整体)"  # 总提前/延期列
forwarder_delay_col = "提前/延期（货代）"  # 货代延期分类列
warehouse_delay_col = "提前/延期（仓库）"  # 仓库延期分类列
# 🔥 按物流口径修正：仅发货-提取属于货代，其余归为其他原因
forwarder_stage_col =["发货-起飞", "到港-提取"]   # 货代仅负责这一个环节
other_stage_cols = "提取签收"  # 其他原因环节
warehouse_stage_col = "签收-完成上架"  # 仓库负责环节
all_stage_cols = forwarder_stage_col + [other_stage_cols] + [warehouse_stage_col]  # 所有环节

# 1.1 基础字段清洗（统一格式，避免筛选错误）
df_current[main_delay_col] = df_current[main_delay_col].fillna("未知").apply(
    lambda x: x.strip() if isinstance(x, str) else "未知")
df_current[forwarder_delay_col] = df_current[forwarder_delay_col].fillna("未知").apply(
    lambda x: x.strip() if isinstance(x, str) else "未知")
df_current[warehouse_delay_col] = df_current[warehouse_delay_col].fillna("未知").apply(
    lambda x: x.strip() if isinstance(x, str) else "未知")

# 1.2 环节字段数值化（确保均值计算准确）
for col in all_stage_cols:
    df_current[col] = pd.to_numeric(df_current[col], errors="coerce").fillna(0.0)

# --------------------------
# 2. 按修正后业务逻辑筛选数据集
# --------------------------
# 2.1 正常订单集：总状态=提前/准时
df_normal = df_current[df_current[main_delay_col] == "提前/准时"].copy()
# 2.2 货代延期订单集：总状态=延期 + 货代状态=延期（仅关联发货-提取环节）
df_forwarder_delay = df_current[
    (df_current[main_delay_col] == "延期") &
    (df_current[forwarder_delay_col] == "延期")
    ].copy()
# 2.3 其他原因延期订单集：总状态=延期 + 非货代/非仓库原因（提取-到港/到港-签收）
df_other_delay = df_current[
    (df_current[main_delay_col] == "延期") &
    (df_current[forwarder_delay_col] != "延期") &
    (df_current[warehouse_delay_col] != "延期")
    ].copy()
# 2.4 仓库延期订单集：总状态=延期 + 仓库状态=延期
df_warehouse_delay = df_current[
    (df_current[main_delay_col] == "延期") &
    (df_current[warehouse_delay_col] == "延期")
    ].copy()
# 2.5 总延期订单数（用于占比计算）
df_total_delay = df_current[df_current[main_delay_col] == "延期"].copy()
total_delay = len(df_total_delay)
total_normal = len(df_normal)
total_current = len(df_current)

# --------------------------
# 3. 无延期订单时的展示
# --------------------------
if total_delay == 0:
    st.success("✅ 本月无延期订单，各物流环节时效均符合预期！")
    # 仅展示正常订单的各环节均值
    st.markdown("### 📈 各环节耗时均值（仅正常订单）")
    normal_mean = df_normal[all_stage_cols].mean().round(2)
    for stage in all_stage_cols:
        st.markdown(f"- **{stage}**：正常均值 {float(normal_mean[stage])} 天")
else:
    # --------------------------
    # 4. 统计三类延期订单数（精准匹配）
    # --------------------------
    forwarder_count = int(len(df_forwarder_delay))
    other_count = int(len(df_other_delay))
    warehouse_count = int(len(df_warehouse_delay))

    # 计算占比（纯Python原生计算，防错）
    forwarder_pct = round((forwarder_count / total_delay) * 100, 1) if total_delay > 0 else 0.0
    other_pct = round((other_count / total_delay) * 100, 1) if total_delay > 0 else 0.0
    warehouse_pct = round((warehouse_count / total_delay) * 100, 1) if total_delay > 0 else 0.0
    normal_pct = round((total_normal / total_current) * 100, 1) if total_current > 0 else 0.0
    delay_pct = round((total_delay / total_current) * 100, 1) if total_current > 0 else 0.0

    # --------------------------
    # 5. 基础数据汇总
    # --------------------------
    st.markdown(f"""
    ### 📊 基础数据
    - 当月总订单数：{total_current} 单
    - 正常订单数：{total_normal} 单（占比 {normal_pct}%）
    - 延期订单数：{total_delay} 单（占比 {delay_pct}%）
    """)

    # --------------------------
    # 6. 延期订单主因占比（修正为三类）
    # --------------------------
    st.markdown("### 🎯 延期订单主因占比")
    st.markdown(f"- **货代原因（发货-起飞/到港-提取）**：{forwarder_count} 单（占延期订单的 {forwarder_pct}%）")
    st.markdown(f"- **其他原因（提取-签收）**：{other_count} 单（占延期订单的 {other_pct}%）")
    st.markdown(f"- **仓库原因（签收-完成上架）**：{warehouse_count} 单（占延期订单的 {warehouse_pct}%）")

    # --------------------------
    # 7. 合并展示+红色异常标记（全大白话，无专业术语）
    # --------------------------
    st.markdown("### 📈 各环节耗时对比（大白话版）")
    normal_mean = df_normal[all_stage_cols].mean().round(2)
    # 预计算各环节均值
    forwarder_delay_mean = df_forwarder_delay[forwarder_stage_col].mean().round(2) if forwarder_count > 0 else 0.0
    other_delay_mean = {col: df_other_delay[col].mean().round(2) for col in other_stage_cols} if other_count > 0 else {}
    warehouse_delay_mean = df_warehouse_delay[warehouse_stage_col].mean().round(2) if warehouse_count > 0 else 0.0
    abnormal_threshold = 2.2  # 超过正常2.2倍=严重超时

    # 7.1 货代环节（仅发货-提取）
    st.markdown("#### 🔹 货代环节（发货-提取）")
    n_mean = float(normal_mean[forwarder_stage_col])
    if forwarder_count > 0:
        d_mean = float(forwarder_delay_mean)
        slow_days = round(d_mean - n_mean, 2)
        # 大白话描述
        if slow_days <= 0:
            info = f"✅ 比正常还快 {-slow_days} 天"
        else:
            info = f"🔸 平均慢了 {slow_days} 天"
        # 严重超时标红
        if d_mean >= n_mean * abnormal_threshold:
            st.markdown(f"- **{forwarder_stage_col}**：正常 {n_mean} 天 → 货代延期 **:red[{d_mean} 天]** | **:red[严重超时，慢了 {slow_days} 天]**")
        else:
            st.markdown(f"- **{forwarder_stage_col}**：正常 {n_mean} 天 → 货代延期 {d_mean} 天 | {info}")
    else:
        st.markdown(f"- **{forwarder_stage_col}**：正常 {n_mean} 天 | 无货代延期订单")

    # 7.2 其他原因环节（提取-到港/到港-签收）
    st.markdown("#### 🔹 其他原因环节（提取-签收）")
    for stage in other_stage_cols:
        n_mean = float(normal_mean[stage])
        if other_count > 0:
            d_mean = other_delay_mean.get(stage, 0.0)
            slow_days = round(d_mean - n_mean, 2)
            if slow_days <= 0:
                info = f"✅ 比正常还快 {-slow_days} 天"
            else:
                info = f"🔸 平均慢了 {slow_days} 天"
            if d_mean >= n_mean * abnormal_threshold:
                st.markdown(f"- **{stage}**：正常 {n_mean} 天 → 延期订单 **:red[{d_mean} 天]** | **:red[严重超时，慢了 {slow_days} 天]**")
            else:
                st.markdown(f"- **{stage}**：正常 {n_mean} 天 → 延期订单 {d_mean} 天 | {info}")
        else:
            st.markdown(f"- **{stage}**：正常 {n_mean} 天 | 无该环节延期订单")

    # 7.3 仓库环节（签收-完成上架）
    st.markdown("#### 🔹 仓库环节（签收-完成上架）")
    n_mean = float(normal_mean[warehouse_stage_col])
    if warehouse_count > 0:
        d_mean = float(warehouse_delay_mean)
        slow_days = round(d_mean - n_mean, 2)
        if slow_days <= 0:
            info = f"✅ 比正常还快 {-slow_days} 天"
        else:
            info = f"🔸 平均慢了 {slow_days} 天"
        if d_mean >= n_mean * abnormal_threshold:
            st.markdown(f"- **{warehouse_stage_col}**：正常 {n_mean} 天 → 仓库延期 **:red[{d_mean} 天]** | **:red[严重超时，慢了 {slow_days} 天]**")
        else:
            st.markdown(f"- **{warehouse_stage_col}**：正常 {n_mean} 天 → 仓库延期 {d_mean} 天 | {info}")
    else:
        st.markdown(f"- **{warehouse_stage_col}**：正常 {n_mean} 天 | 无仓库延期订单")

    # --------------------------
    # 8. 针对性优化建议（大白话）
    # --------------------------
    st.markdown("### 💡 优化建议")
    suggestions = []
    # 货代建议
    if forwarder_count > 0:
        if forwarder_delay_mean >= normal_mean[forwarder_stage_col] * abnormal_threshold:
            suggestions.append(f"⚠️ 货代环节【{forwarder_stage_col}】严重超时，平均慢了 {round(forwarder_delay_mean - normal_mean[forwarder_stage_col], 2)} 天，需立刻跟进货代改善揽收/提取效率。")
        else:
            suggestions.append(f"⚠️ 货代环节【{forwarder_stage_col}】存在延期，平均慢了 {round(forwarder_delay_mean - normal_mean[forwarder_stage_col], 2)} 天，建议与货代沟通优化。")
    # 其他原因建议
    if other_count > 0:
        abnormal_other_stages = [
            stage for stage in other_stage_cols
            if other_delay_mean.get(stage, 0.0) >= normal_mean[stage] * abnormal_threshold
        ]
        if abnormal_other_stages:
            suggestions.append(f"⚠️ 其他原因环节【{'、'.join(abnormal_other_stages)}】严重超时，需关注航线稳定性、清关效率或末端派送服务。")
        else:
            suggestions.append(f"⚠️ 其他原因环节（提取-签收）存在延期，建议排查物流链路中间环节问题。")
    # 仓库建议
    if warehouse_count > 0:
        if warehouse_delay_mean >= normal_mean[warehouse_stage_col] * abnormal_threshold:
            suggestions.append(f"⚠️ 仓库环节【{warehouse_stage_col}】严重超时，平均慢了 {round(warehouse_delay_mean - normal_mean[warehouse_stage_col], 2)} 天，需紧急优化仓内操作流程、增加上架人手或调整优先级。")
        else:
            suggestions.append(f"⚠️ 仓库环节【{warehouse_stage_col}】存在延期，平均慢了 {round(warehouse_delay_mean - normal_mean[warehouse_stage_col], 2)} 天，建议优化仓内协作效率。")
    # 无建议时的提示
    if not suggestions:
        suggestions.append("💡 各环节延期均在合理范围，建议维持现有操作流程并持续监控。")
    # 展示建议
    for idx, suggestion in enumerate(suggestions, 1):
        st.markdown(f"{idx}. {suggestion}")

# ---------------------- 货代准时率-物流时效分析（终极无报错版） ----------------------
st.divider()
st.subheader("📦 物流方式-准时率对应物流时效分析（上架完成-发货时间）")

# ====================== 1. 全局变量初始化（核心：避免未定义报错） ======================
target_rates = [75, 80, 85, 90, 95, 100]  # 目标累计占比（准时率）
time_col = "上架完成-发货时间"  # 核心统计列
logistics_col = "物流方式"  # 物流方式列
is_all_logistics = False  # 是否筛选全部物流方式
all_results = []  # 所有物流方式的计算结果
group_data = {}  # 每个物流方式的排序后数据
unique_logistics = []  # 唯一物流方式列表
df_analysis = pd.DataFrame()  # 预处理后的数据集

# ====================== 2. 数据预处理（容错处理） ======================
# 检查核心列是否存在
if time_col not in df_current.columns or logistics_col not in df_current.columns:
    st.warning(f"⚠️ 数据中缺失核心列：「{time_col}」 或 「{logistics_col}」")
else:
    # 复制数据并清洗：仅保留有效数据
    df_analysis = df_current.copy()
    df_analysis[time_col] = pd.to_numeric(df_analysis[time_col], errors="coerce")
    # 过滤条件：时效>0 且 物流方式非空
    df_analysis = df_analysis[
        (df_analysis[time_col] > 0) &
        (df_analysis[logistics_col].notna())
        ].reset_index(drop=True)

    # 检查清洗后是否有数据
    if len(df_analysis) == 0:
        st.warning("⚠️ 无有效数据（时效为空/≤0 或 物流方式为空）")
    else:
        # 获取唯一物流方式并判断是否为「全部」筛选
        unique_logistics = df_analysis[logistics_col].unique()
        is_all_logistics = len(unique_logistics) > 1

        # ====================== 3. 核心计算逻辑 ======================
        if is_all_logistics:
            st.info(f"✅ 当前筛选为「全部」，按「{logistics_col}」分组分析（共 {len(unique_logistics)} 种方式）")
            # 遍历每种物流方式单独计算
            for logistics_type in unique_logistics:
                df_group = df_analysis[df_analysis[logistics_col] == logistics_type].copy()
                group_total = len(df_group)
                if group_total == 0:
                    continue

                # 按时效升序排序
                df_sorted = df_group.sort_values(by=time_col, ascending=True).reset_index(drop=True)
                # 计算累计订单数和累计占比
                df_sorted["累计订单数"] = range(1, group_total + 1)
                df_sorted["累计占比(%)"] = (df_sorted["累计订单数"] / group_total) * 100
                group_data[logistics_type] = df_sorted

                # 匹配每个目标准时率的时效阈值
                for target_rate in target_rates:
                    df_matched = df_sorted[df_sorted["累计占比(%)"] >= target_rate]
                    if not df_matched.empty:
                        min_time = df_matched[time_col].min()
                        actual_rate = df_matched[df_matched[time_col] == min_time]["累计占比(%)"].iloc[0]
                        pass_orders = len(df_sorted[df_sorted[time_col] <= min_time])
                        all_results.append({
                            "物流方式": logistics_type,
                            "目标准时率(%)": target_rate,
                            "实际累计占比(%)": round(actual_rate, 1),
                            "对应时效上限(天)": round(min_time, 1),
                            "达标订单数": pass_orders,
                            "总订单数": group_total
                        })
                    else:
                        all_results.append({
                            "物流方式": logistics_type,
                            "目标准时率(%)": target_rate,
                            "实际累计占比(%)": "-",
                            "对应时效上限(天)": "-",
                            "达标订单数": 0,
                            "总订单数": group_total
                        })
        else:
            # 单个物流方式计算
            logistics_type = unique_logistics[0] if len(unique_logistics) > 0 else ""
            if logistics_type:
                df_sorted = df_analysis.sort_values(by=time_col, ascending=True).reset_index(drop=True)
                group_total = len(df_sorted)
                df_sorted["累计订单数"] = range(1, group_total + 1)
                df_sorted["累计占比(%)"] = (df_sorted["累计订单数"] / group_total) * 100
                group_data[logistics_type] = df_sorted

                # 匹配每个目标准时率的时效阈值
                for target_rate in target_rates:
                    df_matched = df_sorted[df_sorted["累计占比(%)"] >= target_rate]
                    if not df_matched.empty:
                        min_time = df_matched[time_col].min()
                        actual_rate = df_matched[df_matched[time_col] == min_time]["累计占比(%)"].iloc[0]
                        pass_orders = len(df_sorted[df_sorted[time_col] <= min_time])
                        all_results.append({
                            "物流方式": logistics_type,
                            "目标准时率(%)": target_rate,
                            "实际累计占比(%)": round(actual_rate, 1),
                            "对应时效上限(天)": round(min_time, 1),
                            "达标订单数": pass_orders,
                            "总订单数": group_total
                        })
                    else:
                        all_results.append({
                            "物流方式": logistics_type,
                            "目标准时率(%)": target_rate,
                            "实际累计占比(%)": "-",
                            "对应时效上限(天)": "-",
                            "达标订单数": 0,
                            "总订单数": group_total
                        })

# ====================== 4. 展示结果总表 ======================
if all_results:
    st.markdown("#### 📊 各物流方式-准时率-时效阈值对应表")
    df_results = pd.DataFrame(all_results)
    st.dataframe(
        df_results,
        use_container_width=True,
        column_config={
            "物流方式": st.column_config.TextColumn("物流方式"),
            "目标准时率(%)": st.column_config.NumberColumn("目标准时率(%)", format="%d"),
            "实际累计占比(%)": st.column_config.NumberColumn("实际累计占比(%)", format="%.1f"),
            "对应时效上限(天)": st.column_config.NumberColumn("时效上限(天)", format="%.1f"),
            "达标订单数": st.column_config.NumberColumn("达标订单数", format="%d"),
            "总订单数": st.column_config.NumberColumn("总订单数", format="%d")
        }
    )
else:
    st.info("ℹ️ 暂无有效数据生成时效阈值表")

# ====================== 5. 可视化：分开展示独立图表（核心优化） ======================
st.markdown("#### 📈 各物流方式时效分布 & 累计准时率分析")
if group_data and len(group_data) > 0:  # 确保group_data有数据
    if is_all_logistics:
        # 筛选「全部」：为每个物流方式生成独立图表
        import plotly.graph_objects as go
        import plotly.express as px
        from plotly.subplots import make_subplots

        # 分配专属颜色
        color_palette = px.colors.qualitative.Plotly
        logistics_colors = {lt: color_palette[i % len(color_palette)] for i, lt in enumerate(unique_logistics)}

        # 遍历生成每个物流方式的图表
        for idx, logistics_type in enumerate(unique_logistics, 1):
            if logistics_type not in group_data:
                continue
            df_sorted = group_data[logistics_type]
            time_count = df_sorted[time_col].value_counts().sort_index().reset_index()
            time_count.columns = [time_col, "订单数"]

            # 创建双轴图
            fig_single = make_subplots(specs=[[{"secondary_y": True}]])

            # 1. 柱形图：时效分布
            fig_single.add_trace(
                go.Bar(
                    x=time_count[time_col],
                    y=time_count["订单数"],
                    name="各时效订单数",
                    marker_color=logistics_colors[logistics_type],
                    opacity=0.7,
                    hovertemplate=f"{logistics_type}<br>时效：%{{x}}天<br>订单数：%{{y}}单<extra></extra>"
                ),
                secondary_y=False
            )

            # 2. 折线图：累计占比
            fig_single.add_trace(
                go.Scatter(
                    x=df_sorted[time_col],
                    y=df_sorted["累计占比(%)"],
                    name="累计占比（准时率）",
                    line=dict(color=logistics_colors[logistics_type], width=3, shape="spline"),
                    marker=dict(color=logistics_colors[logistics_type], size=5),
                    hovertemplate=f"{logistics_type}<br>时效：%{{x}}天<br>累计准时率：%{{y:.1f}}%<extra></extra>"
                ),
                secondary_y=True
            )

            # 3. 阈值散点标注
            scatter_x = []
            scatter_y = []
            scatter_text = []
            for res in all_results:
                if res["物流方式"] == logistics_type and res["对应时效上限(天)"] != "-":
                    scatter_x.append(res["对应时效上限(天)"])
                    scatter_y.append(res["实际累计占比(%)"])
                    scatter_text.append(f"{res['目标准时率(%)']}% → {res['对应时效上限(天)']}天")
            if scatter_x:
                fig_single.add_trace(
                    go.Scatter(
                        x=scatter_x,
                        y=scatter_y,
                        mode="markers+text",
                        text=scatter_text,
                        textposition="top center",
                        marker=dict(color="darkblue", size=10, symbol="star"),
                        showlegend=False,
                        hovertemplate="目标：%{text}<extra></extra>"
                    ),
                    secondary_y=True
                )

            # 4. 参考线
            for rate in target_rates:
                fig_single.add_hline(
                    y=rate,
                    line_dash="dash",
                    line_color="gray",
                    opacity=0.5,
                    secondary_y=True,
                    annotation_text=f"{rate}% 目标",
                    annotation_position="right",
                    annotation_font={"size": 9, "color": "gray"}
                )

            # 5. 样式优化
            fig_single.update_layout(
                height=450,
                title=f"({idx}/{len(unique_logistics)}) {logistics_type} - 时效分布 & 累计准时率",
                xaxis_title="上架完成-发货时间(天)",
                xaxis=dict(tickangle=0, showgrid=False),
                yaxis=dict(title="订单数", showgrid=True, gridcolor="lightgray"),
                yaxis2=dict(title="累计占比（准时率）(%)", range=[0, 105], showgrid=False),
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                template="simple_white",
                margin=dict(b=80)
            )

            st.plotly_chart(fig_single, use_container_width=True)
            st.divider()  # 分隔不同物流方式的图表
    else:
        # 筛选「单个」：单物流方式图表
        logistics_type = next(iter(group_data.keys()))
        df_sorted = group_data[logistics_type]
        time_count = df_sorted[time_col].value_counts().sort_index().reset_index()
        time_count.columns = [time_col, "订单数"]

        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # 柱形图
        fig.add_trace(
            go.Bar(
                x=time_count[time_col],
                y=time_count["订单数"],
                name="各时效订单数",
                marker_color="#87CEEB",
                opacity=0.7,
                hovertemplate="时效：%{x}天<br>订单数：%{y}单<extra></extra>"
            ),
            secondary_y=False
        )

        # 折线图
        fig.add_trace(
            go.Scatter(
                x=df_sorted[time_col],
                y=df_sorted["累计占比(%)"],
                name="累计占比（准时率）",
                line=dict(color="red", width=2, shape="spline"),
                hovertemplate="时效：%{x}天<br>累计占比：%{y:.1f}%<extra></extra>"
            ),
            secondary_y=True
        )

        # 阈值散点
        scatter_x = []
        scatter_y = []
        scatter_text = []
        for res in all_results:
            if res["对应时效上限(天)"] != "-":
                scatter_x.append(res["对应时效上限(天)"])
                scatter_y.append(res["实际累计占比(%)"])
                scatter_text.append(f"{res['目标准时率(%)']}% → {res['对应时效上限(天)']}天")
        if scatter_x:
            fig.add_trace(
                go.Scatter(
                    x=scatter_x,
                    y=scatter_y,
                    mode="markers+text",
                    text=scatter_text,
                    textposition="top center",
                    marker=dict(color="darkblue", size=10, symbol="star"),
                    showlegend=False,
                    hovertemplate="目标：%{text}<extra></extra>"
                ),
                secondary_y=True
            )

        # 参考线
        for rate in target_rates:
            fig.add_hline(
                y=rate,
                line_dash="dash",
                line_color="orange",
                opacity=0.5,
                secondary_y=True,
                annotation_text=f"{rate}%",
                annotation_position="right"
            )

        # 样式
        fig.update_layout(
            height=550,
            title=f"{logistics_type} - 时效分布 & 累计准时率",
            xaxis_title="上架完成-发货时间(天)",
            yaxis=dict(title="订单数"),
            yaxis2=dict(title="累计占比（准时率）(%)", range=[0, 105]),
            template="simple_white"
        )

        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("ℹ️ 暂无有效数据生成对比图表")

# ====================== 6. 业务解读 ======================
st.markdown("#### 📝 各物流方式核心结论")
if all_results:
    # 90%准时率汇总对比
    df_summary = pd.DataFrame(all_results)
    rate_90_summary = df_summary[df_summary["目标准时率(%)"] == 90].copy()
    if not rate_90_summary.empty:
        st.markdown("##### 🔍 90%准时率核心对比")
        display_cols = ["物流方式", "对应时效上限(天)", "达标订单数", "总订单数"]
        st.dataframe(
            rate_90_summary[display_cols],
            use_container_width=True,
            column_config={
                "物流方式": st.column_config.TextColumn("物流方式"),
                "对应时效上限(天)": st.column_config.NumberColumn("时效上限(天)", format="%.1f"),
                "达标订单数": st.column_config.NumberColumn("达标订单数"),
                "总订单数": st.column_config.NumberColumn("总订单数")
            }
        )

    # 逐方式详细解读
    st.markdown("##### 📋 各方式详细结论")
    for logistics_type in unique_logistics:
        lt_results = [r for r in all_results if r["物流方式"] == logistics_type]
        if not lt_results:
            continue

        rate_75 = next((r for r in lt_results if r["目标准时率(%)"] == 75), None)
        rate_90 = next((r for r in lt_results if r["目标准时率(%)"] == 90), None)
        rate_100 = next((r for r in lt_results if r["目标准时率(%)"] == 100), None)

        desc = f"**{logistics_type}**：<br>"
        if rate_75 and rate_75["对应时效上限(天)"] != "-":
            desc += f"- 75%准时率：时效≤{rate_75['对应时效上限(天)']}天（达标{rate_75['达标订单数']}单）<br>"
        if rate_90 and rate_90["对应时效上限(天)"] != "-":
            desc += f"- 90%准时率：时效≤{rate_90['对应时效上限(天)']}天（达标{rate_90['达标订单数']}单）<br>"
        if rate_100 and rate_100["对应时效上限(天)"] != "-":
            desc += f"- 100%准时率：时效≤{rate_100['对应时效上限(天)']}天（覆盖所有{rate_100['总订单数']}单）<br>"

        st.markdown(desc, unsafe_allow_html=True)
else:
    st.info("ℹ️ 暂无有效数据生成分析解读")

# ---------------------- 货代准时情况分析（独立版：发货-签收环节，无仓库关联） ----------------------
st.markdown("### 货代准时情况分析（发货-提取环节）")

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
    freight_display["时效差值均值"] = freight_display["时效差值均值"].apply(lambda x: f"{x:.2f}")
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
df_warehouse_valid = df_current_FBA[
    (df_current_FBA[WAREHOUSE_COLUMN_MAPPING["仓库列名"]].notna() &
     (df_current_FBA[WAREHOUSE_COLUMN_MAPPING["仓库列名"]] != "")) &
    (df_current_FBA[WAREHOUSE_COLUMN_MAPPING["签收上架时长列名"]].notna())
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
    # 4.1 上方：仓库订单量占比 + 准时率 双轴图（全屏宽度）
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

    # 4.2 下方：所有仓库核心表现迷你卡片（一行3列 + 按优质→合格→异常排序）
    st.markdown("#### 仓库核心表现")


    # 第一步：给仓库数据添加评级标识，用于排序
    def get_grade_flag(rate):
        """返回排序标识：优质=0，合格=1，异常=2"""
        if rate >= 90:
            return 0
        elif rate >= 80:
            return 1
        else:
            return 2


    # 新增排序列
    warehouse_stats["评级排序"] = warehouse_stats["准时率(%)"].apply(get_grade_flag)
    # 按评级排序（优质→合格→异常），同评级按准时率降序
    warehouse_stats_sorted = warehouse_stats.sort_values(
        by=["评级排序", "准时率(%)"],
        ascending=[True, False]
    ).reset_index(drop=True)

    # 第二步：一行3列展示卡片（兼容不足3个的情况）
    from itertools import zip_longest

    # 每3个仓库分为一组
    warehouse_groups = list(zip_longest(*[iter(warehouse_stats_sorted.to_dict('records'))] * 3))

    # 第三步：循环渲染每组的3列卡片
    for group in warehouse_groups:
        # 创建3列布局
        col1, col2, col3 = st.columns(3)
        cols = [col1, col2, col3]

        # 为每组内的每个仓库渲染卡片
        for idx, warehouse in enumerate(group):
            if warehouse is None:  # 处理最后一组不足3个的情况
                continue
            with cols[idx]:
                # 准时率颜色分级：优质≥90% | 合格80-90% | 异常<80%
                if warehouse["准时率(%)"] >= 90:
                    card_bg = "#f0f8f0"
                    rate_color = "#2e7d32"
                    tag = "优质"
                    level_desc = "准时率表现优秀"
                elif warehouse["准时率(%)"] >= 80:
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
                diff_val = warehouse["准时率差值(%)"]
                prev_rate = warehouse["上月准时率(%)"]
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

                # 生成仓库迷你卡片（保留原有样式，适配列布局）
                st.markdown(f"""
                <div style='background-color: {card_bg}; padding: 12px; border-radius: 8px; margin-bottom: 12px; border-left: 4px solid {rate_color};'>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <p style='margin: 0; font-weight: bold; font-size: 16px;'>{warehouse['仓库']}</p>
                        <span style='font-size: 12px; padding: 2px 6px; border-radius: 12px; background: {rate_color}; color: white;'>{tag}</span>
                    </div>
                    <p style='margin: 6px 0 0; font-size: 14px;'>
                        准时率：<span style='color: {rate_color}; font-weight: bold; font-size: 18px;'>{warehouse['准时率(%)']:.2f}%</span>
                    </p>
                    <p style='margin: 4px 0 0; font-size: 12px; color: #666;'>订单：{warehouse['总订单数']}单（{warehouse['订单量占比(%)']:.2f}%）</p>
                    <p style='margin: 4px 0 0; font-size: 12px; color: #666;'>差值：<span style='color: {diff_color}; font-weight: bold;'>{diff_text}</span> {prev_rate_text}</p>
                    <p style='margin: 4px 0 0; font-size: 12px; color: #666;'>最长上架时长：{warehouse['最长上架时长']:.1f}天</p>
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

    # ===================== 强制固定 2 位小数（核心修复） =====================
    # 强制转为数值 → 四舍五入2位 → 格式化为字符串 "xx.xx"
    warehouse_display["签收上架时长均值"] = warehouse_stats["签收上架时长均值"].astype(float).round(2)
    warehouse_display["签收上架时长均值"] = warehouse_display["签收上架时长均值"].apply(lambda x: f"{x:.2f}")

    # 最短/最长也统一格式
    warehouse_display["最短上架时长"] = warehouse_stats["最短上架时长"].astype(float).round(1).apply(
        lambda x: f"{x:.1f}天")
    warehouse_display["最长上架时长"] = warehouse_stats["最长上架时长"].astype(float).round(1).apply(
        lambda x: f"{x:.1f}天")

    # 百分比列格式化
    for col in ["订单量占比(%)", "延期率(%)", "准时率(%)", "上月准时率(%)", "准时率差值(%)"]:
        warehouse_display[col] = warehouse_display[col].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "")


    # 表格高亮规则
    def highlight_warehouse(row):
        styles = [""] * len(row)
        # 准时率差值为负标红
        if row["准时率差值(%)"] and isinstance(row["准时率差值(%)"], str):
            val = float(row["准时率差值(%)"].replace("%", ""))
            if val < 0:
                styles[display_cols.index(
                    "准时率差值(%)")] = "background-color: #fff5f5; color: #c62828; font-weight: bold;"
        # 延期率>20%标红
        if row["延期率(%)"] and isinstance(row["延期率(%)"], str):
            val = float(row["延期率(%)"].replace("%", ""))
            if val > 20:
                styles[
                    display_cols.index("延期率(%)")] = "background-color: #fff5f5; color: #c62828; font-weight: bold;"
        # 准时率<80%标红
        if row["准时率(%)"] and isinstance(row["准时率(%)"], str):
            val = float(row["准时率(%)"].replace("%", ""))
            if val < 80:
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
st.markdown("## 📈 不同月份整体趋势分析")
st.divider()

# ===== 1. 数据预处理（先加物流方式筛选，再聚合年月）=====
# ---------------------- 【修改1】：新增物流方式列校验 ----------------------
required_cols = ["到货年月", "FBA号", "提前/延期(整体)", "物流方式"]  # 新增：物流方式
missing_cols = [col for col in required_cols if col not in df_selected.columns]
if missing_cols:
    st.error(f"缺少月度分析必要列：{missing_cols}，请检查数据列名！")
else:
    # ---------------------- 【新增1】：物流方式筛选器（核心新增） ----------------------
    st.markdown("### 筛选条件")  # 保留原标题，位置提前
    # 新增：控制筛选器列宽，界面更美观
    col_logistics, col_empty = st.columns([1, 3])
    with col_logistics:
        # 新增：获取唯一的物流方式（去重+排序）
        unique_logistics = sorted(df_selected["物流方式"].dropna().unique())
        # 新增：添加"全部"选项，默认选中
        logistics_options = ["全部"] + unique_logistics
        selected_logistics = st.selectbox(
            "物流方式",
            options=logistics_options,
            index=0,
            key="selected_logistics"
        )

    # 新增：根据选中的物流方式过滤原始数据
    if selected_logistics == "全部":
        df_filtered_by_logistics = df_selected.copy()  # 全部物流方式
    else:
        df_filtered_by_logistics = df_selected[df_selected["物流方式"] == selected_logistics].copy()

    # 新增：容错处理 - 筛选后无数据的提示
    if len(df_filtered_by_logistics) == 0:
        st.warning(f"所选物流方式「{selected_logistics}」暂无数据")
    else:
        # ---------------------- 【修改2】：聚合数据从df_selected改为df_filtered_by_logistics ----------------------
        # 按到货年月分组计算核心指标（基于筛选后的物流方式数据）
        monthly_stats = df_filtered_by_logistics.groupby("到货年月").agg(
            总订单数=("FBA号", "count"),
            提前准时订单数=("提前/延期(整体)", lambda x: len(x[x == "提前/准时"])),
            延期订单数=("提前/延期(整体)", lambda x: len(x[x == "延期"]))
        ).reset_index()

        # 计算准时率（保留2位小数）
        monthly_stats["准时率(%)"] = round(monthly_stats["提前准时订单数"] / monthly_stats["总订单数"] * 100, 2)


        # ---------------------- 【修改3】：新增日期解析容错（避免报错） ----------------------
        # 生成中文月份标签（如：2026年1月）
        # 新增：容错函数 - 处理到货年月格式不统一的问题
        def safe_parse_ym(ym):
            try:
                return pd.to_datetime(str(ym) + "-01")
            except:
                return pd.NaT


        # 时间计算工具函数：根据最新月份，计算往前推N个月的年月（格式YYYY-MM）
        def calc_prev_n_month(latest_ym, n):
            try:
                latest_dt = pd.to_datetime(latest_ym + "-01")
                prev_dt = latest_dt - pd.DateOffset(months=n - 1)
                return prev_dt.strftime("%Y-%m")
            except:
                return latest_ym


        # 修改：用容错函数解析日期
        monthly_stats["年月排序"] = monthly_stats["到货年月"].apply(safe_parse_ym)
        # 新增：过滤无效日期行
        monthly_stats = monthly_stats[monthly_stats["年月排序"].notna()].copy()

        if len(monthly_stats) == 0:
            st.warning("暂无有效月份数据可分析")
        else:
            monthly_stats["中文月份"] = monthly_stats["年月排序"].dt.strftime("%Y年%m月")
            # 按时间正序排序（图表从左到右时间递增）
            monthly_stats = monthly_stats.sort_values("年月排序", ascending=True).reset_index(drop=True)

            # 计算环比变化（总订单数、准时率）
            monthly_stats["总订单数环比变化"] = monthly_stats["总订单数"].diff(1).fillna(0)
            monthly_stats["准时率环比变化(百分点)"] = monthly_stats["准时率(%)"].diff(1).fillna(0)

            # ===== 2. 新增【自动化快捷筛选器】+ 保留原有自定义筛选（核心修改）=====
            # 自动提取数据中最新的月份（作为时间基准，无需手动改）
            latest_ym = monthly_stats["到货年月"].max()
            # 快捷筛选器：一行两列布局（快捷筛选+说明），更美观
            col_quick, col_tip = st.columns([2, 3])
            with col_quick:
                st.markdown("#### 快捷时间筛选")
                quick_options = ["自定义时间范围", "上个月", "近三个月", "近半年", "近一年"]
                selected_quick = st.selectbox(
                    "快速选择",
                    options=quick_options,
                    index=0,
                    key="quick_time_filter",
                    label_visibility="collapsed"
                )
            with col_tip:
                st.markdown(f"#### 筛选基准（自动获取）")
                st.caption(f"📌 数据最新月份：{latest_ym}")
                st.caption(f"💡 快捷筛选均以最新月份为结束时间")

            # 初始化起始/结束年月
            start_ym = None
            end_ym = latest_ym

            # 自动化计算各快捷筛选的时间范围
            if selected_quick == "上个月":
                start_ym = latest_ym  # 上个月=仅最新月份
            elif selected_quick == "近三个月":
                start_ym = calc_prev_n_month(latest_ym, 3)  # 往前推2个月，共3个月
            elif selected_quick == "近半年":
                start_ym = calc_prev_n_month(latest_ym, 6)  # 往前推5个月，共6个月
            elif selected_quick == "近一年":
                start_ym = calc_prev_n_month(latest_ym, 12)  # 往前推11个月，共12个月
            else:
                # 自定义时间范围：保留原有双下拉框
                st.markdown("#### 自定义时间范围")
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
                # 转换回原始年月格式
                start_ym = monthly_stats[monthly_stats["中文月份"] == start_month]["到货年月"].iloc[0]
                end_ym = monthly_stats[monthly_stats["中文月份"] == end_month]["到货年月"].iloc[0]

            # 最终数据筛选（统一逻辑，兼容快捷/自定义）
            df_filtered = monthly_stats[
                (monthly_stats["到货年月"] >= start_ym) &
                (monthly_stats["到货年月"] <= end_ym)
                ].copy()
            df_filtered = df_filtered.sort_values("年月排序", ascending=True).reset_index(drop=True)

            # ---------------------- 【新增2】：筛选结果展示（优化） ----------------------
            start_cn = df_filtered["中文月份"].min() if len(df_filtered) > 0 else ""
            end_cn = df_filtered["中文月份"].max() if len(df_filtered) > 0 else ""
            st.info(f"✅ 筛选结果：{selected_logistics} | {start_cn} 至 {end_cn}（共{len(df_filtered)}个月份）")
            # 可选：展示核心筛选数据预览
            st.dataframe(df_filtered[["中文月份", "总订单数", "准时率(%)"]], use_container_width=True, hide_index=True)

            # ===== 后续原有逻辑（无需任何修改，直接保留）=====
            # 计算平均准时率（用于红色虚线）
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

            # ========== 关键修改：强制准时率保留2位小数 ==========
            df_display["准时率(%)"] = df_display["准时率(%)"].apply(lambda x: f"{x:.2f}")


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

# ===== 【简化版】多物流方式-多阈值时效趋势折线图（新增累计订单数）=====
st.markdown("### 📦 各物流方式-不同准时率阈值时效趋势")
st.divider()

# 1. 数据预处理
time_col = "上架完成-发货时间"
target_rates = [75, 80, 85, 90, 95, 100]  # 仅保留这6个阈值
df_time_analysis = df_filtered_by_logistics.copy()

# 只保留筛选时间范围内的月份
selected_ym_list = df_filtered["到货年月"].tolist()
df_time_analysis = df_time_analysis[df_time_analysis["到货年月"].isin(selected_ym_list)].copy()

# 清洗时效列
df_time_analysis[time_col] = pd.to_numeric(df_time_analysis[time_col], errors="coerce")
df_time_analysis = df_time_analysis[
    (df_time_analysis[time_col] > 0) &
    (df_time_analysis["到货年月"].notna()) &
    (df_time_analysis["物流方式"].notna())
    ].reset_index(drop=True)

if len(df_time_analysis) == 0:
    st.warning("暂无有效时效数据（时效为空/≤0 或 物流方式为空）")
else:
    # 2. 核心计算：按【物流方式+月份+阈值】计算时效上限（新增累计订单数）
    trend_results = []
    # 获取所有需要分析的物流方式（根据筛选条件）
    if selected_logistics == "全部":
        analysis_logistics = sorted(df_time_analysis["物流方式"].unique())
    else:
        analysis_logistics = [selected_logistics]

    # 遍历每个物流方式
    for logistics_type in analysis_logistics:
        df_logistics = df_time_analysis[df_time_analysis["物流方式"] == logistics_type].copy()
        # 遍历每个月份
        for ym in sorted(df_logistics["到货年月"].unique()):
            df_month = df_logistics[df_logistics["到货年月"] == ym].copy()
            month_total = len(df_month)
            if month_total < 1:  # 数据量为0的剔除，避免无意义计算
                continue

            # 按时效升序排序并计算累计占比
            df_month_sorted = df_month.sort_values(by=time_col, ascending=True).reset_index(drop=True)
            df_month_sorted["累计订单数"] = range(1, month_total + 1)
            df_month_sorted["累计占比(%)"] = (df_month_sorted["累计订单数"] / month_total) * 100

            # 关联中文月份名称
            month_cn = monthly_stats[monthly_stats["到货年月"] == ym]["中文月份"].iloc[0]

            # 遍历每个阈值计算时效上限+累计订单数
            for target_rate in target_rates:
                df_matched = df_month_sorted[df_month_sorted["累计占比(%)"] >= target_rate]
                if not df_matched.empty:
                    min_time = round(df_matched[time_col].min(), 1)  # 保留1位小数
                    # 新增：计算该阈值下的累计订单数
                    pass_orders = len(df_month_sorted[df_month_sorted[time_col] <= min_time])
                    trend_results.append({
                        "物流方式": logistics_type,
                        "到货年月": ym,
                        "中文月份": month_cn,
                        "准时率阈值(%)": target_rate,
                        "时效上限(天)": min_time,
                        "当月总订单数": month_total,  # 新增
                        "达标累计订单数": pass_orders  # 新增：达到该阈值的累计订单数
                    })

    # 3. 生成折线图（每个物流方式1张图）
    if trend_results:
        import plotly.graph_objects as go
        import pandas as pd

        df_trend = pd.DataFrame(trend_results)
        # 按中文月份排序（保证时间顺序）
        df_trend["年月排序"] = df_trend["中文月份"].apply(
            lambda x: pd.to_datetime(x.replace("年", "-").replace("月", "-01")))
        df_trend = df_trend.sort_values("年月排序")

        # 遍历每个物流方式生成独立折线图
        for logistics_type in analysis_logistics:
            df_single_log = df_trend[df_trend["物流方式"] == logistics_type].copy()
            if len(df_single_log) == 0:
                continue

            # 创建图表
            fig = go.Figure()

            # 为每个阈值生成一条折线
            for rate in target_rates:
                df_rate = df_single_log[df_single_log["准时率阈值(%)"] == rate].copy()
                if len(df_rate) > 0:
                    fig.add_trace(go.Scatter(
                        x=df_rate["中文月份"],
                        y=df_rate["时效上限(天)"],
                        name=f"{rate}%准时率",
                        mode="lines+markers+text",  # 线+点+文字标注
                        text=df_rate["时效上限(天)"].astype(str) + "天",  # 折点显示数值
                        textposition="top center",  # 文字在点上方
                        marker=dict(size=8),
                        line=dict(width=2)
                    ))

            # 图表样式配置
            fig.update_layout(
                title=f"【{logistics_type}】不同准时率阈值时效趋势",
                xaxis_title="到货年月",
                yaxis_title="时效上限（天）",
                xaxis=dict(tickangle=45),  # 横坐标文字倾斜45度，避免重叠
                yaxis=dict(gridcolor="#eee"),
                legend=dict(title="准时率阈值", x=0.02, y=0.98),
                height=500,
                plot_bgcolor="#fff",
                margin=dict(b=80)
            )

            # 显示图表
            st.plotly_chart(fig, use_container_width=True)
            st.divider()

        # 优化：原始数据表格（新增累计订单数）
        st.markdown("#### 📊 时效趋势原始数据（含累计订单数）")
        # 选择展示的列（调整顺序，突出核心信息）
        display_cols = [
            "物流方式", "中文月份", "准时率阈值(%)",
            "时效上限(天)", "当月总订单数", "达标累计订单数"
        ]
        st.dataframe(
            df_trend[display_cols],
            use_container_width=True,
            hide_index=True
        )

        # 数据下载（包含累计订单数）
        csv_trend = df_trend.to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            label="📥 下载时效趋势数据",
            data=csv_trend,
            file_name=f"物流时效趋势数据_{selected_logistics}.csv",
            mime="text/csv"
        )
    else:
        st.warning("暂无足够数据生成时效趋势图（单月数据量需≥5条）")

# ---------------------- 货代不同月份趋势分析 ----------------------
st.markdown("## 🚢 货代不同月份趋势分析")
st.divider()

# ===== 1. 数据预处理 & 列名校验 =====
FREIGHT_MONTH_COLUMN_MAPPING = {
    "货代列名": "货代",  # 替换为你实际的货代列名
    "到货年月列名": "到货年月",  # 替换为你实际的到货年月列名
    "提前延期列名": "提前/延期（货代）"  # 替换为你实际的提前/延期列名
}
# ---------------------- 【修改1】：新增物流方式列校验 ----------------------
required_cols = [
    FREIGHT_MONTH_COLUMN_MAPPING["货代列名"],
    FREIGHT_MONTH_COLUMN_MAPPING["到货年月列名"],
    FREIGHT_MONTH_COLUMN_MAPPING["提前延期列名"],
    "FBA号",  # 用于统计订单数
    "物流方式"  # 新增：物流方式列
]
missing_cols = [col for col in required_cols if col not in df_selected.columns]
if missing_cols:
    st.error(f"缺少货代月度分析必要列：{missing_cols}，请检查数据列名！")
else:
    # ---------------------- 【新增1】：物流方式筛选器（第一步筛选） ----------------------
    st.markdown("### 筛选条件")
    # 新增：物流方式筛选行（控制列宽）
    col_logistics, col_empty = st.columns([1, 3])
    with col_logistics:
        # 获取唯一的物流方式（去重+排序+去空）
        unique_logistics = sorted(df_selected["物流方式"].dropna().unique())
        # 新增"全部"选项，默认选中
        logistics_options = ["全部"] + unique_logistics
        selected_logistics = st.selectbox(
            "物流方式",
            options=logistics_options,
            index=0,
            key="freight_selected_logistics"  # 独立key，避免冲突
        )

    # 新增：根据选中的物流方式过滤原始数据
    if selected_logistics == "全部":
        df_filtered_by_logistics = df_selected.copy()
    else:
        df_filtered_by_logistics = df_selected[df_selected["物流方式"] == selected_logistics].copy()

    # 新增：容错处理 - 物流方式筛选后无数据
    if len(df_filtered_by_logistics) == 0:
        st.warning(f"所选物流方式「{selected_logistics}」暂无货代数据")
    else:
        # ---------------------- 【修改2】：数据源从df_selected改为df_filtered_by_logistics ----------------------
        # 筛选有效数据（基于物流方式筛选后的数据）
        df_freight_month_valid = df_filtered_by_logistics[
            (df_filtered_by_logistics[FREIGHT_MONTH_COLUMN_MAPPING["货代列名"]].notna()) &
            (df_filtered_by_logistics[FREIGHT_MONTH_COLUMN_MAPPING["到货年月列名"]].notna())
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
            # ---------------------- 【修改3】：移除重复的"筛选条件"标题 ----------------------
            # 原st.markdown("### 筛选条件")已上移，此处删除

            # 新增：日期解析容错函数（避免格式错误）
            def safe_parse_ym(ym):
                try:
                    return pd.to_datetime(str(ym) + "-01")
                except:
                    return pd.NaT


            # 生成中文月份列表（用于下拉框）
            freight_month_stats["年月排序"] = freight_month_stats["到货年月"].apply(safe_parse_ym)
            # 过滤无效日期
            freight_month_stats = freight_month_stats[freight_month_stats["年月排序"].notna()].copy()

            if len(freight_month_stats) == 0:
                st.warning("暂无有效货代月份数据可分析")
            else:
                freight_month_stats["中文月份"] = freight_month_stats["年月排序"].dt.strftime("%Y年%m月")

                # 提取唯一的中文月份（正序）
                unique_months = freight_month_stats.sort_values("年月排序")["中文月份"].unique().tolist()
                unique_ym = freight_month_stats.sort_values("年月排序")["到货年月"].unique().tolist()

                # 双下拉框选择开始/结束月份
                # ====================== 快捷筛选器 + 双下拉框 ======================
                st.markdown("#### 快捷筛选")

                # 基准月：数据最新月份 = 2026年2月（自动获取，不用手动改）
                latest_month = freight_month_stats["年月排序"].max()

                # 快捷筛选选项
                quick_options = [
                    "自定义时间 range",
                    "上个月",
                    "近三个月",
                    "近半年",
                    "近一年"
                ]
                selected_quick = st.selectbox("快捷筛选", options=quick_options, index=0)

                # 根据选项计算 开始月份 & 结束月份
                if selected_quick == "上个月":
                    start_month = latest_month - pd.DateOffset(months=1)
                    end_month = latest_month - pd.DateOffset(months=1)

                elif selected_quick == "近三个月":
                    start_month = latest_month - pd.DateOffset(months=2)
                    end_month = latest_month

                elif selected_quick == "近半年":
                    start_month = latest_month - pd.DateOffset(months=5)
                    end_month = latest_month

                elif selected_quick == "近一年":
                    start_month = latest_month - pd.DateOffset(months=11)
                    end_month = latest_month

                else:  # 自定义时间 range
                    start_month = None
                    end_month = None

                # 生成中文月份映射
                month_cn_list = freight_month_stats.sort_values("年月排序")["中文月份"].tolist()
                month_dt_list = freight_month_stats.sort_values("年月排序")["年月排序"].tolist()
                month_map = dict(zip(month_dt_list, month_cn_list))

                # 如果是快捷筛选 → 自动设置开始/结束月份
                if start_month is not None and end_month is not None:
                    start_month_cn = month_map.get(start_month, month_cn_list[0])
                    end_month_cn = month_map.get(end_month, month_cn_list[-1])
                else:
                    # 原有双下拉框
                    st.markdown("#### 自定义时间范围")
                    col_start, col_end = st.columns(2)
                    with col_start:
                        start_month_cn = st.selectbox("开始月份", options=unique_months, index=0,
                                                      key="freight_start_month")
                    with col_end:
                        end_month_cn = st.selectbox("结束月份", options=unique_months, index=len(unique_months) - 1,
                                                    key="freight_end_month")

                # ====================== 以下你原有代码完全不用动 ======================
                # 安全转换为原始年月格式（避免IndexError）
                start_ym = freight_month_stats[freight_month_stats["中文月份"] == start_month_cn]["到货年月"].iloc[0]
                end_ym = freight_month_stats[freight_month_stats["中文月份"] == end_month_cn]["到货年月"].iloc[0]

                # 安全转换为原始年月格式（避免IndexError）
                start_ym = freight_month_stats[freight_month_stats["中文月份"] == start_month_cn]["到货年月"].iloc[0]
                end_ym = freight_month_stats[freight_month_stats["中文月份"] == end_month_cn]["到货年月"].iloc[0]

                # 筛选时间范围内的数据
                df_freight_filtered = freight_month_stats[
                    (freight_month_stats["到货年月"] >= start_ym) &
                    (freight_month_stats["到货年月"] <= end_ym)
                    ].copy()

                # 按「到货年月降序 + 总订单数降序」排序
                df_freight_filtered["年月排序"] = df_freight_filtered["到货年月"].apply(safe_parse_ym)
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
                            display_cols.index(
                                "货代归类")] = f"background-color: {color}; color: white; font-weight: bold;"
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
                MIN_ORDERS = 5  # 最低订单量门槛
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
# ---------------------- 【修改1】：新增物流方式列校验 ----------------------
required_warehouse_cols = [
    WAREHOUSE_MONTH_COLUMN_MAPPING["仓库列名"],
    WAREHOUSE_MONTH_COLUMN_MAPPING["到货年月列名"],
    WAREHOUSE_MONTH_COLUMN_MAPPING["提前延期列名"],
    "FBA号",
    "物流方式"  # 新增：物流方式列
]
missing_warehouse_cols = [col for col in required_warehouse_cols if col not in df_selected_FBA.columns]
if missing_warehouse_cols:
    st.error(f"缺少仓库月度分析必要列：{missing_warehouse_cols}，请检查数据列名！")
else:
    # ---------------------- 【新增1】：物流方式筛选器（第一步筛选） ----------------------
    st.markdown("### 筛选条件")
    # 新增：物流方式筛选行（控制列宽）
    col_logistics, col_empty = st.columns([1, 3])
    with col_logistics:
        # 获取唯一的物流方式（去重+排序+去空）
        unique_logistics = sorted(df_selected_FBA["物流方式"].dropna().unique())
        # 新增"全部"选项，默认选中
        logistics_options = ["全部"] + unique_logistics
        selected_logistics = st.selectbox(
            "物流方式",
            options=logistics_options,
            index=0,
            key="warehouse_selected_logistics"  # 独立key，避免冲突
        )

    # 新增：根据选中的物流方式过滤原始数据
    if selected_logistics == "全部":
        df_filtered_by_logistics = df_selected_FBA.copy()
    else:
        df_filtered_by_logistics = df_selected_FBA[df_selected_FBA["物流方式"] == selected_logistics].copy()

    # 新增：容错处理 - 物流方式筛选后无数据
    if len(df_filtered_by_logistics) == 0:
        st.warning(f"所选物流方式「{selected_logistics}」暂无仓库数据")
    else:
        # ---------------------- 【修改2】：数据源从df_selected_FBA改为df_filtered_by_logistics ----------------------
        # 筛选有效数据（基于物流方式筛选后的数据）
        df_warehouse_month_valid = df_filtered_by_logistics[
            (df_filtered_by_logistics[WAREHOUSE_MONTH_COLUMN_MAPPING["仓库列名"]].notna()) &
            (df_filtered_by_logistics[WAREHOUSE_MONTH_COLUMN_MAPPING["到货年月列名"]].notna())
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
            # ---------------------- 【修改3】：移除重复的"筛选条件"标题 ----------------------
            # 原st.markdown("### 筛选条件")已上移，此处删除

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
                # ====================== 快捷筛选器 + 双下拉框 ======================
                st.markdown("#### 快捷筛选")

                # 基准月：数据最新月份
                latest_month = warehouse_month_stats["年月排序"].max()

                # 快捷筛选选项
                quick_options = [
                    "自定义时间",
                    "上个月",
                    "近三个月",
                    "近半年",
                    "近一年"
                ]

                # ====================== 修复：加唯一 key ======================
                selected_quick = st.selectbox("快捷筛选", options=quick_options, index=0, key="warehouse_quick_filter")

                # 根据选项计算 开始月份 & 结束月份
                if selected_quick == "上个月":
                    start_month = latest_month - pd.DateOffset(months=1)
                    end_month = latest_month - pd.DateOffset(months=1)

                elif selected_quick == "近三个月":
                    start_month = latest_month - pd.DateOffset(months=2)
                    end_month = latest_month

                elif selected_quick == "近半年":
                    start_month = latest_month - pd.DateOffset(months=5)
                    end_month = latest_month

                elif selected_quick == "近一年":
                    start_month = latest_month - pd.DateOffset(months=11)
                    end_month = latest_month

                else:  # 自定义时间 range
                    start_month = None
                    end_month = None

                # 生成中文月份映射
                month_cn_list = warehouse_month_stats.sort_values("年月排序")["中文月份"].tolist()
                month_dt_list = warehouse_month_stats.sort_values("年月排序")["年月排序"].tolist()
                month_map = dict(zip(month_dt_list, month_cn_list))

                # 如果是快捷筛选 → 自动设置开始/结束月份
                if start_month is not None and end_month is not None:
                    start_month_cn = month_map.get(start_month, month_cn_list[0])
                    end_month_cn = month_map.get(end_month, month_cn_list[-1])

                else:
                    # 自定义时间（原有逻辑不动，只保证不会和上面冲突）
                    st.markdown("#### 自定义时间范围")
                    col_start, col_end = st.columns(2)
                    with col_start:
                        start_month_cn = st.selectbox("开始月份", options=unique_months, index=0, key="warehouse_start")
                    with col_end:
                        end_month_cn = st.selectbox("结束月份", options=unique_months, index=len(unique_months) - 1,
                                                    key="warehouse_end")

                # ====================== 以下你原有代码完全不用动 ======================
                # 安全转换为原始年月格式（避免IndexError）
                start_ym = freight_month_stats[freight_month_stats["中文月份"] == start_month_cn]["到货年月"].iloc[0]
                end_ym = freight_month_stats[freight_month_stats["中文月份"] == end_month_cn]["到货年月"].iloc[0]

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

                # ---------------------- 新增：带搜索功能的仓库筛选 ----------------------
                # 1. 添加搜索输入框
                search_warehouse = st.text_input(
                    "搜索仓库",
                    placeholder="输入仓库名快速筛选（支持模糊匹配）",
                    key="search_warehouse"
                )

                # 2. 根据搜索关键词过滤仓库列表（模糊匹配）
                if search_warehouse.strip():
                    filtered_warehouses = [
                        warehouse for warehouse in unique_warehouses
                        if search_warehouse.strip() in warehouse
                    ]
                    # 无匹配结果时的提示
                    if len(filtered_warehouses) == 0:
                        st.warning(f"未找到包含「{search_warehouse}」的仓库")
                else:
                    filtered_warehouses = unique_warehouses

                # 3. 下拉框显示过滤后的仓库列表（默认选第一个）
                if len(filtered_warehouses) > 0:
                    selected_warehouse = st.selectbox(
                        "选择仓库",
                        options=filtered_warehouses,
                        index=0,
                        key="selected_warehouse"
                    )

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
                else:
                    # 无匹配仓库时不渲染图表
                    st.info("请输入有效仓库名或清空搜索框查看全部仓库")

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

                # 仓库卡片（一行3列 + 按优质→合格→异常排序）
                st.markdown("#### 各仓库详细表现")

                # 1. 计算总订单数（用于占比）
                total_orders = df_summary["累计订单数"].sum()

                # 2. 计算占比，并保留2位小数
                df_summary["订单占比(%)"] = (df_summary["累计订单数"] / total_orders * 100).round(2)

                # ==================== 新增指标（无报错版）====================
                # 平均月订单量
                df_summary["平均月订单量"] = (df_summary["累计订单数"] / df_summary["出现月份数"]).round(0).astype(int)


                # ==============================================
                # 🔥 平均上架时效（从 签收-完成上架 计算）
                # ==============================================
                warehouse_avg_delivery = df_warehouse_month_valid.groupby("仓库")["签收-完成上架"].agg(
                    平均上架时效="mean"
                ).reset_index()

                df_summary = pd.merge(
                    df_summary,
                    warehouse_avg_delivery,
                    on="仓库",
                    how="left"
                )
                df_summary["平均上架时效"] = df_summary["平均上架时效"].round(1).fillna(0)


                # ==============================================
                # 🔥 升级版：最近3个月趋势 + 每月平均时效
                # ==============================================
                # ==============================================
                # 🔥 终极修复：按仓库 + 按月 先汇总，再算趋势
                # 不会再出现重复月份！
                # ==============================================
                def analyze_3month_trend(warehouse_name):
                    # 1. 筛选当前仓库的【所有原始订单数据】
                    wh_data = df_warehouse_month_valid[df_warehouse_month_valid["仓库"] == warehouse_name].copy()

                    # 2. 【关键修复】按【到货年月】分组 → 每个月只算1条平均值（去重）
                    wh_monthly = wh_data.groupby("到货年月").agg(
                        平均时效=("签收-完成上架", "mean")  # 每月平均上架时效
                    ).round(1).reset_index()

                    # 3. 按时间排序，取最新3个“不同月份”
                    wh_monthly_sorted = wh_monthly.sort_values("到货年月", ascending=False).head(3)

                    # 4. 数据不足 → 显示单月
                    if len(wh_monthly_sorted) < 2:
                        month_val = wh_monthly_sorted["到货年月"].iloc[0]
                        day_val = wh_monthly_sorted["平均时效"].iloc[0]
                        return f"📊 单月数据（{month_val}: {day_val}天）"

                    # 5. 按时间正序（旧→新）
                    wh_monthly_sorted = wh_monthly_sorted.sort_values("到货年月", ascending=True)
                    month_list = wh_monthly_sorted["到货年月"].tolist()
                    day_list = wh_monthly_sorted["平均时效"].tolist()

                    # 组合成文字
                    month_str = " → ".join([f"{m}: {d}天" for m, d in zip(month_list, day_list)])

                    # 趋势判断
                    diff = day_list[-1] - day_list[-2]
                    if diff < -0.5:
                        return f"📈 时效变快（{month_str}）"
                    elif diff > 0.5:
                        return f"📉 时效变慢（{month_str}）"
                    else:
                        return f"📊 保持稳定（{month_str}）"


                df_summary["最近3个月趋势"] = df_summary["仓库"].apply(analyze_3month_trend)

                # ==============================================
                # 排序（不变）
                # ==============================================
                grade_order = {"优质": 0, "合格": 1, "异常": 2, "样本不足": 3}
                df_summary["排序标识"] = df_summary["综合评级"].map(grade_order)
                df_summary_sorted = df_summary.sort_values(
                    by=["排序标识", "累计订单数"],
                    ascending=[True, False]
                ).reset_index(drop=True)

                # ==============================================
                # 卡片渲染（最终升级版）
                # ==============================================
                from itertools import zip_longest

                warehouse_groups = list(zip_longest(*[iter(df_summary_sorted.to_dict('records'))] * 3))

                for group in warehouse_groups:
                    col1, col2, col3 = st.columns(3)
                    cols = [col1, col2, col3]
                    for idx, warehouse in enumerate(group):
                        if warehouse is None:
                            continue
                        with cols[idx]:
                            color = "#2e7d32" if warehouse["综合评级"] == "优质" else "#ff9800" if warehouse[
                                                                                                       "综合评级"] == "合格" else "#c62828" if \
                            warehouse["综合评级"] == "异常" else "#718096"
                            desc = "优秀稳定" if warehouse["综合评级"] == "优质" else "达标待优化" if warehouse[
                                                                                                          "综合评级"] == "合格" else "风险较高" if \
                            warehouse["综合评级"] == "异常" else "需持续观察"

                            st.markdown(f"""
                            <div style='border:1px solid #e2e8f0; border-radius:6px; padding:15px; margin:10px 0; border-left:4px solid {color};'>
                              <strong style='font-size:16px;'>{warehouse['仓库']}</strong>
                              <p style='color:{color}; margin:8px 0;'>{warehouse['综合评级']} | {desc}</p>
                              <p style='font-size:14px; margin:4px 0;'>📊 加权准时率：{warehouse['加权平均准时率']}%</p>
                              <p style='font-size:14px; margin:4px 0;'>📦 累计订单：{warehouse['累计订单数']}单 ({warehouse['订单占比(%)']}%)</p>
                              <p style='font-size:14px; margin:4px 0;'>📈 平均月单量：{warehouse['平均月订单量']}单</p>
                              <p style='font-size:14px; margin:4px 0;'>🚀 平均上架时效：{warehouse['平均上架时效']} 天</p>
                              <p style='font-size:14px; margin:4px 0; word-break: break-all;'>📅 最近3个月：{warehouse['最近3个月趋势']}</p>
                              <p style='font-size:14px; margin:4px 0;'>📅 出现月份：{warehouse['出现月份数']}个</p>
                              <p style='font-size:14px; margin:4px 0;'>🔍 最新表现：{warehouse['最新表现']}</p>
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

#物流成本分析区域

# ------------------------------------------------------
# 这一段 完全是你原来的代码，我一个字都没改！
# -----------------------------------------------------

st.title("📊 物流成本分析")

# 1. 加载成本数据
@st.cache_data(show_spinner="加载成本数据中...")
def load_cost_data():
    url = "https://raw.githubusercontent.com/Jane-zzz-123/Logistics/main/CAE.xlsx"
    df_cost = pd.read_excel(url, sheet_name="数据")

    need_cols = ["周期", "月份", "目的仓", "仓库", "区域", "实际物流方式", "货代", "货代渠道", "重量", "总费用"]
    df_cost = df_cost[[col for col in need_cols if col in df_cost.columns]]

    df_cost = df_cost.dropna(subset=["周期", "实际物流方式", "重量", "总费用"])
    df_cost = df_cost[(df_cost["重量"] > 0) & (df_cost["总费用"] >= 0)]

    df_cost["周期"] = pd.to_numeric(df_cost["周期"], errors="coerce").astype(int)
    df_cost["月份"] = pd.to_numeric(df_cost["月份"], errors="coerce").astype(int)
    df_cost = df_cost.sort_values("周期").reset_index(drop=True)
    return df_cost

df_cost = load_cost_data()

# ====================== 自定义颜色映射 ======================
color_map = {
    "空派": "#ff4b4b",        # 红色
    "空派": "#1f77b4",        # 蓝色
    "以星特快": "#2ca02c",    # 绿色
    "以星": "#ff7f0e",        # 橙色
    "正班": "#7f7f7f",        # 灰色
    "普船": "#ffdd00"         # 黄色
}

# 把不在映射里的物流方式默认用一个颜色
default_color = "#9467bd"

# ====================== 切换：按月份 / 按周期 ======================
view_mode = st.radio("筛选维度", ["按周期", "按月份"], horizontal=True)

# ====================== 筛选面板 ======================
with st.expander("🔎 筛选条件", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        if view_mode == "按周期":
            period_list = sorted(df_cost["周期"].unique())
            max_p = max(period_list)
            default_val = [p for p in period_list if p >= max_p - 3]
            selected = st.multiselect("周期", period_list, default=default_val)
        else:
            month_list = sorted(df_cost["月份"].dropna().unique())
            default_val = month_list[-3:] if len(month_list) >= 3 else month_list
            selected = st.multiselect("月份", month_list, default=default_val)

    with col2:
        area_list = ["全部"] + sorted(df_cost["区域"].dropna().unique())
        selected_area = st.selectbox("区域", area_list)

# ====================== 数据筛选 ======================
df = df_cost.copy()
if view_mode == "按周期":
    df = df[df["周期"].isin(selected)]
else:
    df = df[df["月份"].isin(selected)]

if selected_area != "全部":
    df = df[df["区域"] == selected_area]

if df.empty:
    st.warning("无数据")
    st.stop()

# ====================== 核心计算 ======================
group_col = "周期" if view_mode == "按周期" else "月份"

df_sum = df.groupby([group_col, "实际物流方式"], as_index=False).agg(
    总重量=("重量", "sum"),
    总费用=("总费用", "sum")
)
df_sum["折算单价"] = (df_sum["总费用"] / df_sum["总重量"]).round(4)
df_sum = df_sum.sort_values(["实际物流方式", group_col]).reset_index(drop=True)

# 环比
df_sum["上周单价"] = df_sum.groupby("实际物流方式")["折算单价"].shift(1)
df_sum["环比差值"] = (df_sum["折算单价"] - df_sum["上周单价"]).round(2)
df_sum["环比幅度"] = np.where(
    df_sum["上周单价"] > 0,
    (df_sum["环比差值"] / df_sum["上周单价"] * 100).round(2),
    0
)

all_logistics = sorted(df_cost["实际物流方式"].unique())
sorted_values = sorted(df_sum[group_col].unique())

# 最新周期/月份
if selected:
    latest = max(selected)
else:
    latest = sorted_values[-1]

latest_data = df_sum[df_sum[group_col] == latest].copy()

# ====================== 智能总结 ======================
st.subheader("📝 成本变化总结")
summary_html = ""
for logi in all_logistics:
    row = latest_data[latest_data["实际物流方式"] == logi]
    if row.empty:
        summary_html += f"<div>• <b>{logi}</b>：<span style='color:#888'>当前维度无数据</span></div>"
        continue

    price = row["折算单价"].iloc[0]
    diff = row["环比差值"].iloc[0]
    pct = row["环比幅度"].iloc[0]

    if pd.isna(diff):
        summary_html += f"<div>• <b>{logi}</b>：单价 ¥{price:.2f}（首期）</div>"
    elif diff > 0:
        summary_html += f"<div style='color:#ff4b4b'>• <b>{logi}</b>：↑ 上升 ¥{diff:.2f}（+{pct:.2f}%）单价 ¥{price:.2f}</div>"
    else:
        summary_html += f"<div style='color:#00b578'>• <b>{logi}</b>：↓ 下降 ¥{abs(diff):.2f}（{pct:.2f}%）单价 ¥{price:.2f}</div>"

st.markdown(summary_html, unsafe_allow_html=True)

# ====================== 折线图（显示折点数值+自定义颜色） ======================
st.subheader("📈 各物流方式单价趋势")
df_sum["x_str"] = df_sum[group_col].astype(str)

# 给不在映射里的物流方式补充颜色
for logi in all_logistics:
    if logi not in color_map:
        color_map[logi] = default_color

fig = px.line(
    df_sum,
    x="x_str",
    y="折算单价",
    color="实际物流方式",
    color_discrete_map=color_map,
    markers=True,
    category_orders={"x_str": [str(x) for x in sorted_values]}
)

# 关键：折点上直接显示数值（+物流方式可以加在hover里，也可以不加）
fig.update_traces(
    text=df_sum["折算单价"].round(2),  # 折点直接显示数值
    textposition="top center",
    hovertemplate=f"{group_col}：%{{x}}<br>物流方式：%{{fullData.name}}<br>单价：%{{y:.2f}}<extra></extra>"
)
fig.update_xaxes(type="category")
st.plotly_chart(fig, use_container_width=True)

# ====================== 统计表（单价强制2位小数） ======================
st.subheader("📋 折算单价统计表（带环比）")
data_map = {(str(r[group_col]), r["实际物流方式"]): r for _, r in df_sum.iterrows()}

table_html = f"<table style='width:100%;border-collapse:collapse;text-align:center;font-size:14px;'>"
table_html += f"<tr style='background:#f0f2f6;font-weight:bold'><td>{group_col}</td>"
for l in all_logistics:
    table_html += f"<td style='border:1px solid #ddd;padding:8px'>{l}</td>"
table_html += "</tr>"

for val in sorted_values:
    table_html += f"<tr><td style='border:1px solid #ddd;padding:8px'>{val}</td>"
    for logi in all_logistics:
        key = (str(val), logi)
        if key not in data_map:
            table_html += "<td style='border:1px solid #ddd'>-</td>"
            continue

        r = data_map[key]
        price = r["折算单价"]
        diff = r["环比差值"]
        pct = r["环比幅度"]

        if pd.isna(diff):
            txt = "首期"
            color = "#888"
        else:
            sign = "+" if diff > 0 else ""
            txt = f"{sign}{diff:.2f} ({sign}{pct:.2f}%)"
            color = "#ff4b4b" if diff > 0 else "#00b578"

        # 强制两位小数
        cell = f"<div>{price:.2f}</div><div style='font-size:12px;color:{color}'>{txt}</div>"
        table_html += f"<td style='border:1px solid #ddd;padding:8px'>{cell}</td>"
    table_html += "</tr>"
table_html += "</table>"

st.markdown(table_html, unsafe_allow_html=True)
st.caption("📌 红色=上升 | 绿色=下降 ")

# ===================== 数据源链接展示（直接打开/下载） =====================
st.subheader("📋 原始数据源（点击链接直接访问）")

# 你的Excel文件直链
data_source_url = "https://github.com/Jane-zzz-123/Logistics/raw/main/Logisticsdata.xlsx"

# 美化的链接展示（大字体、醒目颜色）
st.markdown(f"""
<div style='background-color: #f0f8fb; padding: 20px; border-radius: 10px; margin: 10px 0;'>
    <p style='font-size: 16px; color: #2d3748; margin: 0 0 10px;'>📌 数据源文件地址：</p>
    <a href='{data_source_url}' target='_blank' style='font-size: 18px; color: #4299e1; font-weight: bold; text-decoration: none;'>
        {data_source_url}
    </a>
    <p style='font-size: 14px; color: #718096; margin: 10px 0 0;'>
        💡 点击链接可直接打开/下载Excel文件 | 建议复制链接到浏览器打开
    </p>
</div>
""", unsafe_allow_html=True)

# 补充提示（方便看板人员操作）
st.caption("✅ 操作说明：")
st.caption("1. 点击链接 → 浏览器会直接打开Excel文件（部分浏览器）或自动下载")
st.caption("2. 若链接无法打开，复制链接到Chrome/Firefox浏览器地址栏访问")
st.caption("3. 文件格式：XLSX | 可直接用Excel/WPS打开校验数据")