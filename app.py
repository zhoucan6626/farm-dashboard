import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 云端环境配置，解决网页中文乱码问题
st.set_page_config(page_title="农田作物长势评估云平台", layout="wide")
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False
np.random.seed(42)

# 读取数据文件，文件名必须和你上传的csv名字一模一样
df = pd.read_csv("清洗后数据.csv", encoding="utf-8-sig")

# 数据导出功能
@st.cache_data
def convert_df(data):
    return data.to_csv(index=False).encode('utf-8-sig')

# ==========页面顶部数据总览==========
st.title("农田作物长势评估与水肥优化云平台")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("地块总数量", value=len(df))
with col2:
    avg_yield = round(df["亩产量"].mean(),1)
    st.metric("区域平均亩产量(kg)", value=avg_yield)
with col3:
    good_num = len(df[df["长势等级"]=="优良"])
    st.metric("优良长势地块数量", value=good_num)

st.divider()

# ==========地块查询与筛选模块==========
st.subheader("地块数据查询模块")
search_id = st.text_input("🔍 输入地块编号精准查询：")
df_show = df.copy()
if search_id:
    df_show = df_show[df_show["地块编号"].astype(str).str.contains(search_id)]

col_a, col_b = st.columns(2)
with col_a:
    crop_list = ["全部"] + list(df["作物类型"].unique())
    crop_type = st.selectbox("选择作物类型", options=crop_list)
with col_b:
    level_list = ["全部"] + list(df["长势等级"].unique())
    growth_level = st.selectbox("选择长势等级", options=level_list)

if crop_type != "全部":
    df_show = df_show[df_show["作物类型"]==crop_type]
if growth_level != "全部":
    df_show = df_show[df_show["长势等级"]==growth_level]

st.dataframe(df_show, use_container_width=True)
csv_data = convert_df(df_show)
st.download_button(
    label="📥 导出当前筛选地块数据",
    data=csv_data,
    file_name="地块筛选结果数据.csv",
    mime="text/csv"
)

st.divider()

# ==========统计图表展示==========
st.subheader("农田统计分析图表")
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14,6))
level_count = df["长势等级"].value_counts()
ax1.pie(level_count.values, labels=level_count.index, autopct='%1.1f%%')
ax1.set_title("地块长势等级分布占比")

crop_yield = df.groupby("作物类型")["亩产量"].mean().sort_values(ascending=False)
ax2.bar(crop_yield.index, crop_yield.values)
ax2.set_title("不同作物平均亩产量对比")
ax2.set_ylabel("亩产量(kg)")
plt.tight_layout()
st.pyplot(fig)

st.divider()

# ==========水肥参数预测产量==========
st.subheader("水肥投入参数交互式产量预测")
features = ["氮肥施用量(kg/亩)","磷肥施用量(kg/亩)","钾肥施用量(kg/亩)","灌溉量(m³/亩)","株高(cm)"]
train_df = df.dropna(subset=features+["亩产量"])
X = train_df[features]
y = train_df["亩产量"]
model = LinearRegression()
model.fit(X,y)

col1,col2,col3,col4,col5 = st.columns(5)
with col1:
    n = st.number_input("氮肥施用量(kg/亩)", value=18)
with col2:
    p = st.number_input("磷肥施用量(kg/亩)", value=10)
with col3:
    k = st.number_input("钾肥施用量(kg/亩)", value=12)
with col4:
    irr = st.number_input("灌溉量(m³/亩)", value=300)
with col5:
    h = st.number_input("株高(cm)", value=75)

if st.button("点击预测亩产量"):
    input_data = np.array([[n,p,k,irr,h]])
    pred_yield = model.predict(input_data)[0]
    st.success(f"模型预测地块亩产量：{round(pred_yield,1)} kg")

#侧边栏智能水肥方案推荐
st.sidebar.header("🌱 系统智能水肥方案推荐")
st.sidebar.markdown("""
### 优良长势地块（维持方案）
氮肥：18 kg/亩
磷肥：10 kg/亩
钾肥：12 kg/亩
灌溉量：300 m³/亩

### 中等长势地块（提质方案）
氮肥：24 kg/亩
磷肥：16 kg/亩
钾肥：18 kg/亩
灌溉量：285 m³/亩

### 较差长势地块（改良方案）
氮肥：16 kg/亩
磷肥：22 kg/亩
钾肥：20 kg/亩
灌溉量：270 m³/亩
""")