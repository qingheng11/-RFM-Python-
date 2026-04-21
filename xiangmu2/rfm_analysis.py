import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置中文显示（避免图表乱码）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ========== 1. 读取数据 ==========
print("正在读取数据...")
df = pd.read_excel('Online Retail.xlsx')  # 注意文件名要一致
print(f"原始数据：{df.shape[0]} 行，{df.shape[1]} 列")

# ========== 2. 数据清洗 ==========
print("\n开始数据清洗...")

# 删除客户ID为空的
df_clean = df.dropna(subset=['CustomerID'])
print(f"删除空CustomerID后：{len(df_clean)} 行")

# 只保留正常交易（剔除退货）
df_clean = df_clean[df_clean['Quantity'] > 0]
df_clean = df_clean[df_clean['UnitPrice'] > 0]
print(f"剔除退货后：{len(df_clean)} 行")

# 计算每笔订单总金额
df_clean['TotalPrice'] = df_clean['Quantity'] * df_clean['UnitPrice']

# ========== 3. 计算RFM ==========
print("\n开始计算RFM...")

# 确定分析基准日（数据中最后一天 + 1天）
snapshot_date = df_clean['InvoiceDate'].max() + pd.Timedelta(days=1)

rfm = df_clean.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,  # Recency
    'InvoiceNo': 'nunique',  # Frequency
    'TotalPrice': 'sum'  # Monetary
}).reset_index()

rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
print(f"共 {len(rfm)} 个客户")

# ========== 4. RFM打分 ==========
print("\n开始RFM打分...")

# Recency: 天数越少分越高
rfm['R_Score'] = pd.qcut(rfm['Recency'], 4, labels=['4', '3', '2', '1'])

# Frequency: 次数越多分越高
rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=['1', '2', '3', '4'])

# Monetary: 金额越高分越高
rfm['M_Score'] = pd.qcut(rfm['Monetary'], 4, labels=['1', '2', '3', '4'])

# 合并总分
rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

# ========== 5. 用户分层 ==========
print("\n开始用户分层...")

def segment_customer(row):
    r, f, m = int(row['R_Score']), int(row['F_Score']), int(row['M_Score'])
    if r >= 3 and f >= 3 and m >= 3:
        return '高价值用户'
    elif r >= 3 and f >= 2:
        return '发展中用户'
    elif r <= 2 and f <= 2:
        return '流失风险用户'
    else:
        return '一般用户'

rfm['Segment'] = rfm.apply(segment_customer, axis=1)

# ========== 6. 统计结果 ==========
print("\n======= 用户分层统计 =======")

segment_stats = rfm.groupby('Segment').agg({
    'CustomerID': 'count',
    'Monetary': 'sum'
}).rename(columns={'CustomerID': '用户数', 'Monetary': '总GMV'})

segment_stats['用户占比'] = segment_stats['用户数'] / segment_stats['用户数'].sum() * 100
segment_stats['GMV占比'] = segment_stats['总GMV'] / segment_stats['总GMV'].sum() * 100

print(segment_stats.round(1))

# ========== 7. 保存结果 ==========
rfm.to_csv('rfm_result.csv', index=False, encoding='utf-8-sig')
print("\n结果已保存到 rfm_result.csv")

# ========== 8. 简单可视化 ==========
print("\n生成可视化图表...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 用户占比饼图
ax1.pie(segment_stats['用户占比'], labels=segment_stats.index, autopct='%1.1f%%')
ax1.set_title('用户分层占比')

# GMV贡献柱状图
ax2.bar(segment_stats.index, segment_stats['GMV占比'])
ax2.set_title('各层级GMV贡献占比')
ax2.set_ylabel('GMV占比 (%)')
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('rfm_charts.png', dpi=150)
plt.show()

print("\n完成！图表已保存为 rfm_charts.png")