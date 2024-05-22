# <editor-fold desc="导入模块语句">
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
# </editor-fold>


# <editor-fold desc="警告禁用">
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# </editor-fold>


# <editor-fold desc="基于高斯核的SVM支持向量机模型实现">
# <editor-fold desc="高斯核函数声明">
def gaussian_kernel(x1, x2, sigma=0.1):
    return np.exp(-np.sum((x1 - x2) ** 2) / (2 * (sigma ** 2)))
# </editor-fold>

# <editor-fold desc="SVM类声明(待修复)">
class SVM:
    def __init__(self, C=1):
        self.C = C
    
    def fit(self, X, y):
        m, n = X.shape
        self.X = X
        self.y = y
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        self.K = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                self.K[i, j] = gaussian_kernel(X[i], X[j])
        
        def objective(alpha):
            alpha = alpha.reshape(-1)
            return 0.5 * np.sum(alpha) - np.sum(alpha * y @ y[:, np.newaxis] * self.K) / 2
        
        constraints = [{'type': 'eq', 'fun': lambda alpha: np.sum(alpha * y)}]
        from scipy.optimize import minimize
        alpha0 = np.zeros(m)
        bounds = [(0, self.C) for _ in range(m)]
        result = minimize(objective, alpha0, bounds=bounds, constraints=constraints)
        if result.success:
            self.alpha = result.x
            support_vectors_idx = np.where(self.alpha > 1e-5)[0]
            if len(support_vectors_idx) > 0:
                self.support_vectors = X[support_vectors_idx]
                self.support_labels = y[support_vectors_idx]
                self.bias = np.mean(self.support_labels - np.sum(
                    self.alpha[support_vectors_idx] * self.support_labels.reshape(-1, 1) * self.K[
                        support_vectors_idx, support_vectors_idx], axis=0))
            else:
                print("No support vectors found. Using default bias value.")
                self.bias = 0
        else:
            print("Optimization failed!")

    def predict(self, X):
        y_pred = np.zeros(len(X))
        for i in range(len(X)):
            prediction = self.bias
            for j in range(len(self.support_vectors)):
                prediction += self.alpha[j] * self.support_labels[j] * gaussian_kernel(X[i],
                                                                                       self.support_vectors[j])
            y_pred[i] = 1 if prediction > 0 else -1
            return y_pred
# </editor-fold>

# <editor-fold desc="SVM库调用(临时)">
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
# </editor-fold>
# </editor-fold>


# <editor-fold desc="原始数据集读取">
"""
# <editor-fold desc="数据子集读取">
file_paths = [
    "/Users/wangmingyu/Downloads/2015-citibike-tripdata/1_January/201501-citibike-tripdata_1.csv",
    "/Users/wangmingyu/Downloads/2015-citibike-tripdata/2_February/201502-citibike-tripdata_1.csv",
    "/Users/wangmingyu/Downloads/2015-citibike-tripdata/3_March/201503-citibike-tripdata_1.csv",
    "/Users/wangmingyu/Downloads/2015-citibike-tripdata/4_April/201504-citibike-tripdata_1.csv",
    "/Users/wangmingyu/Downloads/2015-citibike-tripdata/5_May/201505-citibike-tripdata_1.csv",
    "/Users/wangmingyu/Downloads/2015-citibike-tripdata/6_June/201506-citibike-tripdata_1.csv",
    "/Users/wangmingyu/Downloads/2015-citibike-tripdata/7_July/201507-citibike-tripdata_1.csv",
    "/Users/wangmingyu/Downloads/2015-citibike-tripdata/7_July/201507-citibike-tripdata_2.csv",
    "/Users/wangmingyu/Downloads/2015-citibike-tripdata/8_August/201508-citibike-tripdata_1.csv",
    "/Users/wangmingyu/Downloads/2015-citibike-tripdata/8_August/201508-citibike-tripdata_2.csv",
    "/Users/wangmingyu/Downloads/2015-citibike-tripdata/9_September/201509-citibike-tripdata_1.csv",
    "/Users/wangmingyu/Downloads/2015-citibike-tripdata/9_September/201509-citibike-tripdata_2.csv",
    "/Users/wangmingyu/Downloads/2015-citibike-tripdata/10_October/201510-citibike-tripdata_1.csv",
    "/Users/wangmingyu/Downloads/2015-citibike-tripdata/10_October/201510-citibike-tripdata_2.csv",
    "/Users/wangmingyu/Downloads/2015-citibike-tripdata/11_November/201511-citibike-tripdata_1.csv",
    "/Users/wangmingyu/Downloads/2015-citibike-tripdata/12_December/201512-citibike-tripdata_1.csv"
]
# </editor-fold>

# <editor-fold desc="数据子集合并">
data_frames = []
for path in file_paths:
    df = pd.read_csv(path)
    data_frames.append(df)
citibikeTripdata2015 = pd.concat(data_frames, ignore_index=True)
print("数据集读取完成")
# </editor-fold>

# <editor-fold desc="距离计算函数">
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371
    distance = r * c
    return distance
# </editor-fold>

# <editor-fold desc="数据集预处理">
# <editor-fold desc="订阅者筛选">
subscriberCitibikeTripdata2015 = citibikeTripdata2015[citibikeTripdata2015['usertype'] == 'Subscriber'].copy()
# </editor-fold>

# <editor-fold desc="年龄计算">
subscriberCitibikeTripdata2015['age'] = 2015 - subscriberCitibikeTripdata2015['birth year']
# </editor-fold>

# <editor-fold desc="骑行开始时间及月份计算">
subscriberCitibikeTripdata2015['startmonth'] = subscriberCitibikeTripdata2015['starttime'].str.split('/').str[0]
subscriberCitibikeTripdata2015['starthour'] = \
    subscriberCitibikeTripdata2015['starttime'].str.split(' ').str[1].str.split(':').str[0]
# </editor-fold>

# <editor-fold desc="骑行距离计算">
subscriberCitibikeTripdata2015['tripdistance'] = haversine(subscriberCitibikeTripdata2015['start station latitude'],
                                                           subscriberCitibikeTripdata2015['start station longitude'],
                                                           subscriberCitibikeTripdata2015['end station latitude'],
                                                           subscriberCitibikeTripdata2015['end station longitude'])
# </editor-fold>

# <editor-fold desc="骑行速度计算">
subscriberCitibikeTripdata2015['tripvelocity'] = \
    subscriberCitibikeTripdata2015['tripdistance'] / subscriberCitibikeTripdata2015['tripduration'] * 3600
# </editor-fold>

# <editor-fold desc="性别映射">
subscriberCitibikeTripdata2015['gender'] = subscriberCitibikeTripdata2015['gender'].map({1: 1, 2: -1})
# </editor-fold>

# <editor-fold desc="无关属性删除">
subscriberCitibikeTripdata2015.drop(
    columns=['starttime', 'stoptime', 'start station name', 'end station name', 'start station latitude',
             'start station longitude', 'end station latitude', 'end station longitude', 'bikeid', 'usertype',
             'birth year'], inplace=True)
# </editor-fold>

# <editor-fold desc="新属性逻辑排序">
new_column_order = ['tripduration', 'tripdistance', 'tripvelocity', 'age', 'startmonth',
                    'starthour', 'start station id', 'end station id', 'gender']
subscriberCitibikeTripdata2015 = subscriberCitibikeTripdata2015.reindex(columns=new_column_order)
# </editor-fold>

# <editor-fold desc="缺失值删除">
subscriberCitibikeTripdata2015.dropna(inplace=True)
# </editor-fold>

# <editor-fold desc="明显非法数据删除">
# <editor-fold desc="删除tripduration大于86400的行">
subscriberCitibikeTripdata2015 = subscriberCitibikeTripdata2015[subscriberCitibikeTripdata2015['tripduration'] <= 86400]
# </editor-fold>

# <editor-fold desc="删除tripdistance大于50的行">
subscriberCitibikeTripdata2015 = subscriberCitibikeTripdata2015[subscriberCitibikeTripdata2015['tripdistance'] <= 50]
# </editor-fold>

# <editor-fold desc="删除 tripvelocity小于等于0或大于25的行">
subscriberCitibikeTripdata2015 = subscriberCitibikeTripdata2015[
    (subscriberCitibikeTripdata2015['tripvelocity'] > 0) & (subscriberCitibikeTripdata2015['tripvelocity'] <= 25)]
# </editor-fold>

# <editor-fold desc="删除age大于100的行">
subscriberCitibikeTripdata2015 = subscriberCitibikeTripdata2015[subscriberCitibikeTripdata2015['age'] <= 100]
# </editor-fold>

# <editor-fold desc="删除 startmonth不在1到12之间的行">
subscriberCitibikeTripdata2015 = subscriberCitibikeTripdata2015[
    subscriberCitibikeTripdata2015['startmonth'].astype(int).between(1, 12)]
# </editor-fold>

# <editor-fold desc="删除 starthour不在0到23之间的行">
subscriberCitibikeTripdata2015 = subscriberCitibikeTripdata2015[
    subscriberCitibikeTripdata2015['starthour'].astype(int).between(0, 23)]
# </editor-fold>

# <editor-fold desc="删除 gender不为1或-1的行">
subscriberCitibikeTripdata2015 = subscriberCitibikeTripdata2015[subscriberCitibikeTripdata2015['gender'].isin([1, -1])]
# </editor-fold>
# </editor-fold>

# <editor-fold desc="数据类型转换">
subscriberCitibikeTripdata2015['startmonth'] = subscriberCitibikeTripdata2015['startmonth'].astype(int)
subscriberCitibikeTripdata2015['starthour'] = subscriberCitibikeTripdata2015['starthour'].astype(int)
# </editor-fold>

# <editor-fold desc="数据舍入">
# <editor-fold desc="骑行距离舍入">
subscriberCitibikeTripdata2015['tripdistance'] = subscriberCitibikeTripdata2015['tripdistance'].round(2)
# </editor-fold>

# <editor-fold desc="骑行速度舍入">
subscriberCitibikeTripdata2015['tripvelocity'] = subscriberCitibikeTripdata2015['tripvelocity'].round(2)
# </editor-fold>
# </editor-fold>

# <editor-fold desc="提示语句">
print("数据集预处理完成")
# </editor-fold>
# </editor-fold>

# <editor-fold desc="数据集预处理后属性、信息及头部预览">
print("数据集预处理后属性预览：")
print(subscriberCitibikeTripdata2015.columns)
print("数据集预处理后信息预览：")
print(subscriberCitibikeTripdata2015.dtypes)
print("数据集预处理后头部预览：")
print(subscriberCitibikeTripdata2015.head)
# </editor-fold>

# <editor-fold desc="数据集保存">
subscriberCitibikeTripdata2015.to_csv(
    "/Users/wangmingyu/Downloads/2015-citibike-tripdata/Whole_Year/subscriberCitibikeTripdata2015.csv", index=False)
# </editor-fold>

# <editor-fold desc="基于z-score的数据集归一化">
# <editor-fold desc="计时器头">
start_time_ZS = time.time()  # 记录开始时间
# </editor-fold>

# <editor-fold desc="归一化">
means = subscriberCitibikeTripdata2015.drop(columns=['gender']).mean(axis=0)
stds = subscriberCitibikeTripdata2015.drop(columns=['gender']).std(axis=0)
data = (subscriberCitibikeTripdata2015.drop(columns=['gender']) - means) / stds
data = pd.concat([data, subscriberCitibikeTripdata2015['gender']], axis=1)
data = data.round(2)
data.to_csv("/Users/wangmingyu/Downloads/2015-citibike-tripdata/Whole_Year_ZScore/data.csv", index=False)
print("数据集归一化完成")
# </editor-fold>

# <editor-fold desc="计时器尾">
end_time_ZS = time.time()  # 记录结束时间
elapsed_time_ZS = end_time_ZS - start_time_ZS  # 计算用时
print("归一化用时：", elapsed_time_ZS)
# </editor-fold>
# </editor-fold>

# <editor-fold desc="数据集归一化后头部预览">
print("数据集归一化后头部预览：")
print(data.head)
# </editor-fold>

# <editor-fold desc="用于拟合特征选取的小规模数据集随机抽样(随机抽取1000条，男女比例1:1)">
sampled_male_data = data[data['gender'] == 1].sample(n=500, random_state=42)
sampled_female_data = data[data['gender'] == -1].sample(n=500, random_state=42)
sampled_data_balanced = pd.concat([sampled_male_data, sampled_female_data], ignore_index=True)
final_sampled_data = sampled_data_balanced.sample(n=1000, random_state=42)
final_sampled_data.to_csv("/Users/wangmingyu/Downloads/2015-citibike-tripdata/Sampled_Data/sampledData.csv",
                          index=False)
print("处理后的小规模数据集保存完成")
# </editor-fold>
"""
# </editor-fold>


# <editor-fold desc="预处理后完整数据集及小规模数据集读取">
"""
# <editor-fold desc="预处理后完整数据集(未归一化)读取">
un_normalised_data = pd.read_csv(
    "/Users/wangmingyu/Downloads/2015-citibike-tripdata/Whole_Year/subscriberCitibikeTripdata2015.csv")
print("预处理后完整数据集(未归一化)读取完成")
# </editor-fold>
"""

""""""
# <editor-fold desc="预处理后完整数据集(已归一化)读取">
data = pd.read_csv(
    "/Users/wangmingyu/Downloads/2015-citibike-tripdata/Whole_Year_ZScore/data.csv")
print("预处理后完整数据集(已归一化)读取完成")
print("预处理后完整数据集(已归一化)属性列预览")
print(data.columns)
print("预处理后完整数据集(已归一化)数据类型预览")
print(data.dtypes)
# </editor-fold>
""""""

"""
# <editor-fold desc="预处理后小规模数据集读取">
sampled_data = pd.read_csv(
    "/Users/wangmingyu/Downloads/2015-citibike-tripdata/Sampled_Data/sampledData.csv")
print("预处理后小规模数据集读取完成")
print("预处理后小规模数据集头部预览")
print(sampled_data.head)
# </editor-fold>
"""
# </editor-fold>


# <editor-fold desc="数据可视化探索">
"""
# <editor-fold desc="箱线图">
# <editor-fold desc="参数预设">
# <editor-fold desc="画布预设">
plt.figure(figsize=(15, 7))
# </editor-fold>

# <editor-fold desc="颜色预设">
gender_colors = {1: 'blue', -1: 'red'}
# </editor-fold>
# </editor-fold>

# <editor-fold desc="子图描述">
# <editor-fold desc="性别-年龄">
plt.subplot(1, 2, 1)
sns.boxplot(x='gender', y='age', data=un_normalised_data, palette=gender_colors)
plt.title('Age Distribution by Gender')
plt.xlabel('Gender')
plt.ylabel('Age')
# </editor-fold>

# <editor-fold desc="性别-速度">
plt.subplot(1, 2, 2)
sns.boxplot(x='gender', y='tripvelocity', data=un_normalised_data, palette=gender_colors)
plt.title('Trip Velocity Distribution by Gender')
plt.xlabel('Gender')
plt.ylabel('Trip Velocity')
# </editor-fold>
# </editor-fold>

# <editor-fold desc="母图绘制">
plt.show()
# </editor-fold>
# </editor-fold>

# <editor-fold desc="直方图">
# <editor-fold desc="参数预设">
# <editor-fold desc="画布预设">
plt.figure(figsize=(15, 6))
# </editor-fold>

# <editor-fold desc="颜色预设">
gender_colors = {1: 'blue', -1: 'red'}
# </editor-fold>
# </editor-fold>

# <editor-fold desc="子图描述">
# <editor-fold desc="年龄">
plt.subplot(1, 3, 1)
sns.histplot(data=un_normalised_data, x='age', hue='gender', kde=True, palette=gender_colors)
plt.title('Age Distribution by Gender')
plt.xlabel('Age')
plt.ylabel('Frequency')
# </editor-fold>

# <editor-fold desc="骑行距离">
plt.subplot(1, 3, 2)
sns.histplot(data=un_normalised_data, x='tripdistance', hue='gender', kde=True, palette=gender_colors)
plt.title('Trip Distance Distribution by Gender')
plt.xlabel('Trip Distance')
plt.ylabel('Frequency')
# </editor-fold>

# <editor-fold desc="骑行速度">
plt.subplot(1, 3, 3)
sns.histplot(data=un_normalised_data, x='tripvelocity', hue='gender', kde=True, palette=gender_colors)
plt.title('Trip Velocity Distribution by Gender')
plt.xlabel('Trip Velocity')
plt.ylabel('Frequency')
# </editor-fold>
# </editor-fold>

# <editor-fold desc="母图绘制">
plt.show()
# </editor-fold>
# </editor-fold>

# <editor-fold desc="密度图">
# <editor-fold desc="参数预设">
# <editor-fold desc="画布预设">
plt.figure(figsize=(15, 6))
# </editor-fold>

# <editor-fold desc="颜色预设">
gender_colors = {1: 'blue', -1: 'red'}
# </editor-fold>
# </editor-fold>

# <editor-fold desc="子图描述">
# <editor-fold desc="年龄">
plt.subplot(1, 3, 1)
sns.kdeplot(data=un_normalised_data, x='age', hue='gender', fill=True, palette=gender_colors)
plt.title('Density Plot of Age by Gender')
plt.xlabel('Age')
plt.ylabel('Density')
# </editor-fold>

# <editor-fold desc="骑行距离">
plt.subplot(1, 3, 2)
sns.kdeplot(data=un_normalised_data, x='tripdistance', hue='gender', fill=True, palette=gender_colors)
plt.title('Density Plot of Trip Distance by Gender')
plt.xlabel('Trip Distance')
plt.ylabel('Density')
# </editor-fold>

# <editor-fold desc="骑行速度">
plt.subplot(1, 3, 3)
sns.kdeplot(data=un_normalised_data, x='tripvelocity', hue='gender', fill=True, palette=gender_colors)
plt.title('Density Plot of Trip Velocity by Gender')
plt.xlabel('Trip Velocity')
plt.ylabel('Density')
# </editor-fold>
# </editor-fold>

# <editor-fold desc="母图绘制">
plt.show()
# </editor-fold>
# </editor-fold>

# <editor-fold desc="饼图">
# <editor-fold desc="性别">
gender_counts = un_normalised_data['gender'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(gender_counts, labels=['Male', 'Female'], autopct='%1.1f%%', startangle=140, colors=['navy', 'pink'])
plt.title('Gender Distribution in Dataset')
plt.show()
# </editor-fold>
# </editor-fold>

# <editor-fold desc="热力图">
# <editor-fold desc="参数预设">
# <editor-fold desc="画布预设">
plt.figure(figsize=(15, 6))
# </editor-fold>

# <editor-fold desc="颜色预设">
gender_colors = {1: 'blue', -1: 'red'}
# </editor-fold>
# </editor-fold>

# <editor-fold desc="子图描述">
# <editor-fold desc="全部特征">
plt.subplot(1, 2, 1)
correlation_matrix = un_normalised_data.corr()
high_correlation_indices = (correlation_matrix == 1).values
correlation_matrix[high_correlation_indices] = np.nan
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", mask=high_correlation_indices)
plt.title('Correlation Heatmap of Features by Gender')
plt.xlabel('Features')
plt.ylabel('Features')
# </editor-fold>

# <editor-fold desc="性别">
plt.subplot(1, 2, 2)
gender_correlation = un_normalised_data.corr()['gender'].drop('gender')
gender_correlation_sorted = gender_correlation.sort_values(ascending=False)
sns.heatmap(gender_correlation_sorted.to_frame(), annot=True, cmap='coolwarm', fmt=".2f", cbar=False)
plt.title('Correlation Heatmap of Gender with Other Features')
plt.xlabel('Features')
plt.ylabel('Gender')
# </editor-fold>
# </editor-fold>

# <editor-fold desc="母图绘制">
plt.show()
# </editor-fold>
# </editor-fold>

# <editor-fold desc="雷达图">
male_data = data[data['gender'] == 1]
female_data = data[data['gender'] == -1]
male_means = male_data[['tripduration', 'tripdistance', 'tripvelocity', 'age']].mean()
female_means = female_data[['tripduration', 'tripdistance', 'tripvelocity', 'age']].mean()
attributes = ['tripduration', 'tripdistance', 'tripvelocity', 'age']
male_means_list = male_means.tolist()
female_means_list = female_means.tolist()
num_vars = len(attributes)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.plot(angles, male_means_list, color='blue', linewidth=2, linestyle='solid', label='Male')
ax.fill(angles, male_means_list, color='blue', alpha=0.25)
ax.plot(angles, female_means_list, color='red', linewidth=2, linestyle='solid', label='Female')
ax.fill(angles, female_means_list, color='red', alpha=0.25)
ax.set_yticklabels([])
ax.set_xticks(angles)
ax.set_xticklabels(attributes)
plt.legend(loc='upper right')
plt.title('Gender Comparison Radar Chart')
plt.show()
# </editor-fold>
"""
# </editor-fold>


# <editor-fold desc="SVM模型特征选取">
"""
# <editor-fold desc="提示语句">
print("小规模数据集测试开始")
# </editor-fold>

# <editor-fold desc="定义待测试特征组合和目标列">
features_list = [['tripdistance', 'age', 'startmonth'],
                 ['tripvelocity', 'age', 'tripdistance'],
                 ['tripvelocity', 'tripdistance', 'startmonth', 'age'],
                 ['tripvelocity', 'tripdistance', 'startmonth', 'age', 'tripduration'],
                 ['tripvelocity', 'tripdistance', 'startmonth', 'age', 'tripduration', 'start station id']]
target = 'gender'
# </editor-fold>

# <editor-fold desc="定义SVM模型参数网络">
param_grid = {
    'gamma': [0.01, 0.1, 1, 10, 100],
    'C': [0.1, 1, 10, 100, 1000]
}
# </editor-fold>

# <editor-fold desc="单次测试评估指标记录">
accuracies = []
elapsed_times = []
# </editor-fold>

# <editor-fold desc="遍历特征组合进行测试">
for i, features in enumerate(features_list):
    print(f"测试{i + 1}(features: {features})")
    
    # <editor-fold desc="计时器头">
    start_time = time.time()
    # </editor-fold>
    
    # <editor-fold desc="数据集划分">
    data_size = len(sampled_data)
    train_size = int(0.85 * data_size)
    shuffled_data = sampled_data.sample(frac=1, random_state=42)
    X_train = shuffled_data[features].iloc[:train_size]
    y_train = shuffled_data[target].iloc[:train_size]
    X_test = shuffled_data[features].iloc[train_size:]
    y_test = shuffled_data[target].iloc[train_size:]
    # </editor-fold>
    
    # <editor-fold desc="SVM模型构建和网格搜索调参">
    grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    print(f"最佳参数: {best_params}")
    svm_model = SVC(kernel='rbf', gamma=best_params['gamma'], C=best_params['C'])
    # </editor-fold>
    
    # <editor-fold desc="SVM模型训练">
    svm_model.fit(X_train, y_train)
    # </editor-fold>
    
    # <editor-fold desc="SVM模型预测">
    y_pred = svm_model.predict(X_test)
    # </editor-fold>
    
    # <editor-fold desc="SVM模型拟合效果评估">
    # <editor-fold desc="准确度输出">
    accuracy = np.mean(y_pred == y_test)
    print(f"测试{i + 1}准确度：", accuracy)
    # </editor-fold>

    # <editor-fold desc="测试用时输出">
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"测试{i + 1}用时：", elapsed_time)
    # </editor-fold>

    # <editor-fold desc="准确度和测试用时记录">
    accuracies.append(accuracy)
    elapsed_times.append(elapsed_time)
    # </editor-fold>

    # <editor-fold desc="提示语句">
    print(f"测试{i + 1}完成\n")
    # </editor-fold>
    # </editor-fold>
# </editor-fold>

# <editor-fold desc="多次测试的拟合效果对比">
max_accuracy = max(accuracies)
max_accuracy_indices = [i for i, acc in enumerate(accuracies) if acc == max_accuracy]
best_index = None
min_elapsed_time = float('inf')
for i in max_accuracy_indices:
    if elapsed_times[i] < min_elapsed_time:
        min_elapsed_time = elapsed_times[i]
        best_index = i
if best_index is not None:
    print(f"测试{best_index + 1}的拟合效果最好，应选取该次测试中的特征组合。")
else:
    print("无法确定最佳模型。")
# </editor-fold>

# <editor-fold desc="提示语句">
print("小规模数据集测试结束")
# </editor-fold>
"""
# </editor-fold>


# <editor-fold desc="基于最优特征组合的完整数据集模型训练、测试及评估（迭代次数'max_iter'可调参）">
""""""
# <editor-fold desc="提示语句">
print("基于最优特征组合的完整数据集模型训练开始")
# </editor-fold>

# <editor-fold desc="计时器头">
start_time = time.time()
# </editor-fold>

# <editor-fold desc="SVM模型训练（迭代次数'max_iter'可调参）">
# <editor-fold desc="最优特征组合和目标列选取">
best_features = ['tripvelocity', 'tripdistance', 'startmonth', 'age']
target = 'gender'
# </editor-fold>

# <editor-fold desc="数据集划分">
data_size = len(data)
train_size = int(0.85 * data_size)
shuffled_data = data.sample(frac=1, random_state=42)
X_train = shuffled_data[best_features][:train_size]
y_train = shuffled_data[target][:train_size]
X_test = shuffled_data[best_features][train_size:]
y_test = shuffled_data[target][train_size:]
# </editor-fold>

# <editor-fold desc="SVM模型构建（迭代次数'max_iter'可调参，迭代一次耗时约0.25秒）">
svm_model_final = SVC(kernel='rbf', gamma=0.001, C=1000, max_iter=2000)
# </editor-fold>

# <editor-fold desc="SVM模型训练">
svm_model_final.fit(X_train, y_train)
print("SVM模型训练完成")
# </editor-fold>
# </editor-fold>

# <editor-fold desc="SVM模型预测">
y_pred_final = svm_model_final.predict(X_test)
print("SVM模型预测完成")
# </editor-fold>

# <editor-fold desc="SVM模型评估">
accuracy_final = np.mean(y_pred_final == y_test)
print("最优特征组合的准确度：", accuracy_final)
print("SVM模型评估完成")
# </editor-fold>

# <editor-fold desc="计时器尾">
end_time = time.time()
elapsed_time = end_time - start_time
print("SVM模型用时：", elapsed_time)
# </editor-fold>

# <editor-fold desc="提示语句">
print("实验结束")
# </editor-fold>
""""""
# </editor-fold>


# <editor-fold desc="日志">
# <editor-fold desc="2024.4.25 20:30:00">
"""
文件说明：
1)main.py：SVM-citibikeTripdata实验的完整代码。
2)data.csv：经过数据合并、预处理、归一化的完整数据集。
3)sampledData.csv：经过数据合并、预处理、归一化，并从原始大小数据集中随机抽样所得测试用小规模数据集。

测试说明：
1)将“原始数据集读取”部分保留在注释中。
2)将“预处理后完整数据集及小规模数据集读取”中“预处理后完整数据集读取”部分打上注释。
3)将“预处理后完整数据集及小规模数据集读取”中“预处理后小规模数据集读取”部分中的读取路径替换为本代码包中/citibikeTripdataPackage/sampledData.csv在本地的绝对路径。
4)将“基于最优特征的完整数据集模型训练、测试及评估”暂时保持在注释中，直到修正了“SVM特征选取”。
5)修正“SVM特征选取”后，将“预处理后完整数据集及小规模数据集读取”中“预处理后完整数据集读取”部分移出注释，将该部分中的读取路径替换为本代码包中/citibikeTripdataPackage/data.csv在本地的绝对路径；将“基于最优特征的完整数据集模型训练、测试及评估”移出注释，将其中“SVM模型训练-最优特征组合选取”中的features_2改为“SVM特征选取”多次测试中表现相对最好的一组特征组合后运行。

目前仍未解决的问题：
1)SVM类的实现：虽然写了SVM类，但调用时报错，目前仍依赖sklearn.svm库；若修正了SVM类，则将“SVM特征选取”、“基于最优特征的完整数据集模型训练、测试及评估”中调用的SVC()函数修改为SVM()函数。
2)未知原因的模型准确度相同：“SVM特征选取”多次测试虽选取了不同的特征组合，但经过测试得到的模型准确度完全一致，原因不明。
3)“SVM特征选取”参数预设：尚不确定用于测试训练模型的特征组合和用于SVM构建的高斯核的gamma值的选取策略。
"""
# </editor-fold


# <editor-fold desc="2024.5.22 15:30:00">
"""
优化了模型拟合度极差的问题，将模型准确度由0.5提升至0.67，具体过程如下：
1)采取网格搜索的策略对高斯核SVM模型学习中参数gamma和C的选值进行了5*5的交叉验证，选取了最优参数组合(gamma=0.01, C=1000)。
2)将用于测试模型拟合效果的小规模数据集的性别比例由3:1调整至1:1。
3)对更多潜在的特征组合进行测试，最终确定选择特征组合['tripvelocity', 'tripdistance', 'startmonth', 'age']。
"""
# </editor-fold
# </editor-fold


# <editor-fold desc="电子签名">
print("\n----------\nBY FR13NDS")
# ....................BY  FR13NDS....................
# ...................................................
#                        _oo0oo_
#                       o8888888o
#                       88" . "88
#                       (| -_- |)
#                       0\  =  /0
#                     ___/`---'\___
#                   .' \\|     |// '.
#                  / \\|||  :  |||// \
#                 / _||||| -卍-|||||- \
#                |   | \\\  -  /// |   |
#                | \_|  ''\---/''  |_/ |
#                \ .-\___  '-'  ___/-. /
#              ___'. .'  /--.--\  `. .'___
#           ."" '<  `.___\_<|>_/___.' >' "".
#          | | :  `- \`.;`\ _ /`;.`/ - ` : | |
#          \  \ `_.   \_ __\ /__ _/   .-` /  /
#      =====`-.____`.___ \_____/___.-`___.-'=====
#                        `=---='
# ...................电子佛驱散疑难BUG......................
# </editor-fold
