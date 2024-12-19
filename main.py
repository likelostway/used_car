import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model

# 读取数据集
train_data = pd.read_csv('Train.csv')
test_data = pd.read_csv('Test.csv')

# 清理列名，移除空格和特殊字符
def clean_column_names(df):
    df.columns = [col.strip().replace(' ', '_').replace('-', '_') for col in df.columns]
    return df

train_data = clean_column_names(train_data)
test_data = clean_column_names(test_data)

# 打印列名以检查是否正确
print("Train data columns:", train_data.columns)
print("Test data columns:", test_data.columns)

# 确保测试集的列与训练集一致
missing_columns = [col for col in train_data.columns if col not in test_data.columns]

if missing_columns:
    print("Missing columns in Test data:", missing_columns)
    test_data = test_data.reindex(columns=train_data.columns, fill_value=0)

# 数据预处理
# 将分类变量进行编码
label_encoders = {}
for column in train_data.select_dtypes(include=['object']).columns:
    if column in test_data.columns:
        le = LabelEncoder()
        train_data[column] = le.fit_transform(train_data[column])
        test_data[column] = test_data[column].apply(lambda x: le.transform([x])[0] if x in le.classes_ else np.nan)
        label_encoders[column] = le

# 查找可能的'price'列名
possible_price_columns = [col for col in train_data.columns if 'price' in col.lower()]
if possible_price_columns:
    price_column = possible_price_columns[0]
    print("Found 'price' column:", price_column)
else:
    raise ValueError("'price' column not found in train_data. Check the column names after cleaning.")

# 查找可能的'regDate'列名
possible_reg_date_columns = [col for col in train_data.columns if 'reg' in col.lower() and 'date' in col.lower()]
if possible_reg_date_columns:
    reg_date_column = possible_reg_date_columns[0]
    print("Found 'regDate' column:", reg_date_column)
else:
    raise ValueError("'regDate' column not found in train_data. Check the column names after cleaning.")

# 查找测试集可能的'regDate'列名
possible_reg_date_columns_t = [col for col in test_data.columns if 'reg' in col.lower() and 'date' in col.lower()]
if possible_reg_date_columns_t:
    reg_date_column_t = possible_reg_date_columns_t[0]
    print("Found 'regDate' column:", reg_date_column_t)
else:
    raise ValueError("'regDate' column not found in test_data. Check the column names after cleaning.")

# 查找可能的'SaleID'列名
possible_sale_id_columns = [col for col in train_data.columns if 'saleid' in col.lower()]
if len(possible_sale_id_columns) == 1:
    sale_id_column = possible_sale_id_columns[0]
    print("Found 'SaleID' column:", sale_id_column)
else:
    raise ValueError(
        f"Expected one 'SaleID' column, but found {len(possible_sale_id_columns)} columns with 'saleid' in the name.")

# 创建特征和标签数据
X = train_data.drop([sale_id_column, reg_date_column, 'price'], axis=1)
y = train_data['price']

X_test = test_data.drop([sale_id_column, reg_date_column_t, 'price'], axis=1)

# 删除训练集和测试集中以 'v' 开头的特征
v_features = [col for col in X.columns if col.startswith('v')]
X = X.drop(v_features, axis=1)
X_test = X_test.drop(v_features, axis=1)

# 使用 LabelEncoder 进行特征编码
for column in X.select_dtypes(include=['object']).columns:
    if column in X_test.columns:
        le = label_encoders[column]
        X[column] = le.transform(X[column])
        X_test[column] = le.transform(X_test[column])

# 划分训练集和测试集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征缩放
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# 创建模型
model = Sequential()
model.add(Input(shape=(X_train.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))  # 保持输出为线性

# 编译模型
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# 训练模型
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val), verbose=1)

# 保存模型和编码器
model.save('model.h5')
for column, le in label_encoders.items():
    le.save(f'{column}_encoder.pkl')

# 绘制训练和验证损失
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# 使用模型进行预测
predictions = model.predict(X_test)

# 创建包含 'SaleID' 和预测价格的 DataFrame
output = pd.DataFrame({'SaleID': test_data[sale_id_column], 'price': predictions.ravel()})

# 保存预测结果到一个 CSV 文件
output.to_csv('used_car_submit.csv', index=False)
