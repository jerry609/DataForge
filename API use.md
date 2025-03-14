# 数据预处理与验证模块 API 文档

## DataCleaner 类

数据清洗模块，负责处理缺失值、异常值、重复数据和噪声数据。

### 初始化

```python
cleaner = DataCleaner(missing_threshold=0.3, extreme_threshold=3.0)
```

- **missing_threshold**: 缺失值处理的阈值，超过该比例的特征或样本将被删除
- **extreme_threshold**: 异常值检测的Z-score阈值

### 方法

#### handle_missing_values

处理数据框中的缺失值。

```python
df_cleaned = cleaner.handle_missing_values(df, strategy='auto')
```

- **df**: 输入数据框
- **strategy**: 处理策略，可选值：
  - 'drop': 删除含有缺失值的行
  - 'mean': 均值填充（数值列）
  - 'median': 中位数填充（数值列）
  - 'mode': 众数填充（所有列）
  - 'forward': 前向填充
  - 'backward': 后向填充
  - 'knn': KNN填充（数值列）
  - 'auto': 自动选择最合适的填充策略

#### detect_outliers

检测数据框中的异常值。

```python
df_cleaned, outliers_df = cleaner.detect_outliers(df, method='zscore')
```

- **df**: 输入数据框
- **method**: 检测方法，可选值：
  - 'zscore': 使用Z-score方法
  - 'iqr': 使用四分位距法
  - 'isolation_forest': 使用孤立森林算法
- 返回元组：(处理后的数据框, 异常值数据框)

#### handle_outliers

处理检测到的异常值。

```python
df_cleaned = cleaner.handle_outliers(df, outliers_df, strategy='winsorize')
```

- **df**: 输入数据框
- **outliers_df**: 异常值数据框（由detect_outliers返回）
- **strategy**: 处理策略，可选值：
  - 'remove': 删除异常值所在行
  - 'winsorize': 截断异常值（设为边界值）
  - 'cap': 限制异常值（使用IQR方法计算边界）

#### remove_duplicates

移除重复数据。

```python
df_cleaned = cleaner.remove_duplicates(df, subset=None)
```

- **df**: 输入数据框
- **subset**: 用于判断重复的列子集，None表示使用所有列

#### filter_noise

过滤噪声数据，使用移动平均等方法平滑数据。

```python
df_cleaned = cleaner.filter_noise(df, columns, window_size=3)
```

- **df**: 输入数据框
- **columns**: 需要平滑的列列表
- **window_size**: 移动窗口大小

## DataNormalizer 类

数据标准化/归一化模块，负责数据的尺度转换。

### 初始化

```python
normalizer = DataNormalizer()
```

### 方法

#### z_score_normalize

Z-Score标准化（均值为0，标准差为1）。

```python
df_normalized = normalizer.z_score_normalize(df, columns=None)
```

- **df**: 输入数据框
- **columns**: 需要标准化的列，None表示所有数值列

#### min_max_normalize

Min-Max归一化（缩放到指定范围）。

```python
df_normalized = normalizer.min_max_normalize(df, columns=None, feature_range=(0, 1))
```

- **df**: 输入数据框
- **columns**: 需要归一化的列，None表示所有数值列
- **feature_range**: 归一化的目标范围，默认为(0, 1)

#### robust_scale

Robust缩放（基于四分位数，对异常值不敏感）。

```python
df_scaled = normalizer.robust_scale(df, columns=None)
```

- **df**: 输入数据框
- **columns**: 需要缩放的列，None表示所有数值列

#### log_transform

对数变换（处理偏斜分布）。

```python
df_transformed = normalizer.log_transform(df, columns=None, base=np.e, offset=1.0)
```

- **df**: 输入数据框
- **columns**: 需要变换的列，None表示所有数值列
- **base**: 对数的底，默认为自然对数e
- **offset**: 添加到值的偏移量，避免对0或负值取对数

#### inverse_transform

逆变换（将标准化/归一化的数据转换回原始尺度）。

```python
df_original = normalizer.inverse_transform(df, method='z_score')
```

- **df**: 输入数据框
- **method**: 使用的方法，可选 'z_score', 'min_max', 'robust'

## DataTransformer 类

数据转换模块，负责类型转换、编码和特征提取。

### 初始化

```python
transformer = DataTransformer()
```

### 方法

#### convert_types

数据类型转换。

```python
df_converted = transformer.convert_types(df, type_map)
```

- **df**: 输入数据框
- **type_map**: 列名到类型的映射，例如 `{'age': 'int', 'price': 'float', 'date': 'datetime'}`

#### one_hot_encode

One-Hot编码（将分类变量转换为二进制向量）。

```python
df_encoded = transformer.one_hot_encode(df, columns, drop_first=False, max_categories=None)
```

- **df**: 输入数据框
- **columns**: 需要编码的分类列
- **drop_first**: 是否删除第一个分类以避免多重共线性
- **max_categories**: 每列保留的最大类别数，超过部分归为"其他"

#### label_encode

标签编码（将分类变量映射为数值）。

```python
df_encoded = transformer.label_encode(df, columns, mapping=None)
```

- **df**: 输入数据框
- **columns**: 需要编码的分类列
- **mapping**: 自定义编码映射，格式为 `{列名: {原值: 编码值}}`

#### extract_datetime_features

从日期时间列提取特征。

```python
df_with_features = transformer.extract_datetime_features(
    df, 
    date_columns, 
    features=['year', 'month', 'day', 'dayofweek', 'hour']
)
```

- **df**: 输入数据框
- **date_columns**: 日期时间列
- **features**: 要提取的特征列表，可选项：
  - 'year': 年份
  - 'month': 月份
  - 'day': 日期
  - 'dayofweek': 星期几（0-6）
  - 'weekday_name': 星期名称
  - 'quarter': 季度
  - 'hour': 小时
  - 'minute': 分钟
  - 'is_weekend': 是否周末
  - 'is_month_start': 是否月初
  - 'is_month_end': 是否月末
  - 'cyclical_month': 循环月份特征（正弦和余弦）
  - 'cyclical_hour': 循环小时特征
  - 'cyclical_dayofweek': 循环星期几特征

## DataValidator 类

数据验证模块，负责检查数据完整性、格式和业务规则。

### 初始化

```python
validator = DataValidator()
```

### 方法

#### check_required_fields

检查必要字段是否存在且非空。

```python
passed, result = validator.check_required_fields(df, required_fields)
```

- **df**: 输入数据框
- **required_fields**: 必要字段列表
- 返回元组：(通过验证?, 详细结果)

#### check_aggregation

验证数据聚合（如各子项之和等于总和）。

```python
passed, result = validator.check_aggregation(df, group_col, sum_col, total_col, tolerance=0.01)
```

- **df**: 输入数据框
- **group_col**: 分组列（如类别、部门等）
- **sum_col**: 求和列（如销售额、数量等）
- **total_col**: 总计列
- **tolerance**: 允许的误差范围

#### check_record_count

验证记录数量是否符合预期。

```python
passed, result = validator.check_record_count(df, expected_count, tolerance_percent=5.0)
```

- **df**: 输入数据框
- **expected_count**: 预期记录数
- **tolerance_percent**: 允许的误差百分比

#### check_foreign_key_integrity

检查外键完整性。

```python
passed, result = validator.check_foreign_key_integrity(df, fk_col, reference_df, reference_col)
```

- **df**: 输入数据框（包含外键）
- **fk_col**: 外键列名
- **reference_df**: 参考数据框（包含主键）
- **reference_col**: 参考表中的主键列名

#### check_data_types

检查数据类型是否符合预期。

```python
passed, result = validator.check_data_types(df, type_specs)
```

- **df**: 输入数据框
- **type_specs**: 列名到预期类型的映射，如 `{'id': 'int', 'name': 'str'}`

#### check_value_ranges

检查值是否在指定范围内。

```python
passed, result = validator.check_value_ranges(df, range_specs)
```

- **df**: 输入数据框
- **range_specs**: 列名到范围规范的映射，如 `{'age': {'min': 0, 'max': 120, 'inclusive': True}}`

#### check_regex_patterns

使用正则表达式验证数据格式。

```python
passed, result = validator.check_regex_patterns(df, pattern_specs)
```

- **df**: 输入数据框
- **pattern_specs**: 列名到正则表达式模式的映射，如 `{'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'}`

#### check_structural_consistency

检查数据结构一致性（列名、类型、约束等）。

```python
passed, result = validator.check_structural_consistency(df, schema)
```

- **df**: 输入数据框
- **schema**: 数据架构定义，如 `{'id': {'type': 'int', 'required': True}}`

#### check_domain_rules

检查领域特定规则。

```python
passed, result = validator.check_domain_rules(df, rule_specs)
```

- **df**: 输入数据框
- **rule_specs**: 规则名称到规则函数的映射，如 `{'positive_value': lambda df: df['value'] > 0}`

#### check_cross_field_relations

检查跨字段关系。

```python
passed, result = validator.check_cross_field_relations(df, relation_specs)
```

- **df**: 输入数据框
- **relation_specs**: 关系名称到验证函数的映射，如 `{'price_check': lambda df: df['price'] <= df['max_price']}`

#### check_time_series_consistency

检查时序数据一致性（异常变化率）。

```python
passed, result = validator.check_time_series_consistency(
    df, 
    time_col, 
    value_col, 
    group_col=None, 
    max_change_percent=200.0
)
```

- **df**: 输入数据框
- **time_col**: 时间列
- **value_col**: 值列
- **group_col**: 分组列（如不同产品、地区等）
- **max_change_percent**: 允许的最大变化百分比

#### check_workflow_compliance

检查业务流程合规性（状态转换是否有效）。

```python
passed, result = validator.check_workflow_compliance(
    df, 
    state_col, 
    timestamp_col, 
    id_col, 
    valid_transitions
)
```

- **df**: 输入数据框
- **state_col**: 状态列
- **timestamp_col**: 时间戳列
- **id_col**: ID列（标识不同实体，如订单、客户等）
- **valid_transitions**: 有效的状态转换词典 `{当前状态: [有效的下一个状态列表]}`

#### get_validation_summary

获取验证结果摘要。

```python
summary = validator.get_validation_summary()
```

## DataProcessor 类

集成的数据处理器，结合清洗、标准化、转换和验证功能。

### 初始化

```python
processor = DataProcessor()
```

### 方法

#### process

根据配置处理数据。

```python
processed_df = processor.process(df, config)
```

- **df**: 输入数据框
- **config**: 处理配置字典，包含清洗、标准化、转换和验证的参数

#### get_processing_summary

获取处理摘要。

```python
summary = processor.get_processing_summary()
```

#### save_processed_data

保存处理后的数据。

```python
processor.save_processed_data(filepath, format='csv', index=False)
```

- **filepath**: 文件路径
- **format**: 文件格式，支持 'csv', 'parquet', 'pickle', 'excel'
- **index**: 是否保存索引

#### save_anomalies

保存检测到的异常数据。

```python
processor.save_anomalies(filepath, format='csv', index=False)
```

- **filepath**: 文件路径
- **format**: 文件格式，支持 'csv', 'parquet', 'pickle', 'excel'
- **index**: 是否保存索引