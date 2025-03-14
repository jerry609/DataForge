# data-process使用文档

本文档提供了模块化数据处理与验证框架的详细使用指南，包括各个组件、配置选项和实际案例。

## 1. 概述

该框架提供了一个全面的数据处理解决方案，包括数据清洗、标准化、转换和验证功能。它由以下主要组件组成：

- **DataCleaner**: 处理缺失值、异常值、重复数据和噪声
- **DataNormalizer**: 进行数据标准化、归一化和变换
- **DataTransformer**: 处理类型转换、特征编码和特征提取
- **DataValidator**: 验证数据完整性、格式和业务规则
- **DataProcessor**: 集成以上模块，提供统一接口

## 2. 安装与导入

```python
# 假设您已将代码保存为data_processing.py
from data_processing import DataProcessor, DataCleaner, DataNormalizer, DataTransformer, DataValidator
```

## 3. 基本使用流程

### 3.1 初始化处理器

```python
processor = DataProcessor()
```

### 3.2 配置处理参数

```python
config = {
    'cleaning': {
        # 数据清洗配置
    },
    'normalization': {
        # 数据标准化配置
    },
    'transformation': {
        # 数据转换配置
    },
    'validation': {
        # 数据验证配置
    }
}
```

### 3.3 处理数据

```python
processed_df = processor.process(df, config)
```

### 3.4 获取处理摘要

```python
summary = processor.get_processing_summary()
print(summary)
```

### 3.5 保存处理结果

```python
processor.save_processed_data('processed_data.csv')
processor.save_anomalies('anomalies.csv')
```

## 4. 数据清洗 (DataCleaner)

### 4.1 处理缺失值

```python
'cleaning': {
    'missing_values': {
        'strategy': 'auto'  # 可选: 'auto', 'drop', 'mean', 'median', 'mode', 'forward', 'backward', 'knn'
    }
}
```

- **auto**: 自动选择适合每列数据类型的策略
- **drop**: 删除含有缺失值的行
- **mean/median/mode**: 使用均值/中位数/众数填充
- **forward/backward**: 前向/后向填充
- **knn**: 使用KNN算法填充

### 4.2 处理异常值

```python
'cleaning': {
    'outliers': {
        'detection_method': 'zscore',  # 可选: 'zscore', 'iqr', 'isolation_forest'
        'handle': True,
        'handle_strategy': 'winsorize'  # 可选: 'remove', 'winsorize', 'cap'
    }
}
```

- **检测方法**:
  - **zscore**: 基于Z分数检测
  - **iqr**: 基于四分位距(IQR)检测
  - **isolation_forest**: 使用隔离森林算法检测

- **处理策略**:
  - **remove**: 删除异常值行
  - **winsorize**: 将异常值限制在分位数范围内
  - **cap**: 使用IQR边界截断异常值

### 4.3 移除重复数据

```python
'cleaning': {
    'remove_duplicates': True,
    'duplicate_subset': ['col1', 'col2']  # 可选，指定列子集
}
```

### 4.4 过滤噪声数据

```python
'cleaning': {
    'noise_filtering': {
        'columns': ['value1', 'value2'],
        'window_size': 3  # 移动平均窗口大小
    }
}
```

## 5. 数据标准化 (DataNormalizer)

### 5.1 Z-Score标准化

```python
'normalization': {
    'z_score': True,
    'z_score_columns': ['col1', 'col2']  # 可选，默认所有数值列
}
```

### 5.2 Min-Max归一化

```python
'normalization': {
    'min_max': True,
    'min_max_columns': ['col1', 'col2'],  # 可选
    'min_max_range': (0, 1)  # 可选，指定目标范围
}
```

### 5.3 Robust缩放

```python
'normalization': {
    'robust': True,
    'robust_columns': ['col1', 'col2']  # 可选
}
```

### 5.4 对数变换

```python
'normalization': {
    'log_transform': True,
    'log_columns': ['col1', 'col2'],  # 可选
    'log_base': 10,  # 可选，默认为e
    'log_offset': 1.0  # 可选，防止取0或负值的对数
}
```

## 6. 数据转换 (DataTransformer)

### 6.1 类型转换

```python
'transformation': {
    'type_conversion': {
        'col1': 'int',
        'col2': 'float',
        'col3': 'datetime',
        'col4': 'category',
        'col5': 'str'
    }
}
```

### 6.2 One-Hot编码

```python
'transformation': {
    'one_hot_encoding': {
        'columns': ['category1', 'category2'],
        'drop_first': True,  # 可选，避免多重共线性
        'max_categories': 10  # 可选，限制类别数量
    }
}
```

### 6.3 标签编码

```python
'transformation': {
    'label_encoding': {
        'columns': ['category1', 'category2'],
        'mapping': {  # 可选，自定义映射
            'category1': {'A': 0, 'B': 1, 'C': 2}
        }
    }
}
```

### 6.4 时间特征提取

```python
'transformation': {
    'datetime_features': {
        'columns': ['date'],
        'features': [
            'year', 'month', 'day', 'dayofweek', 
            'weekday_name', 'quarter', 'hour', 'minute',
            'is_weekend', 'is_month_start', 'is_month_end',
            'cyclical_month', 'cyclical_hour', 'cyclical_dayofweek'
        ]
    }
}
```

## 7. 数据验证 (DataValidator)

### 7.1 完整性检查

```python
'validation': {
    'completeness': {
        'required_fields': ['id', 'name', 'date'],
        'record_count': {
            'expected': 1000,
            'tolerance_percent': 5.0
        },
        'aggregation': {
            'group_col': 'category',
            'sum_col': 'subtotal',
            'total_col': 'total',
            'tolerance': 0.01
        },
        'foreign_key': {
            'fk_col': 'product_id',
            'reference_df': products_df,
            'reference_col': 'id'
        }
    }
}
```

### 7.2 格式验证

```python
'validation': {
    'format': {
        'data_types': {
            'id': 'int',
            'price': 'float',
            'date': 'datetime'
        },
        'value_ranges': {
            'price': {'min': 0, 'max': 1000, 'inclusive': True},
            'age': {'min': 18, 'max': None}
        },
        'regex_patterns': {
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'phone': r'^\d{3}-\d{3}-\d{4}$'
        },
        'schema': {
            'id': {'type': 'int', 'required': True},
            'name': {'type': 'str', 'required': True},
            'age': {'type': 'int', 'required': False}
        }
    }
}
```

### 7.3 业务规则验证

```python
'validation': {
    'business_rules': {
        'domain_rules': {
            'positive_price': lambda df: df['price'] > 0,
            'valid_age': lambda df: (df['age'] >= 18) & (df['age'] <= 100)
        },
        'cross_field_relations': {
            'discount_valid': lambda df: df['discounted_price'] <= df['original_price'],
            'end_date_after_start': lambda df: df['end_date'] > df['start_date']
        },
        'time_series_consistency': {
            'time_col': 'date',
            'value_col': 'value',
            'group_col': 'category',  # 可选
            'max_change_percent': 200.0
        },
        'workflow_compliance': {
            'state_col': 'status',
            'timestamp_col': 'update_time',
            'id_col': 'order_id',
            'valid_transitions': {
                'new': ['in_progress', 'cancelled'],
                'in_progress': ['completed', 'on_hold'],
                'on_hold': ['in_progress', 'cancelled'],
                'completed': ['closed'],
                'cancelled': []
            }
        }
    }
}
```

## 8. 高级配置示例

### 8.1 时间序列数据处理

```python
config = {
    'cleaning': {
        'missing_values': {'strategy': 'forward'},
        'outliers': {'detection_method': 'isolation_forest', 'handle_strategy': 'winsorize'},
        'noise_filtering': {'columns': ['value'], 'window_size': 5}
    },
    'transformation': {
        'datetime_features': {
            'columns': ['date'],
            'features': ['year', 'month', 'dayofweek', 'is_weekend', 'cyclical_month']
        }
    },
    'validation': {
        'business_rules': {
            'time_series_consistency': {
                'time_col': 'date',
                'value_col': 'value',
                'group_col': 'category',
                'max_change_percent': 50.0
            }
        }
    }
}
```

### 8.2 销售数据处理

```python
config = {
    'cleaning': {
        'missing_values': {'strategy': 'auto'},
        'outliers': {'detection_method': 'iqr', 'handle_strategy': 'cap'}
    },
    'normalization': {
        'robust': True,
        'robust_columns': ['revenue', 'cost', 'profit']
    },
    'transformation': {
        'one_hot_encoding': {'columns': ['product', 'region']},
        'datetime_features': {'columns': ['date'], 'features': ['year', 'month', 'quarter']}
    },
    'validation': {
        'business_rules': {
            'domain_rules': {
                'revenue_equals_quantity_times_price': lambda df: 
                    (df['revenue'] - df['quantity'] * df['unit_price']).abs() < 0.01
            }
        }
    }
}
```

## 9. 实用技巧

1. **选择合适的缺失值处理策略**:
   - 时间序列数据通常使用`forward`填充
   - 分类变量通常使用`mode`填充
   - 数值变量可以使用`mean`或`median`填充

2. **异常值处理**:
   - 对于需要保留所有数据点的情况，使用`winsorize`或`cap`
   - 如果确信异常值是错误的，可以使用`remove`

3. **性能优化**:
   - 对大数据集，先对关键列进行处理
   - 考虑分批处理大型数据集

4. **自定义处理流程**:
   - 可以单独使用各个模块进行特定处理
   - 例如：`cleaner = DataCleaner(); cleaned_df = cleaner.handle_missing_values(df)`

## 10. 日志与监控

该框架配备了详细的日志记录功能，可以帮助您监控和调试数据处理过程:

```python
# 日志会记录到控制台和data_processor.log文件
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data_processor.log")
    ]
)
```

## 11. 常见问题

### Q: 如何解决KNN填充中的内存问题?
A: 对于大型数据集，可以限制KNN填充应用的列，或增加n_neighbors参数提高效率。

### Q: 如何处理极高基数的分类变量?
A: 使用one_hot_encoding时设置max_categories参数，将低频类别归为"其他"类别。

### Q: 如何自定义业务规则验证?
A: 创建返回布尔Series的lambda函数，通过domain_rules或cross_field_relations添加到配置中。