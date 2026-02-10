# Nano Banana Optimizer Design

## 目标
基于用户选择精准生成，最小化成本

## 核心策略

### 1. 精准生成
- 仅对用户确认的方案调用 Nano Banana
- 利用 STEP1 结果优化 STEP2 prompt
- 避免重复生成

### 2. 缓存策略
```python
cache_key = hash(module_list + user_preferences)
if cache_key in cache:
    return cache[cache_key]
```

### 3. 成本追踪
```python
@dataclass
class CostTracker:
    preview_cost: float      # 几乎为零
    generation_cost: float   # Nano Banana 费用
    saved_cost: float        # 避免的无效生成
    
def report(self):
    return f"Saved {self.saved_cost:.2f} ({saved_cost/total*100:.1f}%)"
```

## 优化技巧

### Prompt 优化
- 从 MODULE LIST 提取关键信息
- 精简 prompt 减少 token
- 使用结构化输出格式

### 批量生成优化
- 合并相似请求
- 异步并行调用
- 优先级队列

## 成本对比示例
```
场景: 10篇论文，每篇生成3个预览

传统方式:
- 直接生成: 10次 × $X = $10X
- 成功率假设30%: 实际需要 10/0.3 = 33次
- 总成本: $33X

预览+选择方式:
- STEP1 (批量): 10次 × $0.001X = $0.01X
- 预览生成: 30次 × ~$0 = $0
- 最终生成: 10次 × $X = $10X
- 总成本: ~$10.01X

节省: $23X (70%成本降低)
```

## 回退策略
- Nano Banana 失败时降级到备用模型
- 支持重试机制
