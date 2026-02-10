# Quality Evaluator Design

## 目标
自动评估和排序多个预览方案，提供客观的质量评分和排序推荐。

## 评估维度

### 1. 结构完整性评分 (Structural Completeness)
**权重: 30%**

检查 MODULE LIST 是否包含必要组件，验证数据流完整性。

#### 检查项
| 检查项 | 分值 | 说明 |
|--------|------|------|
| Input 节点存在 | 10 | 至少一个输入源 |
| Process 节点存在 | 10 | 至少一个处理节点 |
| Output 节点存在 | 10 | 至少一个输出目标 |

#### 算法
```python
def check_structure(module_list: dict) -> float:
    """
    结构完整性评分 0-30
    """
    score = 0
    node_types = {node['type'] for node in module_list.get('nodes', [])}
    
    # 必要组件检查
    required_types = ['input', 'process', 'output']
    for node_type in required_types:
        if any(t.startswith(node_type) for t in node_types):
            score += 10
    
    return score
```

---

### 2. 布局合理性评分 (Layout Rationality)
**权重: 25%**

评估组件数量和层级深度是否合理。

#### 检查项
| 检查项 | 分值 | 标准 |
|--------|------|------|
| 组件数量适中 | 10 | 5-15个节点为佳 |
| 层级深度合理 | 10 | 最大深度3-5层 |
| 无孤立节点 | 5 | 所有节点有连接 |

#### 算法
```python
def evaluate_layout(preview: dict) -> float:
    """
    布局合理性评分 0-25
    """
    score = 25
    nodes = preview.get('nodes', [])
    edges = preview.get('edges', [])
    
    # 组件数量检查 (最优: 5-15)
    node_count = len(nodes)
    if node_count < 3:
        score -= 8  # 过少
    elif node_count > 20:
        score -= 8  # 过多
    elif 5 <= node_count <= 15:
        pass  # 最优范围
    else:
        score -= 3  # 轻微偏离
    
    # 计算层级深度 (BFS)
    max_depth = calculate_max_depth(nodes, edges)
    if max_depth < 2:
        score -= 5  # 过于扁平
    elif max_depth > 6:
        score -= 5  # 层级过深
    
    # 孤立节点检查
    connected_nodes = set()
    for edge in edges:
        connected_nodes.add(edge['source'])
        connected_nodes.add(edge['target'])
    isolated = len(nodes) - len(connected_nodes)
    score -= isolated * 5  # 每个孤立节点扣5分
    
    return max(0, score)

def calculate_max_depth(nodes, edges):
    """计算图的最大深度 (从Input到Output的最长路径)"""
    from collections import defaultdict, deque
    
    # 构建邻接表
    graph = defaultdict(list)
    in_degree = defaultdict(int)
    node_ids = {n['id'] for n in nodes}
    
    for edge in edges:
        if edge['source'] in node_ids and edge['target'] in node_ids:
            graph[edge['source']].append(edge['target'])
            in_degree[edge['target']] += 1
    
    # 找到所有起始节点 (入度为0)
    start_nodes = [n['id'] for n in nodes if in_degree[n['id']] == 0]
    
    # BFS计算深度
    depth = {node_id: 1 for node_id in start_nodes}
    queue = deque(start_nodes)
    max_depth = 1
    
    while queue:
        current = queue.popleft()
        for neighbor in graph[current]:
            depth[neighbor] = max(depth.get(neighbor, 0), depth[current] + 1)
            max_depth = max(max_depth, depth[neighbor])
            queue.append(neighbor)
    
    return max_depth
```

---

### 3. 语义清晰度评分 (Semantic Clarity)
**权重: 25%**

评估标签清晰度和连接关系明确性。

#### 检查项
| 检查项 | 分值 | 标准 |
|--------|------|------|
| 标签非空 | 8 | 所有节点有标签 |
| 标签语义清晰 | 8 | 标签有意义且简洁 |
| 连接有类型 | 5 | 边有明确的连接类型 |
| 无歧义命名 | 4 | 避免同名不同义 |

#### 算法
```python
def check_clarity(preview: dict) -> float:
    """
    语义清晰度评分 0-25
    """
    score = 25
    nodes = preview.get('nodes', [])
    edges = preview.get('edges', [])
    
    # 标签非空检查
    empty_labels = sum(1 for n in nodes if not n.get('label', '').strip())
    score -= empty_labels * 2
    
    # 标签语义检查 (长度和关键词)
    vague_keywords = ['temp', 'tmp', 'data', 'item', 'stuff', 'thing']
    for node in nodes:
        label = node.get('label', '').lower()
        if any(kw in label for kw in vague_keywords):
            score -= 2
        if len(label) < 2 or len(label) > 30:
            score -= 1
    
    # 连接类型检查
    untyped_edges = sum(1 for e in edges if not e.get('type') or e.get('type') == 'default')
    score -= untyped_edges * 0.5
    
    # 命名冲突检查
    labels = [n.get('label', '').lower() for n in nodes]
    duplicates = len(labels) - len(set(labels))
    score -= duplicates * 3
    
    return max(0, score)
```

---

### 4. 创新度评分 (Innovation Score)
**权重: 20%**

评估设计的独特性，避免过于模板化。

#### 检查项
| 检查项 | 分值 | 标准 |
|--------|------|------|
| 非标准结构 | 8 | 非简单线性流程 |
| 多样节点类型 | 6 | 使用多种处理类型 |
| 有反馈/循环 | 6 | 存在反馈连接 |

#### 算法
```python
def evaluate_innovation(preview: dict) -> float:
    """
    创新度评分 0-20
    """
    score = 0
    nodes = preview.get('nodes', [])
    edges = preview.get('edges', [])
    
    # 结构复杂度 (非线性)
    node_count = len(nodes)
    edge_count = len(edges)
    
    # 平均出度 > 1 表示有分支
    out_degree = {}
    for edge in edges:
        out_degree[edge['source']] = out_degree.get(edge['source'], 0) + 1
    
    avg_out_degree = sum(out_degree.values()) / len(nodes) if nodes else 0
    
    if avg_out_degree > 1.2:
        score += 8  # 有分支结构
    elif avg_out_degree > 1.0:
        score += 4  # 轻微分支
    
    # 节点类型多样性
    node_types = set(n.get('type', 'unknown') for n in nodes)
    type_count = len(node_types)
    if type_count >= 5:
        score += 6
    elif type_count >= 3:
        score += 3
    
    # 检查循环/反馈
    has_cycle = detect_cycle(edges)
    if has_cycle:
        score += 6
    
    return min(20, score)

def detect_cycle(edges):
    """检测图中是否存在环"""
    from collections import defaultdict
    
    graph = defaultdict(list)
    nodes = set()
    for edge in edges:
        graph[edge['source']].append(edge['target'])
        nodes.add(edge['source'])
        nodes.add(edge['target'])
    
    # DFS检测环
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {node: WHITE for node in nodes}
    
    def dfs(node):
        color[node] = GRAY
        for neighbor in graph[node]:
            if color[neighbor] == GRAY:
                return True  # 发现回边，存在环
            if color[neighbor] == WHITE and dfs(neighbor):
                return True
        color[node] = BLACK
        return False
    
    for node in nodes:
        if color[node] == WHITE:
            if dfs(node):
                return True
    return False
```

---

## 综合评分算法

### 主评分函数
```python
from dataclasses import dataclass

@dataclass
class Score:
    structural: float      # 0-30
    layout: float          # 0-25
    clarity: float         # 0-25
    innovation: float      # 0-20
    total: float           # 0-100
    
    def to_dict(self):
        return {
            'structural': round(self.structural, 2),
            'layout': round(self.layout, 2),
            'clarity': round(self.clarity, 2),
            'innovation': round(self.innovation, 2),
            'total': round(self.total, 2)
        }


def evaluate_preview(
    module_list: dict,
    preview: dict
) -> Score:
    """
    综合评估预览方案，返回 0-100 的评分
    
    Args:
        module_list: 模块清单，包含节点类型信息
        preview: 预览图数据，包含 nodes 和 edges
    
    Returns:
        Score: 各维度评分和总分
    """
    # 各维度评分
    structural = check_structure(module_list)
    layout = evaluate_layout(preview)
    clarity = check_clarity(preview)
    innovation = evaluate_innovation(preview)
    
    # 计算加权总分
    total = structural + layout + clarity + innovation
    
    return Score(
        structural=structural,
        layout=layout,
        clarity=clarity,
        innovation=innovation,
        total=total
    )


def weighted_sum(
    structural: float,
    layout: float,
    clarity: float,
    innovation: float
) -> float:
    """
    权重分配:
    - 结构完整性: 30%
    - 布局合理性: 25%
    - 语义清晰度: 25%
    - 创新度: 20%
    """
    return structural + layout + clarity + innovation
```

---

## 排序策略

### 排序算法
```python
from typing import List, Tuple

def rank_previews(
    previews: List[Tuple[str, dict, dict]],
    top_n: int = 5
) -> List[Tuple[str, Score]]:
    """
    对多个预览方案进行排序
    
    Args:
        previews: [(preview_id, module_list, preview_data), ...]
        top_n: 返回前N个结果
    
    Returns:
        排序后的 [(preview_id, score), ...]
    """
    scored = []
    
    for preview_id, module_list, preview_data in previews:
        score = evaluate_preview(module_list, preview_data)
        scored.append((preview_id, score))
    
    # 排序: 总分降序，同分按结构完整性优先
    scored.sort(key=lambda x: (x[1].total, x[1].structural), reverse=True)
    
    return scored[:top_n]
```

### 排序规则
1. **首要排序**: 综合总分降序
2. **次要排序**: 同分时，结构完整性分数高者优先
3. **返回数量**: 默认返回 TOP-5，可配置

---

## 人工反馈集成

### 反馈记录
```python
from datetime import datetime
from typing import Optional

@dataclass
class UserFeedback:
    preview_id: str
    user_choice: bool  # True = 被选中
    rating: Optional[int]  # 1-5 星评分 (可选)
    timestamp: datetime
    context: dict  # 当时的评估分数


class FeedbackStore:
    """用户反馈存储"""
    
    def __init__(self, db_path: str = "feedback.json"):
        self.db_path = db_path
        self.feedbacks = []
    
    def record(self, feedback: UserFeedback):
        """记录用户选择"""
        self.feedbacks.append(feedback)
        self._save()
    
    def get_recent(self, n: int = 100):
        """获取最近N条反馈"""
        return self.feedbacks[-n:]
```

### 在线学习优化
```python
class AdaptiveWeights:
    """
    根据用户反馈动态调整权重
    """
    
    def __init__(self):
        self.weights = {
            'structural': 0.30,
            'layout': 0.25,
            'clarity': 0.25,
            'innovation': 0.20
        }
        self.learning_rate = 0.01
    
    def update(self, feedbacks: List[UserFeedback]):
        """
        根据反馈调整权重
        
        原理: 如果某维度高分但用户不选，降低该维度权重
              如果某维度高分且用户选择，保持/增加权重
        """
        if not feedbacks:
            return
        
        for fb in feedbacks:
            if not fb.user_choice:
                # 分析哪个维度分数最高但未被选择
                context = fb.context
                max_dim = max(context, key=context.get)
                # 轻微降低该维度权重
                self.weights[max_dim] -= self.learning_rate
        
        # 归一化
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}
    
    def get_weights(self) -> dict:
        return self.weights.copy()
```

### 反馈闭环流程
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   生成预览   │ → │   质量评分   │ → │  推荐TOP-N  │
└─────────────┘    └─────────────┘    └──────┬──────┘
                                             ↓
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   优化权重   │ ← │   在线学习   │ ← │  用户选择   │
└─────────────┘    └─────────────┘    └─────────────┘
```

---

## 使用示例

```python
# 示例数据
module_list = {
    'nodes': [
        {'id': 'input_1', 'type': 'input_api'},
        {'id': 'process_1', 'type': 'process_transform'},
        {'id': 'process_2', 'type': 'process_filter'},
        {'id': 'output_1', 'type': 'output_db'}
    ]
}

preview = {
    'nodes': [
        {'id': 'input_1', 'label': 'API Gateway', 'type': 'input_api'},
        {'id': 'process_1', 'label': 'Data Transform', 'type': 'process_transform'},
        {'id': 'process_2', 'label': 'Filter Rules', 'type': 'process_filter'},
        {'id': 'output_1', 'label': 'Database', 'type': 'output_db'}
    ],
    'edges': [
        {'source': 'input_1', 'target': 'process_1', 'type': 'data_flow'},
        {'source': 'process_1', 'target': 'process_2', 'type': 'transformed'},
        {'source': 'process_2', 'target': 'output_1', 'type': 'filtered'}
    ]
}

# 评估
score = evaluate_preview(module_list, preview)
print(f"总分: {score.total}")
print(f"结构完整性: {score.structural}")
print(f"布局合理性: {score.layout}")
print(f"语义清晰度: {score.clarity}")
print(f"创新度: {score.innovation}")

# 排序多个方案
previews = [
    ('preview_1', module_list_1, preview_1),
    ('preview_2', module_list_2, preview_2),
    ('preview_3', module_list_3, preview_3),
]
ranked = rank_previews(previews, top_n=3)
for preview_id, score in ranked:
    print(f"{preview_id}: {score.total}")
```

---

## 配置参数

```python
EVALUATION_CONFIG = {
    # 权重配置
    'weights': {
        'structural': 0.30,
        'layout': 0.25,
        'clarity': 0.25,
        'innovation': 0.20
    },
    
    # 布局参数
    'layout': {
        'optimal_node_count': (5, 15),
        'optimal_depth': (3, 5),
        'penalty_isolated_node': 5
    },
    
    # 清晰度参数
    'clarity': {
        'max_label_length': 30,
        'min_label_length': 2,
        'vague_keywords': ['temp', 'tmp', 'data', 'item', 'stuff', 'thing']
    },
    
    # 创新度参数
    'innovation': {
        'branching_threshold': 1.2,  # 平均出度阈值
        'min_diverse_types': 3
    },
    
    # 学习参数
    'learning': {
        'learning_rate': 0.01,
        'feedback_batch_size': 50
    }
}
```
