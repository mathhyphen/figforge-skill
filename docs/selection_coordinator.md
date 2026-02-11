# Selection Coordinator Design

## 目标
优雅地展示预览并收集用户选择

## 交互流程

### 方式一: Telegram 按钮交互
```
请选择最佳预览方案:

[Preview 1] [Preview 2] [Preview 3]
   🥇         🥈         🥉
  
[查看详情] [确认选择] [重新生成]
```

### 方式二: 命令行交互
```bash
$ figforge batch -i papers/

Generated 3 previews:

[1] Mermaid diagram (Score: 85)
[2] ASCII art      (Score: 78) 
[3] HTML preview   (Score: 82)

Select best option (1-3): 1
Confirm generate with nano banana? [Y/n]: Y
```

### 方式三: Web UI
- 并排对比多个预览
- 悬停查看详情
- 点击选择

## 状态管理
```python
class SelectionState:
    batch_id: str
    papers: List[PaperInput]
    previews: List[Preview]
    selected_indices: Dict[str, int]  # paper_id -> preview_idx
    confirmed: bool
```

## 超时处理
- 默认超时: 5分钟
- 超时后自动选择最高分
- 可配置

## 取消/重试
- 支持取消当前批次
- 支持重新生成特定预览
