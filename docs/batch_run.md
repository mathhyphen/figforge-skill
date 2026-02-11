# Batch Run Design

## 功能概述

批量 STEP1 处理器（Batch STEP1 Processor）用于并行处理多篇学术论文的 MODULE LIST 生成任务。该系统通过协调多个 Worker Agent 并发工作，大幅提升大批量论文的处理效率。

**核心目标：**
- 支持 3-6 个 Worker Agent 并行处理
- 提供流式结果收集和实时进度追踪
- 具备完善的错误处理和自动重试机制
- 支持部分失败处理和断点续传

---

## 输入/输出规范

### 数据模型

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Union
from datetime import datetime
from enum import Enum


class ProcessingStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    RETRYING = "retrying"


@dataclass
class PaperInput:
    """单篇论文输入数据"""
    id: str                          # 论文唯一标识
    text: str                        # 论文正文内容
    metadata: Dict[str, Any]         # 元数据（标题、作者、来源等）
    priority: int = 0                # 处理优先级（数值越高越优先）
    created_at: Optional[datetime] = None


@dataclass
class ModuleListResult:
    """单篇论文的 MODULE LIST 生成结果"""
    paper_id: str                    # 对应输入论文 ID
    success: bool                    # 处理是否成功
    modules: List[Dict[str, Any]]    # 生成的模块列表
    raw_output: Optional[str] = None # LLM 原始输出（用于调试）
    error_message: Optional[str] = None
    processing_time_ms: int = 0
    retry_count: int = 0


@dataclass
class BatchStats:
    """批量处理统计信息"""
    total: int = 0                   # 总任务数
    completed: int = 0               # 完成数
    successful: int = 0              # 成功数
    failed: int = 0                  # 失败数
    pending: int = 0                 # 待处理数
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    @property
    def duration_seconds(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    @property
    def throughput_per_minute(self) -> float:
        if self.duration_seconds > 0:
            return (self.completed / self.duration_seconds) * 60
        return 0.0


@dataclass
class BatchResult:
    """批量处理最终返回结果"""
    results: List[ModuleListResult]  # 所有成功的结果
    failed: List[str]                # 失败的论文 ID 列表
    stats: BatchStats                # 处理统计信息
    partial_results: List[ModuleListResult] = field(default_factory=list)  # 部分失败时的结果


@dataclass
class WorkerTask:
    """分配给 Worker 的任务单元"""
    task_id: str
    paper: PaperInput
    attempt: int = 1
    max_retries: int = 3
    timeout_seconds: int = 300
```

---

## 子 Agent 协调方案

### 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                    Batch Coordinator                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Worker 1   │  │   Worker 2   │  │   Worker N   │      │
│  │  (Agent 1)   │  │  (Agent 2)   │  │  (Agent N)   │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                 │                 │               │
│         └─────────────────┴─────────────────┘               │
│                           │                                 │
│                    ┌──────┴──────┐                          │
│                    │ Result Queue│                          │
│                    └─────────────┘                          │
└─────────────────────────────────────────────────────────────┘
```

### Worker Agent 配置

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| Worker 数量 | 3-6 | 根据 API 限制和系统资源调整 |
| 任务队列大小 | 1000 | 待处理任务队列容量 |
| 结果队列大小 | 1000 | 已完成结果队列容量 |
| 心跳间隔 | 10s | Worker 健康检查间隔 |

### 任务分配策略

**策略 1: 轮询分配 (Round-Robin)**
```python
def assign_task_round_robin(tasks: List[WorkerTask], workers: int) -> List[List[WorkerTask]]:
    """简单轮询，适合任务复杂度均匀的场景"""
    buckets = [[] for _ in range(workers)]
    for i, task in enumerate(tasks):
        buckets[i % workers].append(task)
    return buckets
```

**策略 2: 按长度分配 (Length-Based)**
```python
def assign_task_by_length(tasks: List[WorkerTask], workers: int) -> List[List[WorkerTask]]:
    """按论文长度分配，平衡各 Worker 负载"""
    # 按文本长度降序排序
    sorted_tasks = sorted(tasks, key=lambda t: len(t.paper.text), reverse=True)
    
    # 使用贪心算法分配到当前总负载最小的 Worker
    buckets = [[] for _ in range(workers)]
    bucket_loads = [0] * workers
    
    for task in sorted_tasks:
        # 找到当前负载最小的 bucket
        min_idx = bucket_loads.index(min(bucket_loads))
        buckets[min_idx].append(task)
        bucket_loads[min_idx] += len(task.paper.text)
    
    return buckets
```

**策略 3: 优先级加权 (Priority-Weighted)**
```python
def assign_task_priority_weighted(tasks: List[WorkerTask], workers: int) -> List[List[WorkerTask]]:
    """按优先级加权分配，高优先级任务优先处理"""
    # 按优先级分组
    priority_groups = {}
    for task in tasks:
        p = task.paper.priority
        if p not in priority_groups:
            priority_groups[p] = []
        priority_groups[p].append(task)
    
    # 高优先级先分配，内部使用轮询
    buckets = [[] for _ in range(workers)]
    worker_idx = 0
    
    for priority in sorted(priority_groups.keys(), reverse=True):
        for task in priority_groups[priority]:
            buckets[worker_idx % workers].append(task)
            worker_idx += 1
    
    return buckets
```

### 结果聚合方式

**流式收集 (Streaming Collection)**
```python
from queue import Queue
import threading

class ResultAggregator:
    """流式结果收集器"""
    
    def __init__(self):
        self.result_queue = Queue()
        self.results = []
        self._stop_event = threading.Event()
        self._collector_thread = None
    
    def start(self):
        """启动后台收集线程"""
        self._collector_thread = threading.Thread(target=self._collect_loop)
        self._collector_thread.start()
    
    def _collect_loop(self):
        """持续收集结果的循环"""
        while not self._stop_event.is_set():
            try:
                result = self.result_queue.get(timeout=1.0)
                if result is not None:
                    self.results.append(result)
                    self._notify_callback(result)
            except:
                continue
    
    def submit_result(self, result: ModuleListResult):
        """Worker 调用此方法提交结果"""
        self.result_queue.put(result)
    
    def _notify_callback(self, result: ModuleListResult):
        """通知进度回调"""
        if self.on_result:
            self.on_result(result)
    
    def stop(self):
        """停止收集"""
        self._stop_event.set()
        if self._collector_thread:
            self._collector_thread.join()
```

---

## 并行策略

### 主入口函数

```python
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from typing import Iterator


def batch_step1(
    papers: List[PaperInput],
    max_workers: int = 3,
    callback: Optional[Callable[[ModuleListResult], None]] = None,
    progress_callback: Optional[Callable[[BatchStats], None]] = None,
    task_assignment_strategy: str = "length_based",
    enable_caching: bool = True,
    cache_ttl_hours: int = 24
) -> BatchResult:
    """
    批量执行 STEP1 处理
    
    Args:
        papers: 待处理的论文列表
        max_workers: 并行 Worker 数量 (3-6)
        callback: 单个结果回调函数
        progress_callback: 进度统计回调函数
        task_assignment_strategy: 任务分配策略 ("round_robin" | "length_based" | "priority")
        enable_caching: 是否启用缓存
        cache_ttl_hours: 缓存有效期（小时）
    
    Returns:
        BatchResult: 包含所有成功结果、失败ID和统计信息
    """
    
    # 初始化统计信息
    stats = BatchStats(
        total=len(papers),
        pending=len(papers),
        start_time=datetime.now()
    )
    
    # 初始化缓存（如果启用）
    cache = CacheManager(ttl_hours=cache_ttl_hours) if enable_caching else None
    
    # 创建任务列表
    tasks = [
        WorkerTask(
            task_id=f"task_{paper.id}",
            paper=paper,
            max_retries=3,
            timeout_seconds=300
        )
        for paper in papers
    ]
    
    # 应用任务分配策略
    if task_assignment_strategy == "round_robin":
        task_buckets = assign_task_round_robin(tasks, max_workers)
    elif task_assignment_strategy == "length_based":
        task_buckets = assign_task_by_length(tasks, max_workers)
    elif task_assignment_strategy == "priority":
        task_buckets = assign_task_priority_weighted(tasks, max_workers)
    else:
        task_buckets = assign_task_round_robin(tasks, max_workers)
    
    # 初始化结果收集器
    aggregator = ResultAggregator()
    aggregator.on_result = callback
    aggregator.start()
    
    # 执行并行处理
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交 Worker 任务
            future_to_worker = {
                executor.submit(
                    _worker_process_bucket,
                    worker_id=i,
                    tasks=bucket,
                    cache=cache,
                    aggregator=aggregator,
                    stats=stats
                ): i
                for i, bucket in enumerate(task_buckets) if bucket
            }
            
            # 等待所有 Worker 完成
            for future in as_completed(future_to_worker):
                worker_id = future_to_worker[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"Worker {worker_id} failed: {e}")
    finally:
        aggregator.stop()
    
    # 完成统计
    stats.end_time = datetime.now()
    stats.pending = 0
    
    # 构建最终结果
    results = aggregator.results
    successful_results = [r for r in results if r.success]
    failed_ids = [r.paper_id for r in results if not r.success]
    
    stats.successful = len(successful_results)
    stats.failed = len(failed_ids)
    stats.completed = stats.total
    
    return BatchResult(
        results=successful_results,
        failed=failed_ids,
        stats=stats,
        partial_results=results if failed_ids else []
    )
```

### Worker 实现

```python
def _worker_process_bucket(
    worker_id: int,
    tasks: List[WorkerTask],
    cache: Optional[CacheManager],
    aggregator: ResultAggregator,
    stats: BatchStats
) -> None:
    """
    单个 Worker 处理分配给它的任务桶
    
    Args:
        worker_id: Worker 标识
        tasks: 分配给该 Worker 的任务列表
        cache: 缓存管理器
        aggregator: 结果聚合器
        stats: 共享统计信息（需线程安全更新）
    """
    
    print(f"[Worker {worker_id}] Started with {len(tasks)} tasks")
    
    for task in tasks:
        result = _process_single_task(task, cache, worker_id)
        
        # 更新统计（使用锁保证线程安全）
        with threading.Lock():
            stats.pending -= 1
            if result.success:
                stats.successful += 1
            else:
                stats.failed += 1
        
        # 提交结果
        aggregator.submit_result(result)
    
    print(f"[Worker {worker_id}] Completed")


def _process_single_task(
    task: WorkerTask,
    cache: Optional[CacheManager],
    worker_id: int
) -> ModuleListResult:
    """处理单个任务，包含重试逻辑"""
    
    paper = task.paper
    start_time = time.time()
    
    # 检查缓存
    if cache:
        cached_result = cache.get(paper.id)
        if cached_result:
            print(f"[Worker {worker_id}] Cache hit for {paper.id}")
            return cached_result
    
    # 执行处理（带重试）
    last_error = None
    raw_output = None
    
    for attempt in range(1, task.max_retries + 1):
        try:
            print(f"[Worker {worker_id}] Processing {paper.id} (attempt {attempt})")
            
            # 调用 STEP1 处理逻辑
            result = _execute_step1_with_timeout(
                paper=paper,
                timeout_seconds=task.timeout_seconds
            )
            
            processing_time = int((time.time() - start_time) * 1000)
            
            module_result = ModuleListResult(
                paper_id=paper.id,
                success=True,
                modules=result["modules"],
                raw_output=result.get("raw_output"),
                processing_time_ms=processing_time,
                retry_count=attempt - 1
            )
            
            # 写入缓存
            if cache:
                cache.set(paper.id, module_result)
            
            return module_result
            
        except TimeoutError:
            last_error = f"Timeout after {task.timeout_seconds}s"
            print(f"[Worker {worker_id}] Timeout for {paper.id}, attempt {attempt}")
            
        except Exception as e:
            last_error = str(e)
            print(f"[Worker {worker_id}] Error processing {paper.id}: {e}")
        
        # 重试前等待（指数退避）
        if attempt < task.max_retries:
            wait_time = min(2 ** attempt, 30)  # 最大等待 30 秒
            time.sleep(wait_time)
    
    # 所有重试失败
    processing_time = int((time.time() - start_time) * 1000)
    
    return ModuleListResult(
        paper_id=paper.id,
        success=False,
        modules=[],
        raw_output=raw_output,
        error_message=last_error,
        processing_time_ms=processing_time,
        retry_count=task.max_retries
    )


def _execute_step1_with_timeout(
    paper: PaperInput,
    timeout_seconds: int
) -> Dict[str, Any]:
    """执行 STEP1 处理，带超时控制"""
    
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Processing exceeded {timeout_seconds} seconds")
    
    # 设置超时（Unix 系统）
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    
    try:
        # 调用实际的 STEP1 处理函数
        result = execute_step1_core(
            paper_text=paper.text,
            metadata=paper.metadata
        )
        return result
    finally:
        signal.alarm(0)  # 取消超时
```

### 流式结果迭代器

```python
def batch_step1_streaming(
    papers: List[PaperInput],
    max_workers: int = 3,
    enable_caching: bool = True
) -> Iterator[ModuleListResult]:
    """
    流式返回处理结果，适合需要实时响应的场景
    
    Usage:
        for result in batch_step1_streaming(papers):
            if result.success:
                print(f"Processed: {result.paper_id}")
            else:
                print(f"Failed: {result.paper_id}")
    """
    
    from queue import Queue
    result_queue = Queue()
    
    def callback(result: ModuleListResult):
        result_queue.put(result)
    
    # 在后台线程启动批量处理
    import threading
    
    def background_process():
        batch_step1(
            papers=papers,
            max_workers=max_workers,
            callback=callback,
            enable_caching=enable_caching
        )
        result_queue.put(None)  # 结束标记
    
    thread = threading.Thread(target=background_process)
    thread.start()
    
    # 流式返回结果
    while True:
        result = result_queue.get()
        if result is None:
            break
        yield result
    
    thread.join()
```

---

## 错误处理

### 超时处理

```python
from functools import wraps
import time


class TimeoutManager:
    """超时管理器"""
    
    DEFAULT_TIMEOUT = 300  # 5 分钟默认超时
    
    @staticmethod
    def with_timeout(timeout_seconds: int = DEFAULT_TIMEOUT):
        """装饰器：为函数添加超时控制"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # 使用 concurrent.futures 实现超时
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(func, *args, **kwargs)
                    try:
                        return future.result(timeout=timeout_seconds)
                    except concurrent.futures.TimeoutError:
                        raise TimeoutError(f"Function {func.__name__} exceeded {timeout_seconds}s")
            return wrapper
        return decorator


# 动态超时调整（根据论文长度）
def calculate_dynamic_timeout(paper: PaperInput, base_timeout: int = 300) -> int:
    """根据论文长度动态计算超时时间"""
    text_length = len(paper.text)
    
    # 基础超时 + 每 10000 字符增加 1 分钟
    additional_time = (text_length // 10000) * 60
    
    return min(base_timeout + additional_time, 900)  # 最大 15 分钟
```

### 重试策略

```python
import random


class RetryStrategy:
    """可配置的重试策略"""
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
    
    def get_delay(self, attempt: int) -> float:
        """计算第 attempt 次重试的等待时间"""
        # 指数退避
        delay = self.base_delay * (self.exponential_base ** (attempt - 1))
        delay = min(delay, self.max_delay)
        
        # 添加随机抖动，避免惊群效应
        if self.jitter:
            delay = delay * (0.5 + random.random())
        
        return delay
    
    def should_retry(self, error: Exception, attempt: int) -> bool:
        """判断是否应该重试"""
        if attempt >= self.max_retries:
            return False
        
        # 可重试的错误类型
        retryable_errors = (
            TimeoutError,
            ConnectionError,
            RateLimitError,
            ServiceUnavailableError
        )
        
        return isinstance(error, retryable_errors)


# 使用示例
retry_strategy = RetryStrategy(
    max_retries=3,
    base_delay=2.0,
    exponential_base=2.0
)
```

### 部分失败处理

```python
@dataclass
class PartialFailureHandler:
    """部分失败处理器"""
    
    checkpoint_interval: int = 10  # 每处理 N 个保存检查点
    checkpoint_file: Optional[str] = None
    
    def save_checkpoint(self, state: Dict[str, Any]):
        """保存处理状态检查点"""
        import json
        checkpoint = {
            "timestamp": datetime.now().isoformat(),
            "processed_ids": state.get("processed_ids", []),
            "failed_ids": state.get("failed_ids", []),
            "results": [
                {
                    "paper_id": r.paper_id,
                    "success": r.success,
                    "modules": r.modules if r.success else None,
                    "error": r.error_message
                }
                for r in state.get("results", [])
            ]
        }
        
        if self.checkpoint_file:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2)
    
    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """加载检查点恢复状态"""
        import json
        import os
        
        if not self.checkpoint_file or not os.path.exists(self.checkpoint_file):
            return None
        
        with open(self.checkpoint_file, 'r') as f:
            return json.load(f)
    
    def filter_completed_papers(
        self,
        papers: List[PaperInput],
        checkpoint: Optional[Dict[str, Any]]
    ) -> List[PaperInput]:
        """过滤掉已处理的论文"""
        if not checkpoint:
            return papers
        
        processed_ids = set(checkpoint.get("processed_ids", []))
        return [p for p in papers if p.id not in processed_ids]


def batch_step1_with_recovery(
    papers: List[PaperInput],
    checkpoint_file: str = "batch_checkpoint.json",
    **kwargs
) -> BatchResult:
    """支持断点续传的批量处理"""
    
    handler = PartialFailureHandler(checkpoint_file=checkpoint_file)
    
    # 加载检查点
    checkpoint = handler.load_checkpoint()
    remaining_papers = handler.filter_completed_papers(papers, checkpoint)
    
    if checkpoint:
        print(f"Resuming from checkpoint: {len(papers) - len(remaining_papers)} already processed")
    
    # 执行处理
    result = batch_step1(remaining_papers, **kwargs)
    
    # 合并历史结果（如果有）
    if checkpoint:
        historical_results = checkpoint.get("results", [])
        # 将历史结果合并到当前结果
        for hr in historical_results:
            if hr.get("success"):
                result.results.append(ModuleListResult(
                    paper_id=hr["paper_id"],
                    success=True,
                    modules=hr.get("modules", [])
                ))
    
    # 清理检查点（全部成功）
    if not result.failed:
        import os
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
    
    return result
```

---

## 性能优化

### 批量 API 调用

```python
class BatchAPIManager:
    """批量 API 调用管理器"""
    
    def __init__(self, batch_size: int = 5):
        self.batch_size = batch_size
        self.request_buffer = []
        self.buffer_lock = threading.Lock()
    
    def add_request(self, request: Dict[str, Any]) -> Optional[List[Dict]]:
        """
        添加请求到缓冲区，当达到批次大小时返回批量请求
        
        Returns:
            如果达到批次大小，返回批量请求列表；否则返回 None
        """
        with self.buffer_lock:
            self.request_buffer.append(request)
            
            if len(self.request_buffer) >= self.batch_size:
                batch = self.request_buffer[:self.batch_size]
                self.request_buffer = self.request_buffer[self.batch_size:]
                return batch
            
            return None
    
    def flush(self) -> List[Dict]:
        """强制刷新缓冲区，返回剩余的所有请求"""
        with self.buffer_lock:
            batch = self.request_buffer.copy()
            self.request_buffer = []
            return batch


def execute_batch_step1_api(
    papers: List[PaperInput],
    batch_size: int = 5
) -> List[ModuleListResult]:
    """使用批量 API 调用处理论文"""
    
    results = []
    
    # 分批处理
    for i in range(0, len(papers), batch_size):
        batch = papers[i:i + batch_size]
        
        # 构建批量请求
        requests = [
            {
                "paper_id": p.id,
                "text": p.text[:10000],  # 截断以避免超长
                "metadata": p.metadata
            }
            for p in batch
        ]
        
        # 调用批量 API（假设支持）
        try:
            batch_results = call_llm_batch_api(requests)
            results.extend(batch_results)
        except Exception as e:
            # 批量失败，降级为逐个处理
            print(f"Batch API failed, falling back to individual calls: {e}")
            for paper in batch:
                individual_result = execute_step1_core(paper.text, paper.metadata)
                results.append(individual_result)
    
    return results
```

### 缓存机制

```python
import hashlib
import pickle
from datetime import datetime, timedelta


class CacheManager:
    """结果缓存管理器"""
    
    def __init__(self, ttl_hours: int = 24, cache_dir: str = ".cache"):
        self.ttl = timedelta(hours=ttl_hours)
        self.cache_dir = cache_dir
        import os
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_key(self, paper_id: str, content: str) -> str:
        """生成缓存键（基于内容哈希）"""
        content_hash = hashlib.md5(content.encode()).hexdigest()[:16]
        return f"{paper_id}_{content_hash}"
    
    def _get_cache_path(self, cache_key: str) -> str:
        """获取缓存文件路径"""
        return f"{self.cache_dir}/{cache_key}.pkl"
    
    def get(self, paper_id: str, content: str = "") -> Optional[ModuleListResult]:
        """从缓存获取结果"""
        cache_key = self._get_cache_key(paper_id, content)
        cache_path = self._get_cache_path(cache_key)
        
        import os
        if not os.path.exists(cache_path):
            return None
        
        # 检查是否过期
        mtime = datetime.fromtimestamp(os.path.getmtime(cache_path))
        if datetime.now() - mtime > self.ttl:
            os.remove(cache_path)
            return None
        
        # 加载缓存
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None
    
    def set(self, paper_id: str, result: ModuleListResult, content: str = ""):
        """保存结果到缓存"""
        cache_key = self._get_cache_key(paper_id, content)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            print(f"Cache save failed: {e}")
    
    def clear_expired(self):
        """清理过期缓存"""
        import os
        
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.pkl'):
                filepath = os.path.join(self.cache_dir, filename)
                mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
                if datetime.now() - mtime > self.ttl:
                    os.remove(filepath)


# 内存缓存（用于热数据）
class InMemoryCache:
    """基于 LRU 的内存缓存"""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.cache = {}
        self.access_order = []
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key in self.cache:
                # 更新访问顺序
                self.access_order.remove(key)
                self.access_order.append(key)
                return self.cache[key]
            return None
    
    def set(self, key: str, value: Any):
        with self.lock:
            if key in self.cache:
                self.access_order.remove(key)
            elif len(self.cache) >= self.max_size:
                # 淘汰最久未使用
                lru_key = self.access_order.pop(0)
                del self.cache[lru_key]
            
            self.cache[key] = value
            self.access_order.append(key)
```

### 进度追踪

```python
from dataclasses import dataclass, asdict
import json


@dataclass
class ProgressSnapshot:
    """进度快照"""
    timestamp: datetime
    total: int
    completed: int
    successful: int
    failed: int
    in_progress: int
    estimated_remaining_seconds: float
    
    @property
    def progress_percentage(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.completed / self.total) * 100
    
    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)


class ProgressTracker:
    """进度追踪器"""
    
    def __init__(self, total: int, update_interval_seconds: float = 5.0):
        self.total = total
        self.update_interval = update_interval_seconds
        self.start_time = time.time()
        self.last_update_time = 0
        self.stats = {
            "completed": 0,
            "successful": 0,
            "failed": 0,
            "in_progress": 0
        }
        self.lock = threading.Lock()
        self.callbacks = []
    
    def register_callback(self, callback: Callable[[ProgressSnapshot], None]):
        """注册进度更新回调"""
        self.callbacks.append(callback)
    
    def update(self, delta_completed: int = 0, delta_successful: int = 0, 
               delta_failed: int = 0, delta_in_progress: int = 0):
        """更新进度统计"""
        with self.lock:
            self.stats["completed"] += delta_completed
            self.stats["successful"] += delta_successful
            self.stats["failed"] += delta_failed
            self.stats["in_progress"] += delta_in_progress
            
            current_time = time.time()
            if current_time - self.last_update_time >= self.update_interval:
                self._notify()
                self.last_update_time = current_time
    
    def _notify(self):
        """通知所有回调"""
        elapsed = time.time() - self.start_time
        completed = self.stats["completed"]
        
        # 估算剩余时间
        if completed > 0:
            avg_time_per_task = elapsed / completed
            remaining_tasks = self.total - completed
            estimated_remaining = avg_time_per_task * remaining_tasks
        else:
            estimated_remaining = 0
        
        snapshot = ProgressSnapshot(
            timestamp=datetime.now(),
            total=self.total,
            completed=completed,
            successful=self.stats["successful"],
            failed=self.stats["failed"],
            in_progress=self.stats["in_progress"],
            estimated_remaining_seconds=estimated_remaining
        )
        
        for callback in self.callbacks:
            try:
                callback(snapshot)
            except Exception as e:
                print(f"Progress callback error: {e}")


# 使用示例：控制台进度显示
def console_progress_callback(snapshot: ProgressSnapshot):
    """控制台进度回调"""
    bar_length = 30
    filled = int(bar_length * snapshot.progress_percentage / 100)
    bar = "█" * filled + "░" * (bar_length - filled)
    
    eta_mins = snapshot.estimated_remaining_seconds / 60
    
    print(f"\r[{bar}] {snapshot.progress_percentage:.1f}% | "
          f"{snapshot.completed}/{snapshot.total} | "
          f"✓ {snapshot.successful} ✗ {snapshot.failed} | "
          f"ETA: {eta_mins:.1f}m", end="", flush=True)
    
    if snapshot.completed >= snapshot.total:
        print()  # 换行


# WebSocket 实时推送（可选）
class WebSocketProgressReporter:
    """WebSocket 实时进度推送"""
    
    def __init__(self, websocket_connection):
        self.ws = websocket_connection
    
    async def report(self, snapshot: ProgressSnapshot):
        """推送进度到客户端"""
        import asyncio
        
        data = {
            "type": "progress",
            "data": {
                "percentage": snapshot.progress_percentage,
                "completed": snapshot.completed,
                "total": snapshot.total,
                "successful": snapshot.successful,
                "failed": snapshot.failed,
                "eta_seconds": snapshot.estimated_remaining_seconds
            }
        }
        
        try:
            await self.ws.send(json.dumps(data))
        except Exception as e:
            print(f"WebSocket send failed: {e}")
```

---

## 完整使用示例

```python
# 1. 准备输入数据
papers = [
    PaperInput(
        id="paper_001",
        text="论文1的完整文本内容...",
        metadata={"title": "论文标题1", "authors": ["作者A"]},
        priority=1
    ),
    PaperInput(
        id="paper_002", 
        text="论文2的完整文本内容...",
        metadata={"title": "论文标题2", "authors": ["作者B"]},
        priority=2
    ),
    # ... 更多论文
]

# 2. 执行批量处理（带进度追踪）
tracker = ProgressTracker(total=len(papers), update_interval_seconds=5.0)
tracker.register_callback(console_progress_callback)

result = batch_step1(
    papers=papers,
    max_workers=4,
    task_assignment_strategy="length_based",
    enable_caching=True,
    cache_ttl_hours=24
)

# 3. 处理结果
print(f"\n处理完成！")
print(f"成功: {result.stats.successful}")
print(f"失败: {result.stats.failed}")
print(f"耗时: {result.stats.duration_seconds:.2f} 秒")
print(f"吞吐: {result.stats.throughput_per_minute:.2f} 篇/分钟")

# 4. 处理失败的论文
if result.failed:
    print(f"\n失败的论文 ID: {result.failed}")
    # 可选择重新处理或记录到文件
```

---

## 配置参考

```python
# config.py
BATCH_STEP1_CONFIG = {
    # Worker 配置
    "max_workers": 4,
    "worker_timeout_seconds": 300,
    
    # 重试配置
    "max_retries": 3,
    "retry_base_delay": 2.0,
    "retry_max_delay": 30.0,
    
    # 缓存配置
    "enable_caching": True,
    "cache_ttl_hours": 24,
    "cache_dir": ".cache/step1",
    
    # 进度追踪
    "progress_update_interval": 5.0,
    
    # 检查点
    "checkpoint_enabled": True,
    "checkpoint_interval": 10,
    "checkpoint_file": "batch_checkpoint.json",
    
    # 任务分配策略
    "task_assignment_strategy": "length_based"  # round_robin | length_based | priority
}
```
