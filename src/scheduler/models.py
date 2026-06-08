"""进程模型定义"""

from typing import Optional


class Process:
    """进程类，包含进程的各项属性"""

    def __init__(self, pid: int, arrival_time: int, run_time: int,
                 priority: Optional[int] = None):
        self.pid = pid
        # 静态属性（创建后不变）
        self.arrival_time = arrival_time
        self.run_time = run_time  # 总需要运行时间
        self.priority = priority

        # 动态属性（每次调度运行前重置）
        self.remaining_time = run_time
        self.start_time: Optional[int] = None
        self.completion_time: Optional[int] = None
        self.waiting_time: Optional[float] = None
        self.turnaround_time: Optional[float] = None
        self.response_time: Optional[float] = None
        self.current_queue: int = 0  # 多级队列用

    def __str__(self):
        prio = self.priority if self.priority is not None else 'N/A'
        return (f"进程 {self.pid} "
                f"(到达:{self.arrival_time}, 运行:{self.run_time}, 优先级:{prio})")

    def calculate_metrics(self):
        """计算进程的性能指标"""
        if self.completion_time is not None and self.arrival_time is not None:
            self.turnaround_time = self.completion_time - self.arrival_time
            if self.run_time is not None:
                self.waiting_time = self.turnaround_time - self.run_time
            if self.start_time is not None:
                self.response_time = self.start_time - self.arrival_time

    def clone(self) -> "Process":
        """深拷贝进程（用于多次调度运行）"""
        return Process(self.pid, self.arrival_time, self.run_time, self.priority)
