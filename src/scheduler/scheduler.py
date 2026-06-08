"""调度器基类 + 全部算法实现

包含：FCFS、非抢占SJF、抢占SJF(SRTF)、非抢占优先级、抢占优先级、
      时间片轮转(RR)、多级队列(MLQ)、多级反馈队列(MFQS)
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
from .models import Process


class Scheduler(ABC):
    """调度器基类"""

    def __init__(self, processes: List[Process], name: str,
                 priority_higher_is_better: bool = False):
        self.processes = processes
        self.ready_queue: List[Process] = []
        self.current_time = 0
        self.execution_sequence: List[Dict] = []
        self.completed_processes: List[Process] = []
        self.current_process: Optional[Process] = None
        self.name = name
        self.config: Dict = {}
        self.priority_higher_is_better = priority_higher_is_better

    def update_ready_queue(self) -> None:
        """将已到达、未完成且不在就绪队列中的进程加入就绪队列"""
        for p in self.processes:
            if (p.arrival_time <= self.current_time
                    and p.remaining_time > 0
                    and p not in self.ready_queue
                    and p not in self.completed_processes
                    and p != self.current_process):
                self.ready_queue.append(p)

    def execute_process(self, process: Process, duration: int) -> bool:
        """执行进程一段时长，返回是否完成"""
        if process.start_time is None:
            process.start_time = self.current_time

        actual = min(duration, process.remaining_time)
        process.remaining_time -= actual
        end_time = self.current_time + actual
        completed = process.remaining_time <= 0

        self.execution_sequence.append({
            "pid": process.pid,
            "start_time": self.current_time,
            "end_time": end_time,
            "status": "完成" if completed else "执行"
        })
        self.current_time = end_time

        if completed:
            process.completion_time = end_time
            process.calculate_metrics()
            self.completed_processes.append(process)
            if process in self.ready_queue:
                self.ready_queue.remove(process)
            return True
        return False

    def is_preemptive(self) -> bool:
        return False

    @abstractmethod
    def select_process(self) -> Optional[Process]:
        """选择下一个要运行的进程"""
        ...

    def get_execution_time(self, process: Process) -> int:
        return process.remaining_time

    def handle_uncompleted_process(self, process: Process,
                                   executed_duration: int) -> None:
        if process.remaining_time > 0 and process not in self.ready_queue:
            self.ready_queue.append(process)

    def _priority_sort_key(self, p: Process) -> Tuple:
        """优先级排序键：effective_priority 越小越优先"""
        if p.priority is None:
            eff = float('inf')
        else:
            eff = -p.priority if self.priority_higher_is_better else p.priority
        return (eff, p.arrival_time, p.pid)

    def run(self) -> Dict:
        """执行调度模拟"""
        while len(self.completed_processes) < len(self.processes):
            self.update_ready_queue()

            # 空闲时快进到下一个到达时间
            if not self.ready_queue and not self.current_process:
                pending = [p for p in self.processes
                          if p not in self.completed_processes
                          and p.arrival_time > self.current_time]
                if pending:
                    self.current_time = min(p.arrival_time for p in pending)
                    continue
                elif not any(p.remaining_time > 0 for p in self.processes):
                    break

            next_proc = self.select_process()

            # 抢占处理
            if (self.is_preemptive() and self.current_process
                    and self.current_process.remaining_time > 0):
                if next_proc and next_proc != self.current_process:
                    self.execution_sequence.append({
                        "pid": self.current_process.pid,
                        "start_time": self.current_time,
                        "end_time": self.current_time,
                        "status": "抢占"
                    })
                    self.handle_uncompleted_process(self.current_process, 0)
                    self.current_process = None

            if self.current_process is None:
                self.current_process = next_proc
                if self.current_process:
                    self._remove_from_queues(self.current_process)

            elif (not self.is_preemptive()
                  and self.current_process.remaining_time <= 0):
                self.current_process = next_proc
                if self.current_process:
                    self._remove_from_queues(self.current_process)

            if self.current_process:
                dur = self.get_execution_time(self.current_process)
                completed = self.execute_process(self.current_process, dur)
                if completed:
                    self.current_process = None
                elif self.is_preemptive():
                    self.handle_uncompleted_process(self.current_process, dur)
                    self.current_process = None
            else:
                if not self.ready_queue:
                    future = [p.arrival_time for p in self.processes
                             if p.arrival_time > self.current_time
                             and p not in self.completed_processes]
                    if future:
                        self.current_time = min(future)
                    elif len(self.completed_processes) >= len(self.processes):
                        break
                    else:
                        self.current_time += 1

        return self._calculate_system_metrics()

    def _remove_from_queues(self, process: Process) -> None:
        """从就绪队列中移除进程（子类可覆盖）"""
        if process in self.ready_queue:
            self.ready_queue.remove(process)

    def _calculate_system_metrics(self) -> Dict:
        for p in self.completed_processes:
            if p.turnaround_time is None:
                p.calculate_metrics()

        if not self.completed_processes:
            return {"cpu_utilization": 0, "throughput": 0,
                    "avg_turnaround_time": 0, "avg_waiting_time": 0,
                    "avg_response_time": 0}

        total_time = max(
            (p.completion_time for p in self.completed_processes
             if p.completion_time is not None), default=0)
        if total_time == 0:
            return {"cpu_utilization": 0, "throughput": 0,
                    "avg_turnaround_time": 0, "avg_waiting_time": 0,
                    "avg_response_time": 0}

        busy = sum(ex["end_time"] - ex["start_time"]
                   for ex in self.execution_sequence
                   if ex["pid"] != "空闲" and ex["status"] != "抢占")

        n = len(self.completed_processes)
        return {
            "cpu_utilization": busy / total_time,
            "throughput": n / total_time,
            "avg_turnaround_time":
                sum(p.turnaround_time for p in self.completed_processes
                    if p.turnaround_time is not None) / n,
            "avg_waiting_time":
                sum(p.waiting_time for p in self.completed_processes
                    if p.waiting_time is not None) / n,
            "avg_response_time":
                sum(p.response_time for p in self.completed_processes
                    if p.response_time is not None) / n,
        }


# ── 各算法实现 ───────────────────────────────────────────

class FCFS(Scheduler):
    """先来先服务"""
    def __init__(self, processes, priority_higher_is_better=False):
        super().__init__(processes, "FCFS (先来先服务)", priority_higher_is_better)

    def select_process(self):
        if not self.ready_queue:
            return None
        return sorted(self.ready_queue, key=lambda p: (p.arrival_time, p.pid))[0]


class SJF(Scheduler):
    """非抢占式短作业优先"""
    def __init__(self, processes, priority_higher_is_better=False):
        super().__init__(processes, "SJF (短作业优先)", priority_higher_is_better)

    def select_process(self):
        if self.current_process and self.current_process.remaining_time > 0:
            return self.current_process
        if not self.ready_queue:
            return None
        return sorted(self.ready_queue,
                      key=lambda p: (p.run_time, p.arrival_time, p.pid))[0]


class SRTF(Scheduler):
    """抢占式短作业优先（最短剩余时间优先）"""
    def __init__(self, processes, priority_higher_is_better=False):
        super().__init__(processes, "SRTF (抢占式短作业优先)", priority_higher_is_better)

    def is_preemptive(self):
        return True

    def select_process(self):
        candidates = self.ready_queue.copy()
        if self.current_process and self.current_process.remaining_time > 0:
            if self.current_process not in candidates:
                candidates.append(self.current_process)
        if not candidates:
            return None
        return sorted(candidates,
                      key=lambda p: (p.remaining_time, p.arrival_time, p.pid))[0]

    def get_execution_time(self, process):
        to_finish = process.remaining_time
        next_arrival = float('inf')
        for other in self.processes:
            if (other.arrival_time > self.current_time
                    and other not in self.completed_processes
                    and other != process):
                next_arrival = min(next_arrival,
                                  other.arrival_time - self.current_time)
        return min(to_finish, next_arrival, 1)


class PriorityNP(Scheduler):
    """非抢占式优先级调度"""
    def __init__(self, processes, priority_higher_is_better=False):
        super().__init__(processes, "非抢占式优先级调度", priority_higher_is_better)
        self._check()
        self.config["优先级逻辑"] = ("数值越大越优先" if priority_higher_is_better
                                    else "数值越小越优先")

    def _check(self):
        if any(p.priority is None for p in self.processes):
            raise ValueError("所有进程必须定义优先级")

    def select_process(self):
        if self.current_process and self.current_process.remaining_time > 0:
            return self.current_process
        if not self.ready_queue:
            return None
        return sorted(self.ready_queue, key=self._priority_sort_key)[0]


class PriorityP(Scheduler):
    """抢占式优先级调度"""
    def __init__(self, processes, priority_higher_is_better=False):
        super().__init__(processes, "抢占式优先级调度", priority_higher_is_better)
        self._check()
        self.config["优先级逻辑"] = ("数值越大越优先" if priority_higher_is_better
                                    else "数值越小越优先")

    def _check(self):
        if any(p.priority is None for p in self.processes):
            raise ValueError("所有进程必须定义优先级")

    def is_preemptive(self):
        return True

    def select_process(self):
        candidates = self.ready_queue.copy()
        if self.current_process and self.current_process.remaining_time > 0:
            if self.current_process not in candidates:
                candidates.append(self.current_process)
        if not candidates:
            return None
        return sorted(candidates, key=self._priority_sort_key)[0]

    def get_execution_time(self, process):
        to_finish = process.remaining_time
        next_high = float('inf')
        for other in self.processes:
            if (other.arrival_time > self.current_time
                    and other not in self.completed_processes
                    and other != process
                    and other.priority is not None):
                other_eff = (-other.priority if self.priority_higher_is_better
                             else other.priority)
                proc_eff = (-process.priority if self.priority_higher_is_better
                            else process.priority)
                if other_eff < proc_eff:
                    next_high = min(next_high,
                                   other.arrival_time - self.current_time)
        return min(to_finish, next_high, 1)


class RoundRobin(Scheduler):
    """时间片轮转"""
    def __init__(self, processes, time_quantum: int,
                 priority_higher_is_better=False):
        super().__init__(processes, "RR (时间片轮转)", priority_higher_is_better)
        self.time_quantum = time_quantum
        self.config["时间片"] = time_quantum

    def is_preemptive(self):
        return True

    def select_process(self):
        if not self.ready_queue:
            return None
        return sorted(self.ready_queue, key=lambda p: p.arrival_time)[0]

    def get_execution_time(self, process):
        return min(self.time_quantum, process.remaining_time)

    def handle_uncompleted_process(self, process, executed_duration):
        if process.remaining_time > 0:
            if process in self.ready_queue:
                self.ready_queue.remove(process)
            self.ready_queue.append(process)


class MultiLevelQueue(Scheduler):
    """多级队列调度"""
    def __init__(self, processes, queue_configs: List[Dict],
                 priority_higher_is_better=False):
        super().__init__(processes, "多级队列调度", priority_higher_is_better)
        self.queue_configs = queue_configs
        self.queues: List[List[Process]] = [[] for _ in range(len(queue_configs))]

        for p in self.processes:
            if p.priority is None:
                raise ValueError(f"进程 {p.pid} 必须有优先级用于多级队列")
            p.current_queue = min(p.priority, len(self.queues) - 1)

        cfg_strs = []
        for i, cfg in enumerate(queue_configs):
            alg = cfg["algorithm"]
            s = f"队列{i}:{alg}"
            if alg == "RR":
                s += f"(TQ:{cfg['time_quantum']})"
            cfg_strs.append(s)
        self.config["队列配置"] = ", ".join(cfg_strs)

    def update_ready_queue(self):
        for p in self.processes:
            if (p.arrival_time <= self.current_time
                    and p.remaining_time > 0
                    and p not in self.completed_processes
                    and p != self.current_process):
                if not any(p in q for q in self.queues):
                    self.queues[p.current_queue].append(p)
        self.ready_queue = [p for q in self.queues for p in q]

    def select_process(self):
        for i, queue in enumerate(self.queues):
            if not queue:
                continue
            alg = self.queue_configs[i]["algorithm"]
            if alg == "FCFS":
                return sorted(queue, key=lambda p: (p.arrival_time, p.pid))[0]
            elif alg == "SJF":
                return sorted(queue, key=lambda p: (p.run_time, p.arrival_time, p.pid))[0]
            elif alg == "Priority":
                return sorted(queue, key=self._priority_sort_key)[0]
            elif alg == "RR":
                return sorted(queue, key=lambda p: p.arrival_time)[0]
        return None

    def is_preemptive(self):
        if self.current_process:
            for i in range(self.current_process.current_queue):
                if self.queues[i]:
                    return True
            if self.queue_configs[self.current_process.current_queue]["algorithm"] == "RR":
                return True
        return False

    def get_execution_time(self, process):
        q_idx = process.current_queue
        alg = self.queue_configs[q_idx]["algorithm"]

        next_high = float('inf')
        for i in range(q_idx):
            for other in self.processes:
                if (other.current_queue == i
                        and other.arrival_time > self.current_time
                        and other not in self.completed_processes):
                    next_high = min(next_high,
                                   other.arrival_time - self.current_time)

        limit = process.remaining_time
        if alg == "RR":
            limit = min(self.queue_configs[q_idx]["time_quantum"],
                       process.remaining_time)

        return min(limit, next_high,
                   1 if alg in ("SJF", "Priority") else process.remaining_time)

    def handle_uncompleted_process(self, process, executed_duration):
        if process.remaining_time > 0:
            q_idx = process.current_queue
            if process not in self.queues[q_idx]:
                self.queues[q_idx].append(process)
            elif self.queue_configs[q_idx]["algorithm"] == "RR":
                self.queues[q_idx].remove(process)
                self.queues[q_idx].append(process)

    def _remove_from_queues(self, process):
        for q in self.queues:
            if process in q:
                q.remove(process)
                return


class MultiLevelFeedbackQueue(Scheduler):
    """多级反馈队列调度"""
    def __init__(self, processes, queue_count: int, time_quantums: List[int],
                 priority_higher_is_better=False, enable_aging=False,
                 aging_threshold=10):
        super().__init__(processes, "多级反馈队列调度", priority_higher_is_better)
        self.queue_count = queue_count
        self.time_quantums = time_quantums
        self.queues: List[List[Process]] = [[] for _ in range(queue_count)]
        self.enable_aging = enable_aging
        self.aging_threshold = aging_threshold
        self._wait_counters = {p.pid: 0 for p in self.processes}

        for p in self.processes:
            p.current_queue = 0

        tq_strs = ", ".join(str(t) for t in time_quantums[:-1])
        self.config["队列数"] = queue_count
        self.config["各队列时间片"] = f"{tq_strs}, 最后一队:FCFS"
        if enable_aging:
            self.config["老化阈值"] = aging_threshold

    def update_ready_queue(self):
        for p in self.processes:
            if (p.arrival_time <= self.current_time
                    and p.remaining_time > 0
                    and p not in self.completed_processes
                    and p != self.current_process):
                if not any(p in q for q in self.queues):
                    self.queues[p.current_queue].append(p)
        if self.enable_aging:
            self._apply_aging()
        self.ready_queue = [p for q in self.queues for p in q]

    def _apply_aging(self):
        for q_idx in range(1, self.queue_count):
            for p in self.queues[q_idx][:]:
                if p != self.current_process:
                    self._wait_counters[p.pid] += 1
                if self._wait_counters[p.pid] >= self.aging_threshold:
                    self.queues[q_idx].remove(p)
                    p.current_queue = max(0, q_idx - 1)
                    self.queues[p.current_queue].append(p)
                    self._wait_counters[p.pid] = 0

    def select_process(self):
        for i, queue in enumerate(self.queues):
            if not queue:
                continue
            return sorted(queue, key=lambda p: p.arrival_time)[0]
        return None

    def is_preemptive(self):
        return True

    def get_execution_time(self, process):
        q_idx = process.current_queue
        if q_idx == self.queue_count - 1:
            return process.remaining_time
        return min(self.time_quantums[q_idx], process.remaining_time)

    def handle_uncompleted_process(self, process, executed_duration):
        if process.remaining_time > 0:
            self._wait_counters[process.pid] = 0
            q_idx = process.current_queue

            demote = (q_idx < self.queue_count - 1
                     and executed_duration >= self.time_quantums[q_idx])
            if demote:
                process.current_queue = min(q_idx + 1, self.queue_count - 1)
                self.queues[process.current_queue].append(process)
            else:
                self.queues[q_idx].append(process)

    def _remove_from_queues(self, process):
        for q in self.queues:
            if process in q:
                q.remove(process)
                return


# ── 算法注册表 ───────────────────────────────────────────

ALGORITHMS = {
    "fcfs": FCFS,
    "sjf": SJF,
    "srtf": SRTF,
    "priority_np": PriorityNP,
    "priority_p": PriorityP,
    "rr": RoundRobin,
    "mlq": MultiLevelQueue,
    "mfqs": MultiLevelFeedbackQueue,
}
