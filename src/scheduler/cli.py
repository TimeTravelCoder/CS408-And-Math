"""调度器命令行交互界面"""

import random
import platform
import os
from typing import List, Optional

from .models import Process
from .scheduler import (
    FCFS, SJF, SRTF, PriorityNP, PriorityP,
    RoundRobin, MultiLevelQueue, MultiLevelFeedbackQueue
)
from .visualize import print_results, create_gantt_chart


def clear_screen():
    os.system('cls' if platform.system() == "Windows" else 'clear')


def _read_int(prompt: str, min_val: int = 0,
              max_val: float = float('inf')) -> int:
    while True:
        try:
            v = int(input(prompt))
            if min_val <= v <= max_val:
                return v
            print(f"请输入 {min_val} 到 {max_val} 之间的整数。")
        except ValueError:
            print("无效输入，请输入整数。")


def _read_bool(prompt: str) -> bool:
    while True:
        r = input(prompt + " (y/n): ").strip().lower()
        if r in ('y', 'yes', '是', '1'):
            return True
        if r in ('n', 'no', '否', '0'):
            return False


def manual_input_processes(need_priority: bool,
                           higher_is_better: bool) -> List[Process]:
    processes = []
    count = _read_int("请输入进程数量: ", 1, 20)
    for i in range(count):
        print(f"\n--- 进程 {i+1} ---")
        pid = i + 1
        at = _read_int("  到达时间: ", 0)
        bt = _read_int("  运行时间: ", 1)
        prio = None
        if need_priority:
            hint = "(数值越大优先级越高)" if higher_is_better else "(数值越小优先级越高)"
            prio = _read_int(f"  优先级 {hint}: ", 0)
        processes.append(Process(pid, at, bt, prio))
    return processes


def random_generate_processes(need_priority: bool) -> List[Process]:
    count = _read_int("生成进程数量: ", 1, 20)
    max_at = _read_int("最大到达时间: ", 0)
    max_bt = _read_int("最大运行时间: ", 1)
    processes = []
    for i in range(count):
        pid = i + 1
        at = random.randint(0, max_at)
        bt = random.randint(1, max_bt)
        prio = random.randint(0, 9) if need_priority else None
        processes.append(Process(pid, at, bt, prio))
    print("\n随机生成的进程:")
    for p in processes:
        print(f"  {p}")
    input("按回车键继续...")
    return processes


def run_interactive():
    """交互式运行调度模拟"""
    clear_screen()
    print("====== 操作系统调度算法可视化工具 ======\n")

    print("优先级判定逻辑:")
    print("  1. 数值越小，优先级越高")
    print("  2. 数值越大，优先级越高")
    choice = _read_int("输入选择 (1/2): ", 1, 2)
    higher_is_better = (choice == 2)

    print("\n进程数据输入方式:")
    print("  1. 手动输入")
    print("  2. 随机生成")
    mode = _read_int("输入选择 (1/2): ", 1, 2)

    will_use_priority = _read_bool("是否计划使用需要优先级的算法?")
    if mode == 1:
        originals = manual_input_processes(will_use_priority, higher_is_better)
    else:
        originals = random_generate_processes(will_use_priority)

    while True:
        processes = [p.clone() for p in originals]

        clear_screen()
        print("\n当前进程集:")
        for p in processes:
            print(f"  {p}")

        print("\n请选择调度算法:")
        print("  1. FCFS          2. SJF (非抢占)")
        print("  3. SRTF (抢占)   4. 优先级 (非抢占)")
        print("  5. 优先级 (抢占)  6. RR (时间片轮转)")
        print("  7. 多级队列       8. 多级反馈队列")
        algo = _read_int("输入选择 (1-8): ", 1, 8)

        scheduler = None
        if algo == 1:
            scheduler = FCFS(processes, higher_is_better)
        elif algo == 2:
            scheduler = SJF(processes, higher_is_better)
        elif algo == 3:
            scheduler = SRTF(processes, higher_is_better)
        elif algo == 4:
            scheduler = PriorityNP(processes, higher_is_better)
        elif algo == 5:
            scheduler = PriorityP(processes, higher_is_better)
        elif algo == 6:
            tq = _read_int("输入时间片大小: ", 1)
            scheduler = RoundRobin(processes, tq, higher_is_better)
        elif algo == 7:
            nq = _read_int("队列数量 (2-5): ", 2, 5)
            configs = []
            for i in range(nq):
                print(f"\n队列 {i}: 1.FCFS 2.SJF 3.Priority 4.RR")
                qa = _read_int(f"  选择算法 (1-4): ", 1, 4)
                cfg = {}
                if qa == 1:
                    cfg["algorithm"] = "FCFS"
                elif qa == 2:
                    cfg["algorithm"] = "SJF"
                elif qa == 3:
                    cfg["algorithm"] = "Priority"
                else:
                    cfg["algorithm"] = "RR"
                    cfg["time_quantum"] = _read_int("  时间片: ", 1)
                configs.append(cfg)
            scheduler = MultiLevelQueue(processes, configs, higher_is_better)
        elif algo == 8:
            nq = _read_int("队列数量 (2-5): ", 2, 5)
            tqs = []
            for i in range(nq - 1):
                tqs.append(_read_int(f"  队列 {i} 时间片: ", 1))
            tqs.append(0)
            aging = _read_bool("启用老化机制?")
            aging_t = 10
            if aging:
                aging_t = _read_int("老化阈值: ", 1)
            scheduler = MultiLevelFeedbackQueue(
                processes, nq, tqs, higher_is_better, aging, aging_t)

        if scheduler:
            scheduler.run()
            print_results(scheduler)
            if _read_bool("\n显示甘特图?"):
                fig = create_gantt_chart(scheduler)
                fig.show()

        if not _read_bool("\n用当前进程集运行其他算法?"):
            break

    print("\n模拟结束。")
