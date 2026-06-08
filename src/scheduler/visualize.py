"""调度结果可视化 —— Plotly 甘特图 + 指标表格"""

from typing import List, Dict
import plotly.graph_objects as go
from plotly.subplots import make_subplots


COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]


def create_gantt_chart(scheduler) -> go.Figure:
    """为调度器运行结果创建甘特图"""
    seq = scheduler.execution_sequence
    process_colors: Dict = {}
    all_pids = sorted(set(
        ex['pid'] for ex in seq if ex['pid'] != "空闲"))

    for i, pid in enumerate(all_pids):
        process_colors[pid] = COLORS[i % len(COLORS)]
    process_colors["空闲"] = '#D3D3D3'

    fig = make_subplots(
        rows=2, cols=1, row_heights=[0.7, 0.3],
        specs=[[{"type": "scatter"}], [{"type": "table"}]],
        subplot_titles=(f"{scheduler.name} 调度甘特图", "系统性能指标")
    )

    for ex in seq:
        if ex["status"] == "抢占":
            continue
        y_val = f"进程 {ex['pid']}" if ex['pid'] != "空闲" else "空闲"
        fig.add_trace(
            go.Bar(
                x=[ex["end_time"] - ex["start_time"]],
                y=[y_val],
                orientation='h', base=ex["start_time"],
                marker=dict(color=process_colors.get(ex["pid"], '#000')),
                text=f"{ex['pid']}: {ex['start_time']}-{ex['end_time']}",
                hoverinfo="text", showlegend=False
            ), row=1, col=1
        )

    m = scheduler._calculate_system_metrics()
    fig.add_trace(
        go.Table(
            header=dict(values=['指标', '值'], font=dict(size=14),
                       align='center'),
            cells=dict(values=[
                ['CPU 利用率', '吞吐量', '平均周转时间', '平均等待时间', '平均响应时间'],
                [f"{m['cpu_utilization']*100:.2f}%",
                 f"{m['throughput']:.2f} 进程/时间单位",
                 f"{m['avg_turnaround_time']:.2f} 时间单位",
                 f"{m['avg_waiting_time']:.2f} 时间单位",
                 f"{m['avg_response_time']:.2f} 时间单位"]
            ], font=dict(size=12), align='center')
        ), row=2, col=1
    )

    fig.update_layout(
        title_text=f"{scheduler.name} 调度算法可视化",
        title_font_size=20,
        barmode='stack',
        yaxis_title='进程',
        xaxis_title='时间',
        xaxis_dtick=1,
        xaxis_showgrid=True,
        height=800,
        font_family="SimHei, Microsoft YaHei, Arial"
    )
    return fig


def print_results(scheduler) -> None:
    """在控制台打印调度结果"""
    print(f"\n{'='*60}")
    print(f"  {scheduler.name} 调度算法")
    print(f"{'='*60}")

    if scheduler.config:
        print("\n配置参数:")
        for k, v in scheduler.config.items():
            print(f"  {k}: {v}")

    # 进程信息
    print("\n初始进程信息:")
    for p in sorted(scheduler.processes, key=lambda x: x.pid):
        prio = p.priority if p.priority is not None else 'N/A'
        print(f"  进程 {p.pid} (到达:{p.arrival_time}, "
              f"总运行:{p.run_time}, 优先级:{prio})")

    # 执行序列
    print("\n执行序列:")
    for ex in scheduler.execution_sequence:
        print(f"  时间 [{ex['start_time']}-{ex['end_time']}]: "
              f"进程 {ex['pid']} ({ex['status']})")

    # 各进程指标
    print("\n进程性能指标:")
    h = ["PID", "到达", "总运行", "完成", "周转", "等待", "响应"]
    print("  " + " | ".join(f"{x:^6}" for x in h))
    print("  " + "-" * (9 * len(h)))

    for p in sorted(scheduler.completed_processes, key=lambda x: x.pid):
        row = [str(p.pid), str(p.arrival_time), str(p.run_time),
               str(p.completion_time),
               f"{p.turnaround_time:.1f}" if p.turnaround_time else "N/A",
               f"{p.waiting_time:.1f}" if p.waiting_time else "N/A",
               f"{p.response_time:.1f}" if p.response_time else "N/A"]
        print("  " + " | ".join(f"{x:^6}" for x in row))

    # 系统指标
    m = scheduler._calculate_system_metrics()
    print("\n系统性能指标:")
    print(f"  CPU 利用率:     {m['cpu_utilization']*100:.2f}%")
    print(f"  吞吐量:         {m['throughput']:.2f} 进程/时间单位")
    print(f"  平均周转时间:   {m['avg_turnaround_time']:.2f}")
    print(f"  平均等待时间:   {m['avg_waiting_time']:.2f}")
    print(f"  平均响应时间:   {m['avg_response_time']:.2f}")
