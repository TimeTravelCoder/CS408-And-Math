# CS408-And-Math

> 计算机考研 408 + 数学交互式学习与可视化工具集

覆盖 **数据结构**、**计算机操作系统**、**计算机组成原理**、**计算机网络** 四门专业课及 **数学**，包含 ~90 个交互式 HTML 演示和 Python 可视化工具。

## 项目结构

```
CS408-And-Math/
├── README.md
├── requirements.txt              # Python 依赖
├── .gitignore
├── launcher.py                   # 统一启动器
│
├── src/                          # Python 源码
│   ├── scheduler/                # OS 调度算法模拟（FCFS/SJF/SRTF/优先级/RR/多级队列/多级反馈队列）
│   ├── number_system/            # 数制转换（IEEE 754 浮点数/原码反码补码移码）
│   └── math_tools/               # 数学工具（矩阵计算器）
│
├── demos/                        # 交互式 HTML 演示
│   ├── index.html                # 📍 总导航页（从这里开始）
│   ├── 数据结构/                  # 10 个（最小生成树/BFS/DFS/哈希/二叉树…）
│   ├── 计算机操作系统/             # 29 个（进程调度/存储管理/虚拟内存/文件系统）
│   ├── 计算机组成原理/             # 8 个（Cache/DRAM/突发传输…）
│   ├── 计算机网络/                # 32 个（TCP/路由/WLAN/DNS/DHCP…）
│   ├── charts/                   # 7 个图表
│   └── math/                     # 3 个数学笔记
│
└── tools/                        # 实用工具脚本
    ├── flatten_dir.py            # 文件夹展平
    ├── pdf_split.py              # PDF 对半分割
    ├── file_creator.py           # 多格式文件生成器
    ├── dir_tree.py               # 目录树打印
    └── conda_manager.py          # Conda 环境管理
```

## 快速开始

### 1. HTML 演示（无需安装）

直接用浏览器打开 `demos/index.html`，从导航页进入任意演示。

### 2. Python 工具

```bash
# 安装依赖
pip install -r requirements.txt

# 启动调度器模拟（最核心工具）
python -m src.scheduler.cli

# 或使用统一启动器
python launcher.py
```

## 核心功能

### OS 调度算法可视化

支持 8 种调度算法，含 Plotly 甘特图和性能指标：

| 算法 | 类型 |
|------|------|
| FCFS | 先来先服务 |
| SJF | 短作业优先（非抢占） |
| SRTF | 最短剩余时间优先（抢占） |
| Priority | 优先级调度（抢占/非抢占） |
| RR | 时间片轮转 |
| MLQ | 多级队列 |
| MFQS | 多级反馈队列（含老化机制） |

```python
from src.scheduler import FCFS, Process, run_interactive

processes = [
    Process(1, arrival_time=0, run_time=8, priority=1),
    Process(2, arrival_time=1, run_time=4, priority=2),
]
scheduler = FCFS(processes)
scheduler.run()
```

### 其他工具

- **数制转换**：IEEE 754 浮点数 ↔ 二进制/十六进制、原码/反码/补码/移码
- **矩阵计算器**：行列式/逆/特征值/特征向量/LU/QR/谱分解
- **PDF 分割**、**文件生成器**、**目录树** 等辅助工具

## 依赖

```
sympy>=1.12          # 符号数学
plotly>=5.18         # 甘特图可视化
pandas>=2.0           # 数据处理
numpy>=1.24           # 数值计算
pypdf>=4.0            # PDF 操作
python-docx>=1.0      # Word 文件
python-pptx>=0.6      # PPT 文件
```

## License

仅供学习交流使用。
