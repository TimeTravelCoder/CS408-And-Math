#!/usr/bin/env python3
"""CS408-And-Math 统一启动器"""

import os
import sys
import platform


def clear():
    os.system('cls' if platform.system() == 'Windows' else 'clear')


def menu():
    while True:
        clear()
        print("=" * 55)
        print("   CS408-And-Math 工具集")
        print("=" * 55)
        print()
        print("  📊 OS 调度算法")
        print("    1. 交互式调度模拟器（含甘特图）")
        print()
        print("  🔢 数制转换")
        print("    2. IEEE 754 浮点数转换")
        print("    3. 十进制 → 原码/反码/补码/移码")
        print("    4. 浮点数转化过程")
        print()
        print("  📐 数学工具")
        print("    5. 矩阵计算器（行列式/特征值/分解）")
        print()
        print("  🌐 HTML 演示")
        print("    6. 打开 HTML 导航页")
        print()
        print("  🛠 工具脚本")
        print("    7. PDF 对半分割")
        print("    8. 目录树打印")
        print("    9. 文件夹展平")
        print()
        print("  0. 退出")
        print()
        try:
            choice = int(input("请选择 (0-9): "))
        except ValueError:
            continue

        if choice == 0:
            print("再见！")
            break
        elif choice == 1:
            from src.scheduler.cli import run_interactive
            run_interactive()
        elif choice == 2:
            run_script("src/number_system/IEEE754浮点数.py")
        elif choice == 3:
            run_script("src/number_system/十进制转二进制.py")
        elif choice == 4:
            run_script("src/number_system/浮点数转化过程.py")
        elif choice == 5:
            run_script("src/math_tools/matrix_calc.py")
        elif choice == 6:
            import webbrowser
            path = os.path.join(os.path.dirname(__file__), "demos", "index.html")
            webbrowser.open("file://" + os.path.abspath(path))
            input("\n已在浏览器中打开导航页。按回车返回...")
        elif choice == 7:
            run_script("tools/pdf_split.py")
        elif choice == 8:
            run_script("tools/dir_tree.py")
        elif choice == 9:
            run_script("tools/flatten_dir.py")
        else:
            input("无效选择，按回车继续...")


def run_script(relpath: str):
    path = os.path.join(os.path.dirname(__file__), relpath)
    print(f"\n运行: {relpath}\n")
    try:
        exec(open(path, encoding='utf-8').read())
    except Exception as e:
        print(f"运行出错: {e}")
    input("\n按回车返回菜单...")


if __name__ == "__main__":
    menu()
