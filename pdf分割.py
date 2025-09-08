import pypdf
import os

def split_pdf_by_page_count(input_path: str):
    """
    将一个PDF文件按总页数对半分割成两个文件。

    :param input_path: 输入的 PDF 文件路径
    """
    # 1. 检查文件是否存在
    if not os.path.exists(input_path):
        print(f"错误：输入文件 '{input_path}' 不存在。")
        return

    # 2. 生成输出文件名
    # 例如，如果输入是 "mydoc.pdf", 输出将是 "mydoc-part1.pdf" 和 "mydoc-part2.pdf"
    base_name, extension = os.path.splitext(input_path)
    output_path1 = f"{base_name}-part1{extension}"
    output_path2 = f"{base_name}-part2{extension}"

    try:
        # 3. 读取原始PDF
        reader = pypdf.PdfReader(input_path)
        total_pages = len(reader.pages)

        # 如果页数太少，无法分割
        if total_pages < 2:
            print(f"文件 '{input_path}' 只有 {total_pages} 页，无法进行分割。")
            return

        # 4. 计算分割点
        split_point = total_pages // 2
        print(f"文件 '{input_path}' 共有 {total_pages} 页。")
        print(f"将从第 {split_point} 页后分割。")
        print(f"第一部分将包含 {split_point} 页 (1 到 {split_point})。")
        print(f"第二部分将包含 {total_pages - split_point} 页 ({split_point + 1} 到 {total_pages})。")

        # 5. 创建并写入第一个PDF文件 (part 1)
        writer1 = pypdf.PdfWriter()
        # 添加前半部分的页面
        for page_num in range(split_point):
            writer1.add_page(reader.pages[page_num])
        
        with open(output_path1, "wb") as file1:
            writer1.write(file1)
        print(f"第一部分已成功保存为 '{output_path1}'")

        # 6. 创建并写入第二个PDF文件 (part 2)
        writer2 = pypdf.PdfWriter()
        # 添加后半部分的页面
        for page_num in range(split_point, total_pages):
            writer2.add_page(reader.pages[page_num])

        with open(output_path2, "wb") as file2:
            writer2.write(file2)
        print(f"第二部分已成功保存为 '{output_path2}'")

    except Exception as e:
        print(f"处理过程中发生错误: {e}")

# --- 如何使用 ---
if __name__ == "__main__":
    # 设置你的输入文件名
    input_pdf = "C:\\Users\\jiami\\Downloads\\Documents\\26徐涛《核心考案》.pdf"  # 将此替换为你的PDF文件名

    # 调用函数执行分割
    split_pdf_by_page_count(input_pdf)