import sympy
from sympy import Matrix, pprint, symbols, factor

def get_matrix_from_user():
    """
    从用户处获取输入，创建一个 SymPy 矩阵对象。
    该函数会分别提示用户输入行数和列数。
    """
    while True:
        try:
            m_str = input("请输入矩阵的行数 (m): ")
            if not m_str: continue
            m = int(m_str)

            n_str = input("请输入矩阵的列数 (n): ")
            if not n_str: continue
            n = int(n_str)
            
            if m <= 0 or n <= 0:
                print("错误：行数和列数必须是正整数。请重新输入。")
                continue
            break
        except ValueError:
            print("错误：请输入有效的整数作为行数和列数。")

    matrix_data = []
    print(f"请输入 {m}x{n} 矩阵的元素，每行输入后按回车键，元素间用空格隔开:")
    for i in range(m):
        while True:
            row_str = input(f"第 {i+1} 行: ")
            try:
                # 使用 sympy.S 将输入转换为 SymPy 的符号对象，以支持分数、符号等
                row = [sympy.S(item) for item in row_str.split()]
                if len(row) != n:
                    print(f"错误：您需要输入 {n} 个元素，但您输入了 {len(row)} 个。请重新输入这一行。")
                    continue
                matrix_data.append(row)
                break
            except (ValueError, TypeError) as e:
                print(f"错误：输入包含无效字符，请确保所有元素都是数字或合法的数学表达式。错误信息: {e}")

    return Matrix(matrix_data)

def calculate_and_print_all(mat):
    """
    计算给定矩阵的所有属性和分解，并一次性打印出来。
    """
    print("\n" + "="*50)
    print("              计算结果汇总")
    print("="*50)
    
    print("\n您输入的矩阵 (A):")
    pprint(mat)

    # 1. 矩阵的秩 (Rank)
    print("\n--- 1. 矩阵的秩 ---")
    try:
        rank = mat.rank()
        pprint(rank)
    except Exception as e:
        print(f"计算出错: {e}")

    # 2. 行最简形矩阵 (RREF)
    print("\n--- 2. 行最简形矩阵 (RREF) ---")
    try:
        rref_mat, pivots = mat.rref()
        print("行最简形:")
        pprint(rref_mat)
        print("主元所在的列索引:")
        pprint(pivots)
    except Exception as e:
        print(f"计算出错: {e}")

    # --- 以下多为方阵运算 ---
    if not mat.is_square:
        print("\n" + "="*50)
        print("注意：由于该矩阵不是方阵，以下多数运算（如行列式、逆、特征值等）将无法执行。")
        print("="*50)
        return

    # 3. 行列式 (Determinant)
    print("\n--- 3. 矩阵的行列式 ---")
    try:
        det = mat.det()
        pprint(det)
    except Exception as e:
        print(f"计算出错: {e}")

    # 4. 伴随矩阵 (Adjugate)
    print("\n--- 4. 伴随矩阵 ---")
    try:
        adj_mat = mat.adjugate()
        pprint(adj_mat)
    except Exception as e:
        print(f"计算出错: {e}")

    # 5. 逆矩阵 (Inverse)
    print("\n--- 5. 逆矩阵 ---")
    try:
        # 先检查行列式是否为0
        if mat.det() == 0:
            print("矩阵的行列式为0，该矩阵是奇异矩阵，不可逆。")
        else:
            inv_mat = mat.inv()
            pprint(inv_mat)
    except Exception as e:
        print(f"计算出错: {e}")
        
    # 6. 特征多项式 (Characteristic Polynomial)
    print("\n--- 6. 特征多项式 ---")
    try:
        lamda = symbols('λ') # 使用 lambda 符号
        char_poly = mat.charpoly(lamda)
        print("特征多项式 P(λ) = ")
        pprint(char_poly.as_expr())
        print("\n因式分解形式:")
        pprint(factor(char_poly.as_expr()))
    except Exception as e:
        print(f"计算出错: {e}")

    # 7. 特征值 (Eigenvalues)
    print("\n--- 7. 特征值 ---")
    try:
        eigenvals = mat.eigenvals()
        print("格式: {特征值: 代数重数}")
        pprint(eigenvals)
    except Exception as e:
        print(f"计算出错: {e}")

    # 8. 特征向量 (Eigenvectors)
    print("\n--- 8. 特征向量 ---")
    try:
        eigenvects = mat.eigenvects()
        print("格式: [特征值, 代数重数, [对应的特征向量]]")
        # 为了更清晰的展示，我们遍历打印
        for val, mult, vects in eigenvects:
            print(f"\n对于特征值 λ = {val} (代数重数为 {mult}):")
            print("对应的特征向量为:")
            for v in vects:
                pprint(v)
    except Exception as e:
        print(f"计算出错: {e}")

    # 9. 对角化 (Diagonalization)
    print("\n--- 9. 对角化 (A = PDP⁻¹) ---")
    try:
        P, D = mat.diagonalize()
        print("P (由特征向量构成的可逆矩阵):")
        pprint(P)
        print("\nD (由特征值构成的对角矩阵):")
        pprint(D)
    except sympy.matrices.common.NonDiagonalizableMatrixError:
        print("错误：该矩阵不可对角化（通常因为几何重数小于代数重数）。")
    except Exception as e:
        print(f"计算出错: {e}")

    # 10. LU 分解
    print("\n--- 10. LU 分解 (A = LU) ---")
    try:
        L, U, _ = mat.LUdecomposition()
        print("L (下三角矩阵):")
        pprint(L)
        print("\nU (上三角矩阵):")
        pprint(U)
    except Exception as e:
        print(f"计算出错: {e}")

    # 11. QR 分解
    print("\n--- 11. QR 分解 (A = QR) ---")
    try:
        # 检查矩阵是否包含复数，SymPy的QR分解目前对复数支持不佳
        if any(i.is_complex and not i.is_real for i in mat):
             print("错误：当前QR分解实现不支持复数矩阵。")
        else:
            Q, R = mat.QRdecomposition()
            print("Q (正交矩阵):")
            pprint(Q)
            print("\nR (上三角矩阵):")
            pprint(R)
    except Exception as e:
        print(f"计算出错: {e} (QR分解对包含复杂符号的矩阵可能失败)")

    # 12. 谱分解 (Spectral Decomposition)
    print("\n--- 12. 谱分解 (A = PDP*) ---")
    # 谱分解要求矩阵是正规矩阵 (A*A.H = A.H*A)
    # .H 是共轭转置 (Hermitian transpose)
    try:
        # *** FIX: Use the mathematical definition to check for a normal matrix ***
        if mat * mat.H == mat.H * mat:
            # 对于正规矩阵，其对角化即为谱分解
            P, D = mat.diagonalize(reals_only=False)
            print("谱分解要求矩阵是正规矩阵（例如实对称矩阵），此矩阵满足条件。")
            print("\nP (由正交特征向量构成的酉矩阵):")
            pprint(P)
            print("\nD (由特征值构成的对角矩阵):")
            pprint(D)
        else:
            print("错误：该矩阵不是正规矩阵（即 A*A.H != A.H*A），无法进行谱分解。")
    except Exception as e:
        print(f"计算出错: {e}")


def main():
    """主函数，控制程序的整体流程"""
    print("欢迎使用矩阵基本分解与运算程序！")
    print("本程序将在您输入矩阵后，一次性计算所有可能的结果。")
    
    while True:
        # 1. 获取用户矩阵
        mat = get_matrix_from_user()
        
        # 2. 计算并打印所有结果
        calculate_and_print_all(mat)
        
        # 3. 询问是否继续
        while True:
            another = input("\n是否要计算新的矩阵? (y/n): ").lower().strip()
            if another in ['y', 'n']:
                break
            else:
                print("无效输入，请输入 'y' 或 'n'。")
        
        if another == 'n':
            print("感谢使用，程序已退出。")
            break
        print("\n" + "#"*60 + "\n") # 分隔符，开始新的计算

if __name__ == "__main__":
    main()
