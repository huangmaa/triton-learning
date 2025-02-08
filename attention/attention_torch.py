import torch

N, d = 1024, 64  

Q_mat = torch.rand((N, d))
K_mat = torch.rand((N, d))
V_mat = torch.rand((N, d))


def standard_softmax_attention(Q, K, V):
    """
    执行标准的pytorch softmax和attention计算。
    """
    expected_softmax = torch.softmax(Q @ K.T, dim=1)
    expected_attention = expected_softmax @ V
    return expected_softmax, expected_attention

def safe_softmax_attention(Q, K, V):
    """
    执行安全的softmax和attention计算。
    """
    S_mat = Q @ K.T
    row_max = torch.max(S_mat, dim=1).values[:, None]
    input_safe = S_mat - row_max
    softmax_numerator = torch.exp(input_safe)
    softmax_denominator = torch.sum(softmax_numerator, dim=1)[:, None]
    safe_softmax = softmax_numerator / softmax_denominator
    matmul_result = safe_softmax @ V
    return safe_softmax, matmul_result


def flash_attention(Q, K, V, B_r=64, B_c=768):
    """
    使用分块计算和在线softmax校正执行flash attention算法。
    """
    O = torch.zeros((N, d))  # 初始化输出矩阵，对应伪代码的第2行
    l = torch.zeros((N, 1))  # 存储softmax分母，对应伪代码的第2行
    m = torch.full((N, 1), -torch.inf)  # 存储每个block的最大值，对应伪代码的第2行

    # 对应伪代码的第5行，for 1<=j<=T_c，注意这里是把K, V分成了T_c=[N/B_c]块，每一块的大小是[B_c, d]这么大
    # 所以在python实现的时候就直接通过一个步长为B_c的循环来处理
    for j in range(0, N, B_c):
        # 下面三行就对应了伪代码的第6行，Load Kj, Vj from HBM to on-chip SRAM
        # 但是这里是单纯的 python 实现，我们不可能真的把这一块内存从HBM上放到SRAM上
        # 这里只是一个伪代码的逻辑说明，可以假装它做到了，因为在Triton里面真的可以在Python层做到。
        j_end = j + B_c
        Kj = K[j:j_end, :]
        Vj = V[j:j_end, :]

        # 对应伪代码的第7行，for 1<=i<T_r，注意这里是把Q分成了Tr=[N/B_r]块，每一块的大小是[B_r, d]这么大
        # 所以在python实现的时候就直接通过一个步长为B_r的循环来处理
        for i in range(0, N, B_r):
            i_end = i + B_r
            mi = m[i:i_end, :]
            li = l[i:i_end, :]
            Oi = O[i:i_end, :]
            Qi = Q[i:i_end, :]

            # 对应伪代码的第9行：on chip, compute Sij，Sij的形状是[B_r, B_c]
            Sij = Qi @ Kj.T
            # 对应伪代码的第10行
            mij_hat = torch.max(Sij, dim=1).values[:, None]
            pij_hat = torch.exp(Sij - mij_hat)
            lij_hat = torch.sum(pij_hat, dim=1)[:, None]

            # 对应伪代码的第11行求mi_new的操作，注意这里要对两个张量求整体的max，所以才有这个stack操作
            mi_new = torch.max(torch.column_stack([mi, mij_hat]), dim=1).values[:, None]
            # 对应伪代码的第11行求li_new的操作
            li_new = torch.exp(mi - mi_new) * li + torch.exp(mij_hat - mi_new) * lij_hat
            # 对应伪代码的第12行，更新O_i。这里容易有一个疑问，伪代码上有一个diag操作，为什么下面的实现忽略了
            # 这是因为这个diag是作用在vector上的，实际上是为了在伪代码上能对应上维度，而PyTorch的实现是自动
            # 支持张量广播机制的，所以这里可以直接计算。
            O_i = (li * torch.exp(mi - mi_new) * Oi / li_new) + (torch.exp(mij_hat - mi_new) * pij_hat / li_new) @ Vj

            # 对应伪代码的第13行，更新m_i，l_i，O_i。
            m[i:i_end, :] = mi_new
            l[i:i_end, :] = li_new
            O[i:i_end, :] = O_i

    return O


def flash_attention_v2(Q, K, V, B_r=64, B_c=768):
    """
    使用分块计算和在线softmax校正执行flash attention v2算法。
    """
    O = torch.zeros((N, d))  # 初始化O为(N, d)的形状，实际上对应伪代码第5行的O初始化
    l = torch.zeros((N, 1))  # 初始化l为(N)的形状，实际上对应伪代码第5行的l初始化
    m = torch.full((N, 1), -torch.inf)  # 存储每个block的最大值，初始化为负无穷大，对应伪代码的第5行

    # 对应伪代码的第3行，for 1<=i<T_r，注意这里是把Q分成了Tr=[N/B_r]块，每一块的大小是[B_r, d]这么大
    # 所以在python实现的时候就直接通过一个步长为B_r的循环来处理
    for i in range(0, N, B_r):
        Qi = Q[i:i+B_r, :]
        # 对应伪代码的第 6 行，for 1<=j<=T_c，注意这里是把K, V分成了T_c=[N/B_c]块，每一块的大小是[B_c, d]这么大
        # 所以在python实现的时候就直接通过一个步长为B_c的循环来处理 
        for j in range(0, N, B_c):  # 内循环遍历Q的块
            Kj = K[j:j+B_c, :]
            Vj = V[j:j+B_c, :]

            # 对应伪代码的第8行：on chip, compute Sij，Sij的形状是[B_r, B_c]
            Sij = Qi @ Kj.T
            # 对应伪代码的第9行求m_i^(j)的操作，mi_new的形状是B_r
            mi_new = torch.max(torch.column_stack([m[i:i+B_r], torch.max(Sij, dim=1).values[:, None]]), dim=1).values[:, None]
            # 对应伪代码的第9行求Pij_hat的操作，Pij_hat的形状是(B_r x B_c)，和Sij一致
            Pij_hat = torch.exp(Sij - mi_new)
            # 对应伪代码的第9行求lij的操作
            l[i:i+B_r] = torch.exp(m[i:i+B_r] - mi_new) * l[i:i+B_r] + torch.sum(Pij_hat, dim=1)[:, None]
            # 对应伪代码的第10行求O_ij的操作
            O[i:i+B_r] = O[i:i+B_r] * torch.exp(m[i:i+B_r] - mi_new) + Pij_hat @ Vj
            m[i:i+B_r] = mi_new

    O = O / l  # 对应伪代码第12行，根据softmax的分母校正输出
    return O


# 使用标准softmax和attention计算
expected_softmax, expected_attention = standard_softmax_attention(Q_mat, K_mat, V_mat)
# 使用安全softmax和attention计算
safe_softmax, safe_attention = safe_softmax_attention(Q_mat, K_mat, V_mat)

# 执行flash attention计算
flash_attention_output = flash_attention(Q_mat, K_mat, V_mat)

# 执行flash attention计算
flash_attention_v2_output = flash_attention_v2(Q_mat, K_mat, V_mat)