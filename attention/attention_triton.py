# _attn_fwd_inner函数负责的是一小块Q（入参中的q）和KV的计算,即伪代码中的对KV的循环，Q的循环实际上是体现在triton kernel启动的设置
@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q,  #
                    K_block_ptr, V_block_ptr,  #
                    start_m, qk_scale,  #
                    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    N_CTX: tl.constexpr):
    # range of values handled by this stage
    # 根据STAGE的值，函数定义了处理的键（K）和值（V）的范围。
    # 不同的STAGE对应不同的处理范围，支持因果（causal）和非因果（non-causal）的自注意力。
    if STAGE == 1: # causal = True
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, N_CTX
    # 使用tl.advance函数调整K和V指针的位置，以正确地从相应的内存位置加载数据。
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    # loop over k, v and update accumulator
    # 在一个循环中，函数加载键（K）的一个块，计算查询（Q）与这个键块的点积，
    # 然后根据当前STAGE调整计算结果。如果是STAGE 2并且因果关系为真，会应用一个掩码来屏蔽未来的信息。
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(K_block_ptr)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            # 对应算法流程伪代码的第9行的m_ij的计算，和伪代码的区别是这里应用了qk_scale
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        # 计算p，对应伪代码的第9行的p的计算
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        v = tl.load(V_block_ptr)
        acc += tl.dot(p.to(tl.float16), v)
        # update m_i and l_i
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    return acc, l_i, m_i


@triton.jit
# 定义了一个名为_attn_fwd的函数。这个函数是实现注意力机制前向pass的kernel。函数参数包括输入的Query（Q）、Key（K）、Value（V）张量，
# softmax缩放因子（sm_scale），一个中间计算结果（M）和输出张量（Out），以及多个关于这些张量的步长（stride）参数和其他配置常量。
def _attn_fwd(Q, K, V, sm_scale, M, Out,  #
              stride_qz, stride_qh, stride_qm, stride_qk,  #
              stride_kz, stride_kh, stride_kn, stride_kk,  #
              stride_vz, stride_vh, stride_vk, stride_vn,  #
              stride_oz, stride_oh, stride_om, stride_on,  #
              Z, H,  #
              N_CTX: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_DMODEL: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              STAGE: tl.constexpr  #
              ):
    # 注意，输入参数里的Z和H分别表示batch size和注意力头数
    # start_m表示当前kernel program 实例对应的seq维度的偏移，而off_hz表示的是batch*heads维度的偏移。
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    # 这些行计算了两个偏移量off_z和off_h，它们分别代表在batch（或heads）中的位置。
    off_z = off_hz // H
    off_h = off_hz % H
    # 计算用于定位Q、K和V张量中当前处理块的偏移量。这是基于先前计算的偏移量和提供的步长参数。
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

    # block pointers
    # 使用tl.make_block_ptr创建一个指向Q张量当前处理块的指针。这个函数调用指定了基础地址、形状、步长、偏移量和块形状等，以及如何在内存中访问这个数据块。
    # N_CTX 是q.shape[2]，表示的是序列长度，BLOCK_DMODEL是Lk，表示的是每个注意力头的隐藏层维度大小
    # 下面几个make_block_ptr创建的张量类似，分别是对K，V以及输出O创建指向当前处理块的指针
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    # initialize offsets
    # 计算M维度（seq维度）上每个线程应处理的元素的起始偏移量。
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # 计算N维度（batch*heads维度）上每个线程应处理的元素的偏移量。
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    # 初始化m向量，m用于存储每个m维度上的最大logit，初始化为负无穷大。
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    # 初始化l向量，l用于累计softmax的分母，初始化为1。
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    # 初始化累加器，用于累积注意力加权和。注意这里的shape是(BLOCK_M, BLOCK_DMODEL)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale     # 加载softmax缩放因子。
    qk_scale *= 1.44269504  # 将softmax缩放因子乘以1/log(2)，用于后续计算。
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr) # 将Q矩阵的当前块加载到SRAM中，此数据在整个计算过程中保持不变。
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,  #
                                        start_m, qk_scale,  #
                                        BLOCK_M, BLOCK_DMODEL, BLOCK_N,  #
                                        4 - STAGE, offs_m, offs_n, N_CTX  #
                                        )
    # stage 2: on-band
    if STAGE & 2:
        # barrier makes it easier for compielr to schedule the
        # two loops independently
        tl.debug_barrier()
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,  #
                                        start_m, qk_scale,  #
                                        BLOCK_M, BLOCK_DMODEL, BLOCK_N,  #
                                        2, offs_m, offs_n, N_CTX  #
                                        )
    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))


# 定义了一个_attention类，它继承自torch.autograd.Function。允许自定义一个操作的前向和后向传播
# 能够与PyTorch的自动梯度计算系统无缝集成。
class _attention(torch.autograd.Function):

    @staticmethod
    # forward方法定义了这个自定义操作的前向传播逻辑。ctx是一个上下文对象，用于存储用于反向传播的信息。
    # q, k, v分别代表query, key, value三个输入Tensor，causal和sm_scale是额外的控制参数。
    def forward(ctx, q, k, v, causal, sm_scale):
        # shape constraints
        # 这几行代码检查输入Tensor的最后一个维度，确保它们的大小相等且为特定的值（16, 32, 64, 或 128）。这是由于实现的特定性能优化需要。
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        assert Lk in {16, 32, 64, 128}
        # 初始化一个与q相同形状和类型的空Tensoro，用于存储输出结果。
        o = torch.empty_like(q)
        # 这几行设置了几个关键的性能调优参数，包括处理块的大小（BLOCK_M, BLOCK_N）和计算阶段的数量（num_stages）。num_warps指的是每个CUDA block中warp的数量。由于现在的Q，K，V形状和paper中的(N,d)不一样，所以分块的个数也是不一样的，这里是写死了分块数。
        BLOCK_M = 128
        BLOCK_N = 64 if Lk <= 64 else 32
        num_stages = 4 if Lk <= 64 else 3
        num_warps = 4
        stage = 3 if causal else 1
        # 根据CUDA设备的能力（这里检查的是计算能力9.x，即NVIDIA Volta架构及以后的架构），进一步调整num_warps和num_stages。
        # Tuning for H100
        if torch.cuda.get_device_capability()[0] == 9:
            num_warps = 8
            num_stages = 7 if Lk >= 64 else 3
        # 计算Triton kernel的网格尺寸。triton.cdiv是一个辅助函数，用于计算向上取整的除法。
        # q.shape[2]是序列长度，q.shape[0]和q.shape[1]分别是batch和seq length
        grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)
        # 初始化另一个TensorM，用于在计算过程中存储中间结果。
        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        # 调用Triton kernel _attn_fwd执行实际的attention计算。这里传递了大量的参数，包括输入Tensor的各个维度、步长（stride）、形状、调优参数等。
        _attn_fwd[grid](
            q, k, v, sm_scale, M, o,  #
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
            q.shape[0], q.shape[1],  #
            N_CTX=q.shape[2],  #
            BLOCK_M=BLOCK_M,  #
            BLOCK_N=BLOCK_N,  #
            BLOCK_DMODEL=Lk,  #
            STAGE=stage,  #
            num_warps=num_warps,  #
            num_stages=num_stages  #
        )