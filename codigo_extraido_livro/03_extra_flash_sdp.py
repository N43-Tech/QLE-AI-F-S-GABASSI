with torch.backends.cuda.sdp_kernel(enable_flash= True , enable_math=False , enable_mem_efficient = False):
    saida = torch.nn.functional.scaled_dot_product_attention (Q, K, V, is_causal=True)
