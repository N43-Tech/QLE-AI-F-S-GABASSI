import torch
import torch.nn as nn
import torch.nn.functional as F

class NF4Quantizer:
    """
    Quantizador em nivel de bloco  utilizando o tipo de dados  NormalFloat 4 (NF4).
    """
    # Os 16 quantis  oficiais do NormalFloat 4 (NF4) de Dettmers et al.
    NF4_VALS = torch.tensor ([
        -1.00000000 , -0.69619170 ,-0.52507971 , -0.39307481 ,
        -0.27961440 , -0.17298910 ,-0.07007480 , 0.00000000 ,
        0.07007480 ,  0.17298910 , 0.27961440 , 0.39307481 ,
        0.52507971 ,  0.69619170 , 0.85306480 , 1.00000000
    ])

    @classmethod
    def quantizar(cls , x: torch.Tensor , tamanho_bloco:  int = 64):
        """
        Quantiza um tensor  para indices NF4 de 4- bit.

        Retorna:
            indices: tensor  ByteTensor contendo indices de 0 a 15
            escalas: fatores de escala de cada bloco
        """
        original_shape = x.shape
        x_flat = x.flatten ()
        n_elementos = x_flat.numel ()

        # Padding  caso o tamanho nao seja  multiplo do bloco
        padding = (tamanho_bloco - (n_elementos % tamanho_bloco)) % tamanho_bloco
        if padding > 0:
            x_flat = F.pad(x_flat , (0, padding))

        x_blocos = x_flat.view(-1, tamanho_bloco)

        # 1) Calcular  escala absoluta por bloco
        escalas = torch. max(torch.abs(x_blocos), dim=-1, keepdim=True)[0]
        escalas = torch.clamp(escalas , min =1e-7)  # Evitar divisao por zero

        # 2) Normalizar  bloco
        blocos_norm = x_blocos / escalas

        # 3) Mapear  para o index NF4 mais  proximo
        # Diferenca  absoluta em relacao aos 16 quantis NF4
        dists = torch. abs(blocos_norm.unsqueeze (-1) - cls.NF4_VALS.to(x.device))
        indices = torch.argmin(dists , dim=-1).to( torch.uint8)

        return  indices , escalas , original_shape

    @classmethod
    def dequantizar(cls , indices: torch.Tensor , escalas: torch.Tensor ,
                    original_shape , tamanho_bloco: int = 64) -> torch.Tensor:
        """
        Dequantiza  indices 4-bit de volta  para FP32 /FP16 usando escalas.
        """
        # Mapear  indices para valores float
        nf4_vals_device = cls.NF4_VALS.to(indices.device)
        blocos_nf4 = nf4_vals_device [indices.to( torch.long)]

        # Escalonar de volta
        blocos_dequant = blocos_nf4 * escalas
        x_flat = blocos_dequant .flatten ()

        # Remover  padding e retornar ao shape original
        n_total = 1
        for s in original_shape :
            n_total *= s
        return  x_flat [: n_total ]. view(original_shape)


class QLoRALinear(nn.Module):
    """
    Camada Linear com base  quantizada em 4-bit (NF4) e adaptadores  LoRA treinaveis.
    """
    def __init__(self ,
                in_features: int ,
                out_features: int ,
                rank: int = 8,
                alpha: float = 16.0,
                tamanho_bloco: int = 64):
        super ().__init__ ()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.escala = alpha / rank
        self.tamanho_bloco = tamanho_bloco

        # Inicializar  peso temporario de alta precisao para quantizacao
        peso_temp = torch.empty(out_features , in_features)
        nn.init. kaiming_uniform_ (peso_temp , a=0.5)

        # 1) Quantizar  peso original e registrar como buffer congelado
        indices , escalas , shape = NF4Quantizer.quantizar(peso_temp , tamanho_bloco)
        self.register_buffer ("peso_indices" , indices)
        self.register_buffer ("peso_escalas" , escalas)
        self.original_shape = shape

        # 2) Adaptadores  LoRA de alta precisao ( treinaveis)
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros( out_features , rank))

        nn.init. kaiming_uniform_ (self.lora_A , a =1.0)

    def forward(self , x: torch.Tensor) -> torch.Tensor:
        # 1) Dequantizar  peso base para o forward pass
        W_base = NF4Quantizer.dequantizar(
            self.peso_indices ,
            self.peso_escalas ,
            self.original_shape ,
            self.tamanho_bloco
        ).to(x.dtype)

        # 2) Computar  saida do modelo base congelado
        saida_base = F.linear(x, W_base)

        # 3) Computar  ramo LoRA
        saida_lora = F.linear(x, self.lora_A)
        saida_lora = F.linear(saida_lora , self.lora_B)

        # Combinar as saidas
        return  saida_base + self.escala * saida_lora
