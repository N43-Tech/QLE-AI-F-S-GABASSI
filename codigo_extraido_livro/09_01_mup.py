import torch
import torch.nn as nn
import math

class MuPLinear(nn.Module):
    """
    Camada Linear  compatível com Maximal  Update Parameterization (mup).

    Aplica o escalonamento  correto no forward  pass e inicialização de pesos.
    """

    def __init__(self ,
                in_features: int ,
                out_features: int ,
                is_output_layer : bool = False ,
                std_base: float = 0.02):
        super ().__init__ ()
        self.in_features = in_features
        self.out_features = out_features
        self.is_output_layer = is_output_layer

        # Alocar  pesos sem bias para simplicidade
        self.weight = nn.Parameter(torch.empty( out_features , in_features))

        # Inicialização baseada na largura
        if is_output_layer :
            # Regra muP para  camada de saída: std proporcional a 1/ d_in
            std = std_base / in_features
        else:
            # Camada  oculta comum: std proporcional a 1/ sqrt(d_in)
            std = std_base / math.sqrt(in_features)

        nn.init.normal_(self.weight , mean =0.0, std= std)

    def forward(self , x: torch.Tensor) -> torch.Tensor:
        # Computação linear clássica
        out = torch.matmul(x, self.weight.t())

        if self. is_output_layer :
            # muP requer  escalonamento de saída por 1/ d_in paracamada  final
            return out / self.in_features
        return out


def configurar_otimizador_mup (model: nn.Module ,
                                lr_base: float ,
                                weight_decay: float  = 0.01)  -> torch.optim.Optimizer:
    """
    Configura o otimizador  AdamW aplicando as regras de escala de LR do muP.
    """
    params_mup = []

    for name , param in model. named_parameters ():
        if not param.requires_grad:
            continue

        # Determinar se é peso de camada de saída ou oculta
        if 'projecao_saida ' in name or 'W_down ' in name:
            # Escalonar  taxa de aprendizado da camada de saída por 1/ d_in
            # Assumimos  que o tamanho de entrada está na segunda dimensão do peso
            d_in = param.shape [1]
            lr_layer = lr_base / d_in
        else:
            # Camadas  normais ocultas usam o LR base estável
            lr_layer = lr_base

        params_mup.append ({
            'params ': [param],
            'lr' : lr_layer ,
            'weight_decay ': weight_decay
        })

    return torch.optim.AdamW(params_mup)
