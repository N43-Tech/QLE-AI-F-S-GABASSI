import torch
import torch.nn as nn

def contar_parametros (modelo: nn.Module ,
                        so_treinaveis: bool = True) -> dict:
    """
    Conta e categoriza os parâmetros de um modelo PyTorch.

    Parâmetros:
        modelo: instância de nn.Module
        so_treinaveis: se True , conta só parâmetros com grad
    Retorna:
        dicionário com estatísticas  detalhadas
    """
    total = 0
    por_camada = {}

    for nome , param in modelo. named_parameters ():
        if so_treinaveis  and not param.requires_grad:
            continue
        n_params = param.numel ()
        total += n_params

        # Agrupar por módulo de nível  superior
        modulo = nome.split( '.')[0]
        por_camada[modulo] = por_camada.get(modulo , 0) + n_params

    return {
        'total ': total ,
        'total_M ': total / 1e6 ,
        'por_camada ': por_camada
    }


def tabela_parametros (modelo: nn.Module) -> None:
    """
    Imprime tabela  formatada de parâmetros por mó dulo.
    """
    stats = contar_parametros (modelo)
    print(f"\n{ '= '*55}")
    print(f"{'Módulo ':<35} {'Parâmetros ':>10} {'%':>6}" )
    print(f"{'-'*55}" )

    for nome , n in sorted(stats['por_camada ']. items (),
                            key=lambda x: -x[1]):
        pct = 100 * n / stats[ 'total ']
        print(f"{nome :<35} {n:>10,} {pct :>5.1f}%")

    print(f"{ '= '*55}")
    print(f"{'TOTAL ':<35} {stats['total ']:>10,}")
    print(f"{'TOTAL (Milhões) ':<35} {stats['total_M ']: >10.2f}M")
    print(f"{'Memória (FP32) ':<35} {stats['total ']*4/1e9:>9.2f}GB")
    print(f"{'Memória (BF16) ':<35} {stats['total ']*2/1e9:>9.2f}GB")
