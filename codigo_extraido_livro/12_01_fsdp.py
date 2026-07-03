import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp  import (
    FullyShardedDataParallel as FSDP ,
    MixedPrecision , BackwardPrefetch , StateDictType
)
from torch.distributed.fsdp.wrap  import (
    size_based_auto_wrap_policy
)
import functools
import os

def configurar_distribuido () -> None:
    """ Inicializa o grupo de processos  PyTorch."""
    dist. init_process_group (backend='nccl ')
    rank = dist.get_rank ()
    torch.cuda.set_device(rank)
    print(f"Processo {rank }/{ dist. get_world_size ()} inicializado.")


def criar_modelo_fsdp (cfg: dict) -> FSDP:
    """
    Cria MiniLMPT  com FSDP (Fully Sharded  Data Parallel).
    Equivalente ao ZeRO -3, mas nativo no PyTorch.

    Parâmetros:
        cfg: configuração do modelo
    Retorna:
        modelo  FSDP pronto para treinamento
    """
    rank = dist.get_rank ()

    # Criar  modelo na CPU primeiro (evitar OOM)
    with torch.device( 'meta '):
        modelo = MiniLMPT(cfg)

    # Política de auto -wrapping: shardear módulos > 1M parâmetros
    politica_wrap = functools.partial(
        size_based_auto_wrap_policy ,
        min_num_params =1_000_000
    )

    # Configuração de precisão mista
    precisao_mista = MixedPrecision (
        param_dtype=torch.bfloat16 ,
        reduce_dtype=torch.bfloat16 ,
        buffer_dtype=torch.bfloat16 ,
    )

    # Embrulhar  com FSDP
    modelo_fsdp = FSDP(
        modelo ,
        auto_wrap_policy =politica_wrap ,
        mixed_precision =precisao_mista ,
        backward_prefetch = BackwardPrefetch .BACKWARD_PRE ,
        device_id=torch.cuda. current_device (),
        param_init_fn= lambda m: m.to_empty(
            device=torch.cuda.current_device ()
        ),
    )

    if rank == 0:
        # Contar parâmetros  totais
        n_params = sum (p.numel () for p in modelo_fsdp.parameters ())
        print(f"Modelo FSDP criado: {n_params /1e6:.1f}M parâmetros")
        print(f"World size: {dist. get_world_size ()} GPUs")
        print(f"Parâmetros por GPU (ZeRO -3): "
                f"~{ n_params/dist. get_world_size ()/1e6:.1f}M")

    return modelo_fsdp


def salvar_checkpoint_fsdp (modelo: FSDP ,
                                otimizador: torch.optim.Optimizer ,
                                caminho: str ,
                                passo: int) -> None:
    """
    Salva checkpoint de modelo  FSDP de forma eficiente.
    Usa FULL_STATE_DICT  para compatibilidade  com modelos não-FSDP.
    """
    with FSDP. state_dict_type (modelo , StateDictType. FULL_STATE_DICT ):
        estado = modelo.state_dict ()

    if dist.get_rank () == 0:
        torch.save ({
            'passo ': passo ,
            'modelo ': estado ,
            'otimizador ': otimizador.state_dict (),
        }, f"{caminho }/ checkpoint_passo_ {passo }.pt")
        print(f"Checkpoint salvo: passo {passo}" )
