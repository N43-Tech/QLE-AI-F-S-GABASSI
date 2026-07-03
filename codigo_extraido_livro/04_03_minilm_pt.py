import torch
import torch.nn as nn
from typing  import Optional , Tuple , List

class BlocoTransformer (nn.Module):
    """
    Bloco Transformer  moderno com Pre -LN , GQA+RoPE e SwiGLU.

    Parâmetros:
        d_model: dimensão do modelo
        n_cabecas: número de cabeças de query
        n_cabecas_kv: número de cabeças KV (GQA)
        d_ff: dimensão da FFN
        comprimento_maximo : contexto máximo
        dropout: taxa de dropout
    """

    def __init__(self ,
                d_model: int ,
                n_cabecas: int ,
                n_cabecas_kv: int ,
                d_ff: int ,
                comprimento_maximo : int = 1024 ,
                dropout: float = 0.1):
        super ().__init__ ()

        self.norma_atencao = RMSNorm(d_model)
        self.atencao = AtencaoMultiCabeca (
            d_model , n_cabecas , n_cabecas_kv , comprimento_maximo , dropout
        )
        self.norma_ffn = RMSNorm(d_model)
        self.ffn = FFNSwiGLU(d_model , d_ff , dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self ,
                x: torch.Tensor ,
                mascara: Optional[torch.Tensor] = None ,
                cache_kv: Optional[Tuple] = None
                ) -> Tuple[torch.Tensor , Optional[ Tuple ]]:
        """
        Parâmetros:
            x: (batch , seq , d_model)
            mascara: máscara causal opcional
            cache_kv: cache de KV para  inferência incremental
        Retorna:
            saída: (batch , seq , d_model)
            cache_kv atualizado
        """
        # Sub -bloco de atenção com skip  connection (pré-LN)
        attn_saida , cache_kv = self.atencao(
            self.norma_atencao(x), mascara , cache_kv
        )
        x = x + self.dropout(attn_saida)

        # Sub -bloco FFN com skip connection (pré-LN)
        x = x + self.dropout(self.ffn(self.norma_ffn(x)))

        return x, cache_kv


class MiniLMPT(nn.Module):
    """
    MiniLM -PT: LLM de referência com ~85M parâ metros.

    Arquitetura:
        - Embedding: 32000 x 512 com weight   tying
        - 8 blocos  Transformer (Pre -LN , GQA , RoPE , SwiGLU)
        - RMSNorm  final
        - Projeção de saída compartilhada   com embedding

    Parâmetros:
        cfg: dicionário de configuração do modelo
    """

    def __init__(self , cfg: dict):
        super ().__init__ ()
        self.cfg = cfg

        d          = cfg['d_model ']
        n_camadas = cfg[ 'n_camadas ']
        n_q        = cfg['n_cabecas ']
        n_kv       = cfg['n_cabecas_kv ']
        d_ff       = cfg['d_ff ']
        vocab      = cfg['tamanho_vocab ']
        T_max      = cfg['comprimento_max ']
        dropout    = cfg.get('dropout ', 0.1)

        # Embedding de tokens
        self.embedding = nn.Embedding(vocab , d)

        # Pilha de blocos  Transformer
        self.blocos = nn.ModuleList ([
            BlocoTransformer (d, n_q , n_kv , d_ff , T_max , dropout)
            for _ in range (n_camadas)
        ])

        # Normalização final
        self.norma_final = RMSNorm(d)

        # Projeção de saída = transposta do embedding (weight tying)
        self.projecao_saida = nn.Linear(d, vocab , bias=False)
        self.projecao_saida .weight = self.embedding.weight  # weight tying!

        # Inicialização de pesos
        self. _inicializar_pesos ()

    def _inicializar_pesos (self) -> None:
        """
        Inicialização GPT -2 estilo:
        - Pesos  normais com std =0.02
        - Camadas de projeção residual   escalonadas por 1/ sqrt (2*N)
        """
        for nome , param in self. named_parameters ():
            if param.dim() < 2:
                continue
            if 'embedding ' in nome:
                nn.init.normal_(param , mean =0.0, std =0.02)
            elif 'W_o'  in nome or 'W_down ' in nome:
                # Escalonar projeções residuais
                std = 0.02 / (2 * self.cfg[ ' n_camadas ']) ** 0.5
                nn.init.normal_(param , mean =0.0, std=std)
            else :
                nn.init.normal_(param , mean =0.0, std =0.02)

    def _criar_mascara_causal (self , T: int ,
                                device: torch.device) -> torch.Tensor:
        """ Cria máscara causal triangular  superior."""
        mascara = torch.full ((1, 1, T, T), float ('- inf'), device=device)
        return  torch.triu(mascara , diagonal =1)

    def forward(self ,
                ids: torch.Tensor ,
                cache_kvs: Optional[List] = None ,
                retornar_cache : bool = False
                ) -> Tuple[torch.Tensor , Optional[ List ]]:
        """
        Forward  pass completo do MiniLM -PT.

        Parâmetros:
            ids: (batch , seq) -- IDs de tokens
            cache_kvs: lista de caches KV por camada
            retornar_cache : se True , retornacache KV para inferência
        Retorna:
            logits: (batch , seq , vocab_size)
            cache_kvs: lista de caches (ou None)
        """
        B, T = ids.shape

        # 1) Embedding de tokens
        x = self.embedding(ids)   # (B, T, d)

        # 2) Máscara  causal (não necessária com cache KV em inferência)
        if cache_kvs  is None:
            mascara = self. _criar_mascara_causal (T, ids.device)
            cache_kvs = [None] * len(self.blocos)
        else:
            mascara = None  # Inferência incremental: T=1

        # 3) Pilha de blocos  Transformer
        novos_caches = []
        for i, bloco  in enumerate(self.blocos):
            x, novo_cache = bloco(x, mascara , cache_kvs[i])
            novos_caches.append(novo_cache)

        # 4) Normalização final
        x = self.norma_final(x)

        # 5) Logits via projeção (peso compartilhado com embedding)
        logits = self.projecao_saida (x)   # (B, T, vocab)

        cache_retorno = novos_caches  if retornar_cache else None
        return logits , cache_retorno


# Configuração do MiniLM -PT
CFG_MINILM = {
    'd_model ':        512,
    'n_camadas ':        8,
    'n_cabecas ':        8,
    'n_cabecas_kv ':     2,   # GQA: 4 queries  por grupo KV
    'd_ff ':          2048,
    'tamanho_vocab ': 32000 ,
    'comprimento_max ': 1024,
    'dropout ':         0.1,
}

# Instanciar e inspecionar
modelo = MiniLMPT(CFG_MINILM)

total_params = sum (p.numel () for p in modelo.parameters ())
params_treinaveis =  sum(p.numel () for p in modelo.parameters ()
                        if p.requires_grad)

print(f"MiniLM -PT inicializado com sucesso!" )
print(f"Total de parâmetros: {total_params :,} ({ total_params /1e6:.1f}M)")
print(f"Parâmetros  treináveis: { params_treinaveis:,}")
print(f"Memória FP32: {total_params * 4 / 1e9:.2f} GB")
print(f"Memória BF16: {total_params * 2 / 1e9:.2f} GB")

# Teste de forward  pass
ids_teste = torch.randint (0, CFG_MINILM[ ' tamanho_vocab '], (2, 128))
with torch.no_grad ():
    logits , _ = modelo(ids_teste)
print(f"\nForward  pass OK! Logits shape: {logits.shape}" )
