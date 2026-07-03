import torch
import torch.nn as nn
from typing  import Optional , List , Tuple

class ProjetorVisual (nn.Module):
    """
    Projetor  MLP que mapeia features visuais  para o espaço do LLM.

    Equivalente ao MLP  connector do LLaVA -1.5.

    Parâmetros:
        d_vit: dimensão de saída do ViT
        d_model: dimensão de entrada do LLM
        dropout: taxa de dropout
    """

    def __init__(self ,
                d_vit: int = 1024,
                d_model: int = 512,
                dropout: float = 0.0):
        super ().__init__ ()
        self.mlp = nn.Sequential(
            nn.Linear(d_vit , d_model),
            nn.GELU (),
            nn.Dropout(dropout),
            nn.Linear(d_model , d_model),
            nn.LayerNorm(d_model)
        )

    def forward(self , features_vit: torch.Tensor) -> torch.Tensor:
        """
        Parâmetros:
            features_vit: (batch , n_patches , d_vit)
        Retorna:
            tokens_visuais : (batch , n_patches , d_model)
        """
        return  self.mlp(features_vit)


class CodificadorVisualSimples (nn.Module):
    """
    Codificador  visual baseado em ViT pré-treinado (stub para  demonstração).
    Em produção: usar CLIP/SigLIP via  transformers.
    """

    def __init__(self ,
                tamanho_imagem : int = 224,
                tamanho_patch: int = 16,
                d_vit: int = 1024,
                n_cabecas_vit: int = 16,
                n_camadas_vit: int = 24):
        super ().__init__ ()

        n_patches = ( tamanho_imagem // tamanho_patch) ** 2
        d_patch = tamanho_patch ** 2 * 3    # RGB

        self.proj_patch = nn.Linear(d_patch , d_vit)
        self.pos_embed = nn.Parameter(
            torch.randn (1, n_patches + 1, d_vit) * 0.02
        )
        self.cls_token = nn.Parameter(torch.zeros (1, 1, d_vit))

        # Blocos  Transformer do ViT
        encoder_layer = nn. TransformerEncoderLayer (
            d_model=d_vit ,
            nhead=n_cabecas_vit ,
            dim_feedforward =d_vit * 4,
            batch_first=True ,
            norm_first=True
        )
        self.transformer = nn. TransformerEncoder (
            encoder_layer , num_layers=n_camadas_vit
        )
        self.norma = nn.LayerNorm(d_vit)
        self.tamanho_patch = tamanho_patch
        self.n_patches = n_patches

    def _extrair_patches (self , imagem: torch.Tensor) -> torch.Tensor:
        """
        Divide  imagem em patches achatados.
        Parâmetros:
            imagem: (B, C, H, W)
        Retorna:
            patches: (B, n_patches , tamanho_patch ^2 * C)
        """
        B, C, H, W = imagem.shape
        p = self.tamanho_patch

        # Reorganizar em patches (B, n_h , n_w , p, p, C)
        imagem = imagem.unfold (2, p, p).unfold (3, p, p)
        # (B, n_h , n_w , p, p, C) -> (B, n_h*n_w , p* p*C)
        imagem = imagem.contiguous ().view(B, -1, C * p * p)
        return  imagem

    def forward(self , imagem: torch.Tensor) -> torch.Tensor:
        """
        Parâmetros:
            imagem: (batch , 3, 224, 224)
        Retorna:
            features: (batch , n_patches , d_vit) -- sem CLS token
        """
        B = imagem.shape [0]

        patches = self. _extrair_patches (imagem)   # (B, N, p^2 * C)
        x = self.proj_patch(patches) # (B, N, d_vit)

        # Adicionar  CLS token
        cls = self.cls_token.expand(B, -1,   -1)
        x = torch.cat([cls , x], dim =1) # (B, N+1, d_vit)

        # Adicionar  embeddings posicionais
        x = x + self.pos_embed

        # Transformer  encoder
        x = self.transformer(x)
        x = self.norma(x)

        # Retornar  apenas patches (remover  CLS)
        return x[:, 1:, :]


class LLMMultimodal(nn.Module):
    """
    LLM Multimodal = Codificador  Visual + Projetor + LLM de Linguagem.

    Arquitetura  similar ao LLaVA -1.5 e PaliGemma.

    Parâmetros:
        llm: modelo de linguagem  MiniLMPT
        d_vit: dimensão do ViT
        cfg_vit: configuração do ViT
    """

    def __init__(self ,
                llm: nn.Module ,
                d_vit: int = 1024):
        super ().__init__ ()

        self.llm = llm
        d_model = llm.cfg[ 'd_model ']

        self. codificador_visual = CodificadorVisualSimples (d_vit=d_vit)
        self.projetor = ProjetorVisual (d_vit , d_model)

        # Token  especial para início de imagem
        self.id_img_inicio = 4    # Assumindo  vocab especial
        self.id_img_fim     = 5

    def codificar_imagem (self ,
                            imagem: torch.Tensor) -> torch.Tensor:
        """
        Converte  imagem em tokens visuais no espaço do LLM.

        Parâmetros:
            imagem: (batch , 3, 224, 224)
        Retorna:
            tokens_visuais : (batch , n_patches , d_model)
        """
        with  torch.no_grad (): # ViT geralmente congelado inicialmente
            features = self. codificador_visual ( imagem)
        return  self.projetor(features)

    def forward(self ,
                ids_texto: torch.Tensor ,
                imagem: Optional[torch.Tensor] = None ,
                posicoes_img: Optional[List[ int]] = None
                ) -> torch.Tensor:
        """
        Forward  multimodal.

        Parâmetros:
            ids_texto: (batch , seq) -- IDs de tokens de texto
            imagem: (batch , 3, H, W) -- imagem ( opcional)
            posicoes_img: onde inserir  tokens visuais na sequência
        Retorna:
            logits: (batch , seq_total , vocab)
        """
        B = ids_texto.shape [0]
        d = self.llm.cfg[ 'd_model ']

        # Embeddings de texto
        embs_texto = self.llm.embedding(ids_texto) # (B, T, d)

        if imagem is not  None:
            # Tokens  visuais
            tokens_vis = self. codificar_imagem ( imagem)  # (B, N, d)

            # Inserir  tokens visuais antes do texto
            # (abordagem  simplificada: concatenar no início)
            embs = torch.cat([ tokens_vis , embs_texto], dim =1)
        else:
            embs = embs_texto

        # Máscara  causal para sequência  combinada
        T_total = embs.shape [1]
        mascara = torch.full ((1, 1, T_total , T_total), float('-inf'),
                                device=embs.device)
        mascara = torch.triu(mascara , diagonal =1)

        # Forward na pilha  Transformer
        x = embs
        for bloco in  self.llm.blocos:
            x, _ = bloco(x, mascara)

        x = self.llm.norma_final(x)
        logits = self.llm.projecao_saida (x)

        return  logits


# Demonstração
modelo_llm = MiniLMPT(CFG_MINILM)
modelo_mm = LLMMultimodal(modelo_llm , d_vit =256)  # ViT pequeno  para demo

print(f"\nLLM  Multimodal criado!")
n_params = sum (p.numel () for p in modelo_mm.parameters ())
print(f"Total de parâmetros: {n_params /1e6:.1f}M" )

# Teste de forward
ids = torch.randint (0, 32000 , (1, 32))
img = torch.randn(1, 3, 224, 224)

with torch.no_grad ():
    logits = modelo_mm(ids , img)
print(f"Logits  shape: {logits.shape}")
