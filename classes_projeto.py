from dataclasses import dataclass

@dataclass
class Indicadores:
    # Indicadores de Amorim e Veloso (excluindo divisões)
    num_tokens: int 
    num_prim_pessoa_sing: int
    num_pronomes_demonstrativos: int
    num_enclises: int
    num_erros_gramatica: int
    num_erros_ortografia: int
    num_erros_estilo: int
    num_sentencas_maiores_que_70_chars: int
    num_marcadores_discursivos: int
    num_palavras_diferentes: int
    flesch_score: float
    tamanho_medio_palavras_silabas: float
    similaridade_com_tema_bertimbau: float
    similaridade_com_tema_albertina: float
    similaridade_com_tema_bertugues: float

    # Indicadores Li et al.

    tamanho_medio_sentencas: float
    variancia_tamanho_sentencas: float

    numero_substantivos: int
    numero_adverbios: int
    numero_adjetivos: int
    numero_conjuncoes: int

    # Indicadores propostos

    similaridade_declaracao_direitos_humanos_bertimbau: float
    similaridade_declaracao_direitos_humanos_albertina: float
    similaridade_declaracao_direitos_humanos_bertugues: float
    
    similaridade_cap_1_constituicao_88_bertimbau: float
    similaridade_cap_1_constituicao_88_albertina: float
    similaridade_cap_1_constituicao_88_bertugues: float
 
    # Auxiliar apenas
    num_sentencas: int
    
@dataclass
class Redacao:
    id_redacao: int
    prompt: str
    texto_suporte: str
    texto_redacao: str
    notas: list[int]
    indicadores_texto: Indicadores

def retornar_indicadores_como_vetor(redacao: Redacao):
    vetor = [
        redacao.indicadores_texto.num_prim_pessoa_sing,
        redacao.indicadores_texto.num_prim_pessoa_sing / redacao.indicadores_texto.num_tokens,
        redacao.indicadores_texto.num_pronomes_demonstrativos,
        redacao.indicadores_texto.num_pronomes_demonstrativos / redacao.indicadores_texto.num_tokens,
        redacao.indicadores_texto.num_enclises,
        redacao.indicadores_texto.num_enclises / redacao.indicadores_texto.num_tokens,
        redacao.indicadores_texto.num_sentencas_maiores_que_70_chars,
        redacao.indicadores_texto.num_erros_gramatica,
        redacao.indicadores_texto.num_erros_gramatica / redacao.indicadores_texto.num_tokens,
        redacao.indicadores_texto.num_erros_ortografia,
        redacao.indicadores_texto.num_erros_ortografia / redacao.indicadores_texto.num_tokens,
        redacao.indicadores_texto.num_erros_estilo / redacao.indicadores_texto.num_sentencas,
        redacao.indicadores_texto.num_marcadores_discursivos,
        redacao.indicadores_texto.num_marcadores_discursivos / redacao.indicadores_texto.num_sentencas,
        redacao.indicadores_texto.flesch_score,
        redacao.indicadores_texto.tamanho_medio_palavras_silabas,
        redacao.indicadores_texto.num_tokens,
        redacao.indicadores_texto.num_palavras_diferentes,
        redacao.indicadores_texto.tamanho_medio_sentencas,
        redacao.indicadores_texto.variancia_tamanho_sentencas,
        redacao.indicadores_texto.numero_substantivos,
        redacao.indicadores_texto.numero_adjetivos,
        redacao.indicadores_texto.numero_adverbios,
        redacao.indicadores_texto.numero_conjuncoes,
        redacao.indicadores_texto.similaridade_declaracao_direitos_humanos_bertimbau,
        redacao.indicadores_texto.similaridade_cap_1_constituicao_88_bertimbau,
    ]

    return vetor
