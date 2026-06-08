from classes_projeto import *
from constantes_longas import marcadores_discursivos, titulo_1_constituicao_88, declaracao_direitos_humanos, pronomes_demonstrativos
from sentence_transformers import SentenceTransformer, util
import spacy
import statistics
import pyphen
import language_tool_python
import pandas as pd
import time

from transformers import AutoTokenizer

tool = language_tool_python.LanguageTool('pt-BR')
hyphenator = pyphen.Pyphen(lang='pt_BR')
model_bertimbau = SentenceTransformer('neuralmind/bert-base-portuguese-cased')
# model_albertina = SentenceTransformer('PORTULAN/albertina-100m-portuguese-ptbr-encoder')
# model_bertugues = SentenceTransformer('ricardoz/BERTugues-base-portuguese-cased')

tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")

lista_pronomes_prim_pessoa_sing = ["eu", "me", "mim", "comigo", "meu", "minha", "meus", "minhas"]
pronomes_enclise = ["me", "te", "se", "lhe", "nos", "vos", "lhes", "o", "a", "os", "as"]

pln = spacy.load("pt_core_news_lg")

emb_const = model_bertimbau.encode(titulo_1_constituicao_88, convert_to_tensor=True, normalize_embeddings=True)
emb_dh = model_bertimbau.encode(declaracao_direitos_humanos, convert_to_tensor=True, normalize_embeddings=True)

def _indicadores_spacy(dados_redacao: Redacao):
    tokens = pln(dados_redacao.texto_redacao)

    dados_redacao.indicadores_texto.num_tokens = len(tokens)
    
    # -- P/ calcular o tamanho médio e a variância
    tamanhos_sentencas_caracteres = []

    # -- P/ calcular o tamanho médio e a pontuação Flesch adaptada
    tamanhos_silabas = []
    tamanhos_sentencas_palavras = []
    numero_palavras = 0

    conjunto_palavras_distintas = set()

    dados_redacao.indicadores_texto.num_sentencas = len(list(tokens.sents))

    # -- Indicadores que usam o SpaCy
    for sentenca in tokens.sents:

        tamanho = len(sentenca.text.strip())
        
        if tamanho > 70:
            # print(f"Sentenca maior que 70 chars: {sentenca.text}")
            dados_redacao.indicadores_texto.num_sentencas_maiores_que_70_chars += 1

        tamanhos_sentencas_caracteres.append(tamanho)
        tamanhos_sentencas_palavras.append(len(sentenca.text.split()))

    # -- Tamanho médio e variância das sentenças
    dados_redacao.indicadores_texto.tamanho_medio_sentencas = statistics.mean(tamanhos_sentencas_caracteres)
    dados_redacao.indicadores_texto.variancia_tamanho_sentencas = statistics.variance(tamanhos_sentencas_caracteres)

    # print(f"Tamanhos sentencas: {str(tamanhos_sentencas_palavras)}  Tamanho med: {dados_redacao.indicadores_texto.tamanho_medio_sentencas}  Variancia: {dados_redacao.indicadores_texto.variancia_tamanho_sentencas}")

    for token in tokens:
        # print(f"{token.text} {token.pos_} {token.morph}")

        # -- Número de verbos da primeira pessoa do singular
        if token.pos_ == "VERB" and "1" in token.morph.get("Person") and "Sing" in token.morph.get("Number"):
            dados_redacao.indicadores_texto.num_prim_pessoa_sing += 1
        
        # -- Número de pronomes na primeira pessoa do singular
        if token.text.lower() in lista_pronomes_prim_pessoa_sing:
            dados_redacao.indicadores_texto.num_prim_pessoa_sing += 1

        # -- Número de pronomes demonstrativos
        if token.text.lower() in pronomes_demonstrativos:
            dados_redacao.indicadores_texto.num_pronomes_demonstrativos += 1

        # -- Número de ênclises    
        if "-" in token.text and token.pos_ == "VERB":
            pronome = token.text.split("-")[1]
            if pronome in pronomes_enclise:
                dados_redacao.indicadores_texto.num_enclises += 1

        if token.is_alpha:
            # -- Contagem de palavras diferentes
            conjunto_palavras_distintas.add(token.text.lower())

            palavra_separada_silabas = hyphenator.inserted(token.text)
            numero_palavras += 1
            tamanhos_silabas.append(len(palavra_separada_silabas.split('-')))
            # print(f"{token.text}: {len(palavra_separada_silabas.split('-'))}")
            pass
            
        # -- Contagem de classes gramaticais
        if token.pos_ == "NOUN":
            dados_redacao.indicadores_texto.numero_substantivos += 1
        elif token.pos_ == "ADV":
            dados_redacao.indicadores_texto.numero_adverbios += 1
        elif token.pos_ == "ADJ":
            dados_redacao.indicadores_texto.numero_adjetivos += 1
        elif token.pos_ == "CCONJ" or token.pos_ == "SCONJ":
            dados_redacao.indicadores_texto.numero_conjuncoes += 1

    # print(f"Tamanhos silabas: {tamanhos_silabas}  Numero palavras: {numero_palavras}")

    dados_redacao.indicadores_texto.num_palavras_diferentes = len(conjunto_palavras_distintas)

    for sentenca in tokens.sents:
        texto = sentenca.text

        # Os marcadores estão ordenados por maior número de palavras até o menor
        for marcador in marcadores_discursivos:
            if marcador in texto:
                # Remover p/ evitar duplicatas
                # Ex: "X bem como Y" ativaria tanto p/ "bem como" tanto como "como" 
                texto = texto.replace(marcador, "")
                dados_redacao.indicadores_texto.num_marcadores_discursivos += 1

    dados_redacao.indicadores_texto.tamanho_medio_palavras_silabas = statistics.mean(tamanhos_silabas)

    # Flesch PT (Martins et al., 1996)
    # Score = 206.835 - 84.6 * med_silabas_por_palavra - 1.015 * med_palavras_por_sentenca + 42

    dados_redacao.indicadores_texto.flesch_score = 206.835 - 84.6 * statistics.mean(tamanhos_silabas) - 1.015 * statistics.mean(tamanhos_sentencas_palavras) + 42


def _calculo_similaridade(texto_a: str, texto_b: str, modelo_escolhido: int):

    if modelo_escolhido == 0:
        emb_a = model_bertimbau.encode(texto_a, convert_to_tensor=True, normalize_embeddings=True)
        emb_b = model_bertimbau.encode(texto_b, convert_to_tensor=True, normalize_embeddings=True)
    # elif modelo_escolhido == 1:
    #     emb_a = model_albertina.encode(texto_a, convert_to_tensor=True, normalize_embeddings=True)
    #     emb_b = model_albertina.encode(texto_b, convert_to_tensor=True, normalize_embeddings=True)
    # else:
    #     emb_a = model_bertugues.encode(texto_a, convert_to_tensor=True, normalize_embeddings=True)
    #     emb_b = model_bertugues.encode(texto_b, convert_to_tensor=True, normalize_embeddings=True)
        
    # emb_a = model.encode(texto_a, convert_to_tensor=True, normalize_embeddings=True)
    # emb_b = model.encode(texto_b, convert_to_tensor=True, normalize_embeddings=True)

    similaridade = util.cos_sim(emb_a, emb_b).item()
    # print(f"Similaridade cosseno: {similaridade:.4f}")
    return similaridade

def _calculo_similaridade_emb(emb_a, emb_b: str):
    similaridade = util.cos_sim(emb_a, emb_b).item()
    return similaridade


def _indicadores_similaridade(dados_redacao: Redacao):
    global emb_const, emb_dh

    emb_red = model_bertimbau.encode(dados_redacao.texto_redacao, convert_to_tensor=True, normalize_embeddings=True)
    emb_prompt = model_bertimbau.encode(dados_redacao.prompt, convert_to_tensor=True, normalize_embeddings=True)
    
    dados_redacao.indicadores_texto.similaridade_com_tema_bertimbau = _calculo_similaridade_emb(emb_red, emb_prompt)
    
    dados_redacao.indicadores_texto.similaridade_cap_1_constituicao_88_bertimbau = _calculo_similaridade_emb(emb_red, emb_const)
    
    dados_redacao.indicadores_texto.similaridade_declaracao_direitos_humanos_bertimbau = _calculo_similaridade_emb(emb_red, emb_dh)
    

def _indicadores_language_tool(dados_redacao: Redacao):
    erros = tool.check(dados_redacao.texto_redacao)

    for erro in erros:
        if erro.ruleIssueType == "misspelling":
            dados_redacao.indicadores_texto.num_erros_ortografia += 1
        elif erro.ruleIssueType == "style":
            dados_redacao.indicadores_texto.num_erros_estilo += 1
        elif erro.ruleIssueType == "grammar":
            dados_redacao.indicadores_texto.num_erros_gramatica += 1
        elif erro.ruleIssueType == "uncategorized":
            if erro.ruleId == "VERB_COMMA_CONJUNCTION":
                # A maior fatia dos não-categorizados são desse tipo, sendo erros de gramática
                # print("Achou um erro de VERB COMMA CONJUNCTION")
                dados_redacao.indicadores_texto.num_erros_gramatica += 1
            else:
                # Das demais, a maioria são paronímias, onde se assume que a palavra foi escrita errada, sendo um erro de ortografia
                dados_redacao.indicadores_texto.num_erros_ortografia += 1


def calcular_indicadores(dados_redacao: Redacao) -> None:
    """
    Calcula os indicados dado o texto de redação de entrada e o prompt da proposta de redação
    """

    # print("Spacy...")
    _indicadores_spacy(dados_redacao)
    # print("Similaridade...")
    _indicadores_similaridade(dados_redacao)
    # print("LT...")
    _indicadores_language_tool(dados_redacao)

    # print(vars(dados_redacao.indicadores_texto))


if __name__ == "__main__":

    arquivos = ["test.csv", "train.csv", "validation.csv"]

    contador_id = 0

    dados_p_df = {
        'id': [],
        'nota_total': [],
        'nota_c1': [],
        'nota_c2': [],
        'nota_c3': [],
        'nota_c4': [],
        'nota_c5': [],
        'n_verbos_e_pronomes_1ps': [],
        'n_verbos_e_pronomes_1ps_tok': [],
        'n_pron_dem': [],
        'n_pron_dem_tok': [],
        'n_enclises': [],
        'n_enclises_tok': [],
        'n_sentencas_mq_70': [],
        'n_erros_gramatica': [],
        'n_erros_gramatica_tok': [],
        'n_erros_ortografia': [],
        'n_erros_ortografia_tok': [],
        'n_erros_estilo_num_sent': [],
        'n_marcadores_discursivos': [],
        'n_marcadores_discursivos_num_sent': [],
        'flesch_score': [],
        'tam_avg_silabas': [],
        'n_tokens': [],
        'similaridade_proposta_bertimbau': [],
        # 'similaridade_proposta_albertina': [],
        # 'similaridade_proposta_bertugues': [],
        'numero_palavras_diferentes': [],
        'tam_avg_sentencas_caracteres': [],
        'variancia_tam_sentencas': [],
        'n_substantivos': [],
        'n_adjetivos': [],
        'n_adverbios': [],
        'n_conjuncoes': [],
        'similaridade_direitos_humanos_bertimbau': [],
        # 'similaridade_direitos_humanos_albertina': [],
        # 'similaridade_direitos_humanos_bertugues': [],
        'similaridade_tit_1_constituicao_bertimbau': [],
        # 'similaridade_tit_1_constituicao_albertina': [],
        # 'similaridade_tit_1_constituicao_bertugues': [],
    }
    
    tempo_inicio = time.time()

    for nome_arquivo in arquivos:
        df = pd.read_csv(f"redacoes_enem_original/{nome_arquivo}")
        print(f"Arquivo: {nome_arquivo}")

        for i, linha in df.iterrows():
            contador_id += 1
            id_red = contador_id

            print(f"{contador_id}")

            prompt = str(linha.get('prompt', ''))

            texto_suporte = str(linha.get('supporting_text', ''))

            texto_da_redacao = str(linha.get('essay_text', ''))

            notas_str = str(linha.get('grades', ''))
            notas_str = notas_str.replace('[','').replace(']','').strip()
            while "  " in notas_str:
                notas_str = notas_str.replace("  ", " ")
            
            notas = [int(valor) for valor in notas_str.split(" ")]

            dados_redacao: Redacao = Redacao(id_red, prompt, texto_suporte, texto_da_redacao, notas, indicadores_texto=None)
            dados_redacao.indicadores_texto = Indicadores(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

            calcular_indicadores(dados_redacao)

            dados_p_df['id'].append(dados_redacao.id_redacao)
            dados_p_df['nota_total'].append(dados_redacao.notas[5])
            dados_p_df['nota_c5'].append(dados_redacao.notas[4])
            dados_p_df['nota_c4'].append(dados_redacao.notas[3])
            dados_p_df['nota_c3'].append(dados_redacao.notas[2])
            dados_p_df['nota_c2'].append(dados_redacao.notas[1])
            dados_p_df['nota_c1'].append(dados_redacao.notas[0])

            dados_p_df['n_verbos_e_pronomes_1ps'].append(dados_redacao.indicadores_texto.num_prim_pessoa_sing)
            dados_p_df['n_verbos_e_pronomes_1ps_tok'].append(dados_redacao.indicadores_texto.num_prim_pessoa_sing / dados_redacao.indicadores_texto.num_tokens)
            dados_p_df['n_pron_dem'].append(dados_redacao.indicadores_texto.num_pronomes_demonstrativos)
            dados_p_df['n_pron_dem_tok'].append(dados_redacao.indicadores_texto.num_pronomes_demonstrativos / dados_redacao.indicadores_texto.num_tokens)
            dados_p_df['n_enclises'].append(dados_redacao.indicadores_texto.num_enclises)
            dados_p_df['n_enclises_tok'].append(dados_redacao.indicadores_texto.num_enclises / dados_redacao.indicadores_texto.num_tokens)
            dados_p_df['n_sentencas_mq_70'].append(dados_redacao.indicadores_texto.num_sentencas_maiores_que_70_chars)
            dados_p_df['n_erros_gramatica'].append(dados_redacao.indicadores_texto.num_erros_gramatica)
            dados_p_df['n_erros_gramatica_tok'].append(dados_redacao.indicadores_texto.num_erros_gramatica / dados_redacao.indicadores_texto.num_tokens)
            dados_p_df['n_erros_ortografia'].append(dados_redacao.indicadores_texto.num_erros_ortografia)
            dados_p_df['n_erros_ortografia_tok'].append(dados_redacao.indicadores_texto.num_erros_ortografia / dados_redacao.indicadores_texto.num_tokens)
            dados_p_df['n_erros_estilo_num_sent'].append(dados_redacao.indicadores_texto.num_erros_estilo / dados_redacao.indicadores_texto.num_sentencas)
            dados_p_df['n_marcadores_discursivos'].append(dados_redacao.indicadores_texto.num_marcadores_discursivos)
            dados_p_df['n_marcadores_discursivos_num_sent'].append(dados_redacao.indicadores_texto.num_marcadores_discursivos / dados_redacao.indicadores_texto.num_sentencas)
            dados_p_df['flesch_score'].append(dados_redacao.indicadores_texto.flesch_score)
            dados_p_df['tam_avg_silabas'].append(dados_redacao.indicadores_texto.tamanho_medio_palavras_silabas)
            dados_p_df['n_tokens'].append(dados_redacao.indicadores_texto.num_tokens)
            dados_p_df['similaridade_proposta_bertimbau'].append(dados_redacao.indicadores_texto.similaridade_com_tema_bertimbau)
            # dados_p_df['similaridade_proposta_albertina'].append(dados_redacao.indicadores_texto.similaridade_com_tema_albertina)
            # dados_p_df['similaridade_proposta_bertugues'].append(dados_redacao.indicadores_texto.similaridade_com_tema_bertugues)
            dados_p_df['numero_palavras_diferentes'].append(dados_redacao.indicadores_texto.num_palavras_diferentes)
            dados_p_df['tam_avg_sentencas_caracteres'].append(dados_redacao.indicadores_texto.tamanho_medio_sentencas)
            dados_p_df['variancia_tam_sentencas'].append(dados_redacao.indicadores_texto.variancia_tamanho_sentencas)
            dados_p_df['n_substantivos'].append(dados_redacao.indicadores_texto.numero_substantivos)
            dados_p_df['n_adjetivos'].append(dados_redacao.indicadores_texto.numero_adjetivos)
            dados_p_df['n_adverbios'].append(dados_redacao.indicadores_texto.numero_adverbios)
            dados_p_df['n_conjuncoes'].append(dados_redacao.indicadores_texto.numero_conjuncoes)
            dados_p_df['similaridade_direitos_humanos_bertimbau'].append(dados_redacao.indicadores_texto.similaridade_declaracao_direitos_humanos_bertimbau)
            # dados_p_df['similaridade_direitos_humanos_albertina'].append(dados_redacao.indicadores_texto.similaridade_declaracao_direitos_humanos_albertina)
            # dados_p_df['similaridade_direitos_humanos_bertugues'].append(dados_redacao.indicadores_texto.similaridade_declaracao_direitos_humanos_bertugues)
            dados_p_df['similaridade_tit_1_constituicao_bertimbau'].append(dados_redacao.indicadores_texto.similaridade_cap_1_constituicao_88_bertimbau)
            # dados_p_df['similaridade_tit_1_constituicao_albertina'].append(dados_redacao.indicadores_texto.similaridade_cap_1_constituicao_88_albertina)
            # dados_p_df['similaridade_tit_1_constituicao_bertugues'].append(dados_redacao.indicadores_texto.similaridade_cap_1_constituicao_88_bertugues)
            
            tempo_iter = time.time() - tempo_inicio
            tempo_restante = tempo_iter * (1168 - contador_id)

            horas = int(tempo_restante // 3600)
            minutos = int((tempo_restante % 3600) // 60)
            segundos = tempo_restante % 60

            print(f"Iteração {contador_id} | T(s): {tempo_iter:.2f}s | Tempo restante: {tempo_restante:.2f}s ({horas}h {minutos}m {segundos:.2f}s)")
            
            # print(f"Tempo levado: {time.time() - tempo_inicio}, Tempo restante: {(time.time() - tempo_inicio) * (1168 - contador_id)}")
            tempo_inicio = time.time()

            if contador_id % 3 == 0:
                df = pd.DataFrame(dados_p_df)

                df.to_csv('indicadores.csv')

    df = pd.DataFrame(dados_p_df)

    df.to_csv('indicadores.csv')
