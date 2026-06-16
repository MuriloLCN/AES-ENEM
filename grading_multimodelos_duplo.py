import ollama
import csv
import pandas as pd
from datetime import timedelta
import time

# 0            40     80                 120                160   200
# Muito ruim | Ruim | Regular negativo | Regular positivo | Bom | Muito Bom

notas = {
    "muito ruim": 0,
    "ruim": 1,
    "regular negativo": 2,
    "regular positivo": 3,
    "bom": 4,
    "muito bom": 5
}

def parse_grades(grade_str: str) -> list[int]:
    grade_str = grade_str.lower()
    # print(f"String: {grade_str}")
    try:
        return [notas[g] for g in grade_str.split(sep=';')]
    except:
        return None

def notas_originais(str_notas):
    str_notas = str_notas.replace('[','').replace(']','').strip()

    while "  " in str_notas:
        str_notas = str_notas.replace("  ", " ")
            
    notas = [int(valor) for valor in str_notas.split(" ")]
    return notas

def grade_essays():

    modelos = ['gemma4:e4b', 'qwen2.5:7b', 'qwen3:4b', 'deepseek-r1:8b']

    arquivos = ["test.csv", "train.csv", "validation.csv"]

    contador_id = 0

    essays_data = []

    for nome_arquivo in arquivos:
        with open(f'redacoes_enem_original/{nome_arquivo}', mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for row in reader:
                contador_id += 1

                essays_data.append({
                    'id': contador_id,
                    'essay': row['essay_text'],
                    'prompt': row['prompt'],
                    'grades_original': notas_originais(row['grades'])
                })

    results = []
    start_total_time = time.time()

    for model in modelos:
        model_start_time = time.time()
        print(f"\n=== Rodando modelo: {model} ===")

        for essay_data in essays_data:

            total_essays = len(essays_data)

            essay_start_time = time.time()
            i = essay_data['id']
            prompt_redacao = essay_data['prompt']
            essay = essay_data['essay']


            prompt_1 = f"""Você é um corretor especialista de redações do ENEM, aplicadas em alunos do ensino médio. Avalie a redação a seguir para cada uma das cinco competências:

            Competência I: Demonstrar domínio da modalidade escrita formal da língua portuguesa.
            Competência II: Compreender a proposta de redação e aplicar conceitos das várias áreas de conhecimento para desenvolver o tema dentro dos limites estruturais do texto dissertativo-argumentativo em prosa.
            Competência III: Selecionar, relacionar, organizar e interpretar informações, fatos, opiniões e argumentos em defesa de um ponto de vista. 
            Competência IV: Demonstrar conhecimento dos mecanismos linguísticos necessários para a construção da argumentação.
            Competência V: Elaborar proposta de intervenção para o problema abordado, respeitando os direitos humanos.

            Retorne cinco parágrafos, cada um contendo uma curta avaliação sobre o desempenho da redação para aquela competência.

            Proposta de redação/comando: {prompt_redacao}
            
            Redação: {essay}"""
            
            # print(prompt)

            while True:
                print(f"Model {model} - Essay {i}/{total_essays}", end="\r")

                response_1 = ollama.generate(model=model, prompt=prompt_1)
                output_1 = response_1['response'].strip()

                prompt_2 = f"""Você é um corretor especialista de redações do ENEM, aplicadas em alunos do ensino médio. Você receberá cinco parágrafos, cada um analisando uma redação com base nos seguintes critérios de avaliação:

                Competência I: Demonstrar domínio da modalidade escrita formal da língua portuguesa.
                Competência II: Compreender a proposta de redação e aplicar conceitos das várias áreas de conhecimento para desenvolver o tema dentro dos limites estruturais do texto dissertativo-argumentativo em prosa.
                Competência III: Selecionar, relacionar, organizar e interpretar informações, fatos, opiniões e argumentos em defesa de um ponto de vista. 
                Competência IV: Demonstrar conhecimento dos mecanismos linguísticos necessários para a construção da argumentação.
                Competência V: Elaborar proposta de intervenção para o problema abordado, respeitando os direitos humanos.

                Cada competência pode assumir as notas "muito ruim","ruim","regular negativo","regular positivo","bom" ou "muito bom", nenhum outro valor pode ser atribuído. Sua resposta deve ser apenas os cinco valores atribuídos separados por ';', nada mais. Por exemplo, uma redação de nota máxima em todas as competências seria 'muito bom;muito bom;muito bom;muito bom;muito bom', já uma redação com notas zero seria 'muito ruim;muito ruim;muito ruim;muito ruim;muito ruim'.
                Com base na avaliação qualitativa sobre a redação a seguir, retorne a avaliação das competências.

                Avaliação da redação: {output_1}
                """

                # print(output_1)

                response_2 = ollama.generate(model=model, prompt=prompt_2)
                output_2 = response_2['response'].strip()

                # print(output_2)

                try:
                    grades = parse_grades(output_2)
                    print(f"Notas {grades}")

                    if grades is None or len(grades) != 5:
                        raise ValueError("Número inválido de notas retornados")
                        
                    break

                except:
                    print(f"\nErro parse (Modelo {model}, Redação {i}): {output_2}")
                    print("Tentando novamente...")

            essay_data[f'{model}_grades'] = grades
            elapsed_essay = time.time() - essay_start_time
            print(f"[{model}] Essay {i}/{len(essays_data)} - Time: {elapsed_essay:.2f}s")

        model_elapsed = time.time() - model_start_time
        print(f"\n>>> Finished Model {model} in {str(timedelta(seconds=int(model_elapsed)))}")
    
    total_elapsed = time.time() - start_total_time
    print(f"\nTOTAL EXECUTION TIME: {str(timedelta(seconds=int(total_elapsed)))}")

    for essay_data in essays_data:
        row = {}

        row['id'] = essay_data['id']

        original = essay_data['grades_original']
        for i in range(5):
            row[f'nota_c{i+1}'] = original[i] if original else None

        # Model grades
        for model in modelos:
            model_grades = essay_data.get(f'{model}_grades', [None]*5)

            if model_grades is None or len(model_grades) < 5:
                model_grades = [None] * 5

            model_name = model.split(':')[0]  # clean name

            for i in range(5):
                row[f'nota_c{i+1}_{model_name}'] = model_grades[i]

        results.append(row)

    df = pd.DataFrame(results)

    df.to_csv('grades_comparison_double_r3.csv', index=False)

    print("Salvo em grades_comparison_double_r3.csv")


if __name__ == "__main__":
    grade_essays()