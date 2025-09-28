"""
Dataset Generator para Processo Seletivo PUC-Rio
Geração de dados sintéticos para demonstração do MVP de Machine Learning
"""

import pandas as pd
import numpy as np

def gerar_dataset_processo_seletivo(n_samples=2000, random_state=42):
    """
    Gera dataset sintético para o processo seletivo de pós-graduação PUC-Rio
    
    Parameters:
    -----------
    n_samples : int, default=2000
        Número de candidatos a serem gerados
    random_state : int, default=42
        Semente para reprodutibilidade
    
    Returns:
    --------
    pd.DataFrame
        DataFrame com os dados dos candidatos e target de aprovação
    """
    
    np.random.seed(random_state)
    
    print(f"🔄 Gerando dataset sintético do processo seletivo...")
    print(f"📊 Número de candidatos: {n_samples}")
    
    # Criando features do candidato
    data = {
        # Dados Acadêmicos
        'nota_graduacao': np.random.normal(8.2, 1.5, n_samples).clip(5.0, 10.0),
        'experiencia_profissional': np.random.randint(0, 15, n_samples),
        'publicacoes': np.random.poisson(2, n_samples),
        'projetos_pesquisa': np.random.poisson(1.5, n_samples),
        'pontuacao_prova': np.random.normal(75, 15, n_samples).clip(30, 100),
        'pontuacao_entrevista': np.random.normal(80, 12, n_samples).clip(40, 100),
        
        # Dados Categóricos - Programas
        'programa': np.random.choice([
            'Engenharia_Civil', 'Engenharia_Mecanica', 
            'Engenharia_Eletrica', 'Ciencia_Computacao', 
            'Matematica', 'Fisica'
        ], n_samples),
        
        'nivel_pretendido': np.random.choice([
            'Mestrado', 'Doutorado'
        ], n_samples, p=[0.7, 0.3]),
        
        'tipo_instituicao_origem': np.random.choice([
            'Publica', 'Privada', 'Federal'
        ], n_samples, p=[0.4, 0.4, 0.2]),
        
        'regiao_origem': np.random.choice([
            'Sudeste', 'Sul', 'Nordeste', 'Norte', 'Centro-Oeste'
        ], n_samples, p=[0.35, 0.25, 0.2, 0.1, 0.1]),
        
        'modalidade_candidatura': np.random.choice([
            'Regular', 'Cotas', 'Estrangeiro'
        ], n_samples, p=[0.8, 0.15, 0.05]),
        
        # Dados Socioeconômicos
        'idade': np.random.randint(22, 45, n_samples),
        'renda_familiar': np.random.lognormal(8.5, 1.2, n_samples).clip(1000, 50000),
        'ensino_ingles': np.random.choice([
            'Basico', 'Intermediario', 'Avancado'
        ], n_samples, p=[0.3, 0.5, 0.2]),
    }
    
    # Criando DataFrame
    df_candidatos = pd.DataFrame(data)
    
    # Lógica para gerar target (aprovado/rejeitado) baseada nas features
    def calcular_probabilidade_aprovacao(row):
        """Calcula probabilidade de aprovação baseada no perfil do candidato"""
        score = 0
        
        # Peso das notas acadêmicas (40%)
        score += (row['nota_graduacao'] - 5) / 5 * 0.25
        score += (row['pontuacao_prova'] - 30) / 70 * 0.15
        
        # Experiência e produção acadêmica (30%)
        score += min(row['experiencia_profissional'] / 10, 1) * 0.15
        score += min(row['publicacoes'] / 5, 1) * 0.1
        score += min(row['projetos_pesquisa'] / 3, 1) * 0.05
        
        # Entrevista (20%)
        score += (row['pontuacao_entrevista'] - 40) / 60 * 0.2
        
        # Bônus e fatores adicionais (10%)
        if row['nivel_pretendido'] == 'Doutorado' and row['publicacoes'] > 2:
            score += 0.05  # Doutorado com publicações
        if row['ensino_ingles'] == 'Avancado':
            score += 0.03  # Inglês avançado
        if row['programa'] in ['Ciencia_Computacao', 'Engenharia_Eletrica']:
            score += 0.02  # Programas com alta demanda
        
        return min(max(score, 0), 1)
    
    # Aplicando a lógica e criando target
    print("🎯 Calculando probabilidades de aprovação...")
    df_candidatos['prob_aprovacao'] = df_candidatos.apply(calcular_probabilidade_aprovacao, axis=1)
    
    # Gerando target binário com variabilidade
    aprovacao_threshold = np.random.normal(0.6, 0.1, n_samples)
    df_candidatos['aprovado'] = (df_candidatos['prob_aprovacao'] > aprovacao_threshold).astype(int)
    
    # Removendo coluna auxiliar
    df_candidatos = df_candidatos.drop('prob_aprovacao', axis=1)
    
    # Estatísticas do dataset
    taxa_aprovacao = df_candidatos['aprovado'].mean()
    
    print(f"✅ Dataset gerado com sucesso!")
    print(f"📏 Dimensões: {df_candidatos.shape}")
    print(f"🎯 Taxa de aprovação: {taxa_aprovacao:.2%}")
    print(f"📊 Distribuição por nível:")
    print(f"   • Mestrado: {(df_candidatos['nivel_pretendido'] == 'Mestrado').sum()}")
    print(f"   • Doutorado: {(df_candidatos['nivel_pretendido'] == 'Doutorado').sum()}")
    
    return df_candidatos

def salvar_dataset(df, caminho='dataset_processo_seletivo.csv'):
    """
    Salva o dataset em arquivo CSV
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset a ser salvo
    caminho : str
        Caminho para salvar o arquivo
    """
    df.to_csv(caminho, index=False, encoding='utf-8')
    print(f"💾 Dataset salvo em: {caminho}")

def carregar_dataset(caminho='dataset_processo_seletivo.csv'):
    """
    Carrega o dataset de um arquivo CSV
    
    Parameters:
    -----------
    caminho : str
        Caminho do arquivo CSV
    
    Returns:
    --------
    pd.DataFrame
        Dataset carregado
    """
    try:
        df = pd.read_csv(caminho, encoding='utf-8')
        print(f"📂 Dataset carregado de: {caminho}")
        print(f"📏 Dimensões: {df.shape}")
        print(f"🎯 Taxa de aprovação: {df['aprovado'].mean():.2%}")
        return df
    except FileNotFoundError:
        print(f"❌ Arquivo não encontrado: {caminho}")
        print("🔄 Gerando novo dataset...")
        return gerar_dataset_processo_seletivo()

if __name__ == "__main__":
    # Gerar e salvar dataset para uso no notebook
    df = gerar_dataset_processo_seletivo(n_samples=2000, random_state=42)
    salvar_dataset(df, 'dataset_processo_seletivo.csv')
    
    # Exibir primeiras linhas
    print("\n📋 PRIMEIRAS 5 LINHAS DO DATASET:")
    print(df.head())
    
    # Informações sobre colunas
    print("\n📊 INFORMAÇÕES DAS COLUNAS:")
    print(df.info())