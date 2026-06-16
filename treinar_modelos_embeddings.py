from sklearn.linear_model import LinearRegression, Ridge, RidgeClassifier, RidgeClassifierCV, RidgeCV
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, Lasso, LassoCV, LassoLars, LassoLarsCV, LassoLarsIC
from sklearn.linear_model import ElasticNet, ElasticNetCV, Lars, LarsCV
from sklearn.linear_model import OrthogonalMatchingPursuit, OrthogonalMatchingPursuitCV, BayesianRidge, ARDRegression
from sklearn.linear_model import TweedieRegressor, SGDClassifier
from sklearn.linear_model import Perceptron, PassiveAggressiveClassifier
from sklearn.linear_model import TheilSenRegressor, HuberRegressor

from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import BernoulliNB, GaussianNB

from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import StackingClassifier, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor, VotingRegressor, VotingClassifier

import pandas as pd
import numpy as np
import statistics
import joblib

from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler

import scikitplot as skplt
import matplotlib.pyplot as plt

from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, cohen_kappa_score, make_scorer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import cross_validate, cross_val_score, train_test_split

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

from mord import LogisticAT, LogisticIT, OrdinalRidge, LAD

modelos_regressores = {}
modelos_classificadores = {}

modelos_que_sao_normalizados = ["LinearSVR", "LinearSVC", "MLPClassifier", "MLPRegressor", "KNeighbors", "MLP", "Ridge"]

def popular_modelos_lineares():
    global modelos_regressores, modelos_classificadores

    alphas = [100, 50, 10, 5, 1, 0.5, 0.1, 0.005, 0.001, 0.0005, 0.0001]

    # N é linear mas é uma variação do Ridge
    modelos_regressores["KernelRidge"] = KernelRidge()

    # Regressores
    modelos_regressores["LinearRegression"] = LinearRegression()
    
    modelos_regressores[f"OrthogonalMatchingPursuitCV"] = OrthogonalMatchingPursuitCV(cv=5)
    modelos_regressores[f"HuberRegressor"] = HuberRegressor(max_iter=4000)
    # -- lento! modelos_regressores[f"ARDRegression"] = ARDRegression()

    modelos_regressores["RidgeC1"] = Ridge(alpha=5, solver='auto') 
    modelos_regressores["RidgeC2"] = Ridge(alpha=1, solver='lsqr')
    modelos_regressores["RidgeC3"] = Ridge(alpha=0.5, solver='auto')
    modelos_regressores["RidgeC4"] = Ridge(alpha=0.5, solver='auto')
    modelos_regressores["RidgeC5"] = Ridge(alpha=1, solver='auto')

    modelos_regressores["LassoC1"] = Lasso(alpha=0.005, selection='random')
    modelos_regressores["LassoC2"] = Lasso(alpha=0.001, selection='cyclic')
    modelos_regressores["LassoC3"] = Lasso(alpha=0.005, selection='random')
    modelos_regressores["LassoC4"] = Lasso(alpha=0.005, selection='cyclic')
    modelos_regressores["LassoC5"] = Lasso(alpha=0.005, selection='random')

    modelos_regressores["LassoLarsC1"] = LassoLars(alpha=0.005)
    modelos_regressores["LassoLarsC2"] = LassoLars(alpha=0.001)
    modelos_regressores["LassoLarsC3"] = LassoLars(alpha=0.005)
    modelos_regressores["LassoLarsC4"] = LassoLars(alpha=0.005)
    modelos_regressores["LassoLarsC5"] = LassoLars(alpha=0.005)

    modelos_regressores["ElasticNetC1"] = ElasticNet(alpha=0.0001, l1_ratio=0.75)
    modelos_regressores["ElasticNetC2"] = ElasticNet(alpha=0.0005, l1_ratio=0.75)
    modelos_regressores["ElasticNetC3"] = ElasticNet(alpha=0.0001, l1_ratio=0.75)
    modelos_regressores["ElasticNetC4"] = ElasticNet(alpha=0.0001, l1_ratio=0.75)
    modelos_regressores["ElasticNetC5"] = ElasticNet(alpha=0.0001, l1_ratio=0.25)

    modelos_regressores["BayesianRidgeC1"] = BayesianRidge(alpha_init=100)
    modelos_regressores["BayesianRidgeC2"] = BayesianRidge(alpha_init=100)
    modelos_regressores["BayesianRidgeC3"] = BayesianRidge(alpha_init=100)
    modelos_regressores["BayesianRidgeC4"] = BayesianRidge(alpha_init=100)
    modelos_regressores["BayesianRidgeC5"] = BayesianRidge(alpha_init=100)

    modelos_regressores["TweedieRegressorC1"] = TweedieRegressor(alpha=0.005,power=1,solver='newton-cholesky')
    modelos_regressores["TweedieRegressorC2"] = TweedieRegressor(alpha=0.0005, power=1, solver='newton-cholesky')
    modelos_regressores["TweedieRegressorC3"] = TweedieRegressor(alpha=0.0005, power=1, solver='newton-cholesky')
    modelos_regressores["TweedieRegressorC4"] = TweedieRegressor(alpha=0.0001, power=1, solver='newton-cholesky')
    modelos_regressores["TweedieRegressorC5"] = TweedieRegressor(alpha=0.005, power=1, solver='newton-cholesky')

    # Classificadores
    modelos_classificadores["LogisticRegressionCV"] = LogisticRegressionCV(class_weight="balanced")
    modelos_classificadores["RidgeClassifierCV"] = RidgeClassifierCV(class_weight="balanced")
    modelos_classificadores["SGDClassifier"] = SGDClassifier(class_weight="balanced")
    modelos_classificadores["Perceptron"] = Perceptron(class_weight="balanced")

def popular_modelos_support_vector():
    global modelos_regressores, modelos_classificadores

    modelos_regressores["LinearSVR"] = LinearSVR()
    modelos_classificadores["LinearSVC"] = LinearSVC(class_weight='balanced')

def popular_modelos_knn():
    global modelos_classificadores, modelos_regressores

    modelos_classificadores['KNeighborsClassifierC1'] = KNeighborsClassifier(n_neighbors=5, weights='distance')
    modelos_classificadores['KNeighborsClassifierC2'] = KNeighborsClassifier(n_neighbors=5, weights='uniform')
    modelos_classificadores['KNeighborsClassifierC3'] = KNeighborsClassifier(n_neighbors=5, weights='uniform')
    modelos_classificadores['KNeighborsClassifierC4'] = KNeighborsClassifier(n_neighbors=5, weights='uniform')
    modelos_classificadores['KNeighborsClassifierC5'] = KNeighborsClassifier(n_neighbors=13, weights='distance')

    modelos_regressores['KNeighborsRegressorC1'] = KNeighborsRegressor(n_neighbors=5, weights='uniform')
    modelos_regressores['KNeighborsRegressorC2'] = KNeighborsRegressor(n_neighbors=5, weights='uniform')
    modelos_regressores['KNeighborsRegressorC3'] = KNeighborsRegressor(n_neighbors=5, weights='distance')
    modelos_regressores['KNeighborsRegressorC4'] = KNeighborsRegressor(n_neighbors=5, weights='uniform')
    modelos_regressores['KNeighborsRegressorC5'] = KNeighborsRegressor(n_neighbors=9, weights='distance')
    
def popular_modelos_gaussianos():
    global modelos_classificadores, modelos_regressores

    modelos_classificadores["GaussianProcessClassifier"] = GaussianProcessClassifier()
    modelos_regressores["GaussianProcessRegressor"] = GaussianProcessRegressor()

def popular_modelos_naive_bayes():
    global modelos_regressores, modelos_classificadores

    modelos_classificadores["GaussianNB"] = GaussianNB()
    modelos_classificadores[f"BernoulliNB"] = BernoulliNB()

def popular_modelos_arvores_decisao():
    global modelos_classificadores, modelos_regressores

    modelos_classificadores["DecisionTreeClassifier"] = DecisionTreeClassifier(class_weight='balanced')
    modelos_regressores["DecisionTreeRegressor"] = DecisionTreeRegressor()

def popular_modelos_redes_neurais():
    global modelos_classificadores, modelos_regressores

    modelos_classificadores["MLPClassifierC1"] = MLPClassifier(alpha=0.0001, batch_size=64, hidden_layer_sizes=(250, 150, 100, 50), solver='adam', early_stopping=True)
    modelos_classificadores["MLPClassifierC2"] = MLPClassifier(alpha=0.1, batch_size=64, hidden_layer_sizes=(250, 100, 50), solver='adam', early_stopping=True)
    modelos_classificadores["MLPClassifierC3"] = MLPClassifier(alpha=0.1, batch_size=64, hidden_layer_sizes=(200, 100, 50), solver='adam', early_stopping=True)
    modelos_classificadores["MLPClassifierC4"] = MLPClassifier(alpha=0.1, batch_size=64, hidden_layer_sizes=(200, 100, 50), solver='adam', early_stopping=True)
    modelos_classificadores["MLPClassifierC5"] = MLPClassifier(alpha=0.0001, batch_size=16, hidden_layer_sizes=(250, 150, 100, 50), solver='adam', early_stopping=True)

    modelos_regressores["MLPRegressorC1"] = MLPRegressor(alpha=0.1, batch_size=16, hidden_layer_sizes=(250, 150, 100, 50), solver='adam', early_stopping=True)
    modelos_regressores["MLPRegressorC2"] = MLPRegressor(alpha=0.1, batch_size=16, hidden_layer_sizes=(200, 100, 50), solver='adam', early_stopping=True)
    modelos_regressores["MLPRegressorC3"] = MLPRegressor(alpha=0.1, batch_size=16, hidden_layer_sizes=(250, 150, 100, 50), solver='adam', early_stopping=True)
    modelos_regressores["MLPRegressorC4"] = MLPRegressor(alpha=100, batch_size=16, hidden_layer_sizes=(250, 150, 100, 50), solver='adam', early_stopping=True)
    modelos_regressores["MLPRegressorC5"] = MLPRegressor(alpha=0.1, batch_size=64, hidden_layer_sizes=(250, 150, 100, 50), solver='lbfgs', early_stopping=True)

def popular_outros_modelos():
    global modelos_classificadores, modelos_regressores

    modelos_classificadores["HistGradientBoostingClassifierC1"] = HistGradientBoostingClassifier(learning_rate=0.2, max_leaf_nodes=31, class_weight="balanced")
    modelos_classificadores["HistGradientBoostingClassifierC2"] = HistGradientBoostingClassifier(learning_rate=0.2, max_leaf_nodes=45, class_weight="balanced")
    modelos_classificadores["HistGradientBoostingClassifierC3"] = HistGradientBoostingClassifier(learning_rate=0.2, max_leaf_nodes=15, class_weight="balanced")
    modelos_classificadores["HistGradientBoostingClassifierC4"] = HistGradientBoostingClassifier(learning_rate=0.2, max_leaf_nodes=45, class_weight="balanced")
    modelos_classificadores["HistGradientBoostingClassifierC5"] = HistGradientBoostingClassifier(learning_rate=0.2, max_leaf_nodes=45, class_weight="balanced")

    modelos_classificadores["AdaBoostClassifierC1"] = AdaBoostClassifier(learning_rate=0.5, n_estimators=100)
    modelos_classificadores["AdaBoostClassifierC2"] = AdaBoostClassifier(learning_rate=0.5, n_estimators=50)
    modelos_classificadores["AdaBoostClassifierC3"] = AdaBoostClassifier(learning_rate=1, n_estimators=10)
    modelos_classificadores["AdaBoostClassifierC4"] = AdaBoostClassifier(learning_rate=0.5, n_estimators=100)
    modelos_classificadores["AdaBoostClassifierC5"] = AdaBoostClassifier(learning_rate=0.5, n_estimators=150)

    modelos_classificadores["RandomForestClassifierC1"] = RandomForestClassifier(max_features='sqrt', n_estimators=150, class_weight="balanced")
    modelos_classificadores["RandomForestClassifierC2"] = RandomForestClassifier(max_features='sqrt', n_estimators=100, class_weight="balanced")
    modelos_classificadores["RandomForestClassifierC3"] = RandomForestClassifier(max_features=1, n_estimators=200, class_weight="balanced")
    modelos_classificadores["RandomForestClassifierC4"] = RandomForestClassifier(max_features=0.3, n_estimators=50, class_weight="balanced")
    modelos_classificadores["RandomForestClassifierC5"] = RandomForestClassifier(max_features=1, n_estimators=150, class_weight="balanced")

    modelos_regressores["HistGradientBoostingRegressorC1"] = HistGradientBoostingRegressor(learning_rate=0.1, max_leaf_nodes=15)
    modelos_regressores["HistGradientBoostingRegressorC2"] = HistGradientBoostingRegressor(learning_rate=0.05, max_leaf_nodes=31)
    modelos_regressores["HistGradientBoostingRegressorC3"] = HistGradientBoostingRegressor(learning_rate=0.2, max_leaf_nodes=15)
    modelos_regressores["HistGradientBoostingRegressorC4"] = HistGradientBoostingRegressor(learning_rate=0.05, max_leaf_nodes=31)
    modelos_regressores["HistGradientBoostingRegressorC5"] = HistGradientBoostingRegressor(learning_rate=0.05, max_leaf_nodes=45)
    
    modelos_regressores["AdaBoostRegressorC1"] = AdaBoostRegressor(learning_rate=0.5, n_estimators=50)
    modelos_regressores["AdaBoostRegressorC2"] = AdaBoostRegressor(learning_rate=0.5, n_estimators=150)
    modelos_regressores["AdaBoostRegressorC3"] = AdaBoostRegressor(learning_rate=0.5, n_estimators=150)
    modelos_regressores["AdaBoostRegressorC4"] = AdaBoostRegressor(learning_rate=0.5, n_estimators=100)
    modelos_regressores["AdaBoostRegressorC5"] = AdaBoostRegressor(learning_rate=0.5, n_estimators=10)

    modelos_regressores["RandomForestRegressorC1"] = RandomForestRegressor(max_features=0.3, n_estimators=150)
    modelos_regressores["RandomForestRegressorC2"] = RandomForestRegressor(max_features='sqrt', n_estimators=200)
    modelos_regressores["RandomForestRegressorC3"] = RandomForestRegressor(max_features=0.3, n_estimators=100)
    modelos_regressores["RandomForestRegressorC4"] = RandomForestRegressor(max_features=0.3, n_estimators=50)
    modelos_regressores["RandomForestRegressorC5"] = RandomForestRegressor(max_features='sqrt', n_estimators=150)

modelos_mord = ["LAD", "LogisticIT", "LogisticAT", "OrdinalRidge"]

def popular_modelos_mord():
    global modelos_classificadores, modelos_regressores

#    alphas = [100, 10, 1, 0.1, 0.01]

    modelos_regressores[f"LAD"] = LAD()

    modelos_classificadores["LogisticIT"] = LogisticIT(alpha=100)
    modelos_classificadores["LogisticAT"] = LogisticAT(alpha=100)

    modelos_regressores["OrdinalRidge"] = OrdinalRidge(alpha=100)

# modelo_stack = None

def popular_modelos():
    global modelos_classificadores, modelos_regressores
    popular_modelos_lineares()
    popular_modelos_support_vector()
    popular_modelos_knn()
    popular_modelos_gaussianos()
    popular_modelos_naive_bayes()
    popular_modelos_arvores_decisao()
    popular_modelos_redes_neurais()
    popular_modelos_mord()


# arquivo_dataset = "indicadores_atualizados.csv"
arquivo_dataset = "embeddings_bertimbau.csv"

def qwk(Y_real, Y_previsto):
    return cohen_kappa_score(Y_real, Y_previsto, weights="quadratic")

# Scorers, parâmetros e modelos pra fazer o GridSearch

def qwk_arredondado(y_true, y_pred):
    y_pred_class = np.clip(np.round(np.array(y_pred) / 40) * 40, 0, 200).astype(int)
    
    return cohen_kappa_score(y_true, y_pred_class, weights="quadratic")

kappa_scorer = make_scorer(qwk_arredondado, greater_is_better=True)

parametros_ridge = {
                'alpha': [100, 50, 10, 5, 1, 0.5, 0.1, 0.005, 0.001, 0.0005, 0.0001],
                'solver': ('auto', 'svd', 'cholesky', 'sparse_cg', 'lsqr', 'sag')
              }

parametros_lasso = {
    'selection': ('cyclic', 'random'),
    'alpha': [100, 50, 10, 5, 1, 0.5, 0.1, 0.005, 0.001, 0.0005, 0.0001]
}

parametros_alfa_simples = {
                'alpha': [100, 50, 10, 5, 1, 0.5, 0.1, 0.005, 0.001, 0.0005, 0.0001]
}

parametros_elasticnet = {
    'alpha': [100, 50, 10, 5, 1, 0.5, 0.1, 0.005, 0.001, 0.0005, 0.0001],
    'l1_ratio': [0.25, 0.5, 0.75]
}
    
parametros_bayesianridge = {
    'alpha_init': [100, 50, 10, 5, 1, 0.5, 0.1, 0.005, 0.001, 0.0005, 0.0001]
}

parametros_tweedie = {
    'alpha': [100, 50, 10, 5, 1, 0.5, 0.1, 0.005, 0.001, 0.0005, 0.0001],
    'power': [0, 1],
    'solver': ['lbfgs', 'newton-cholesky']
}

parametros_knn = {
    'weights': ['uniform', 'distance'],
    'n_neighbors': list(range(1,41,4))
}

parametros_mlp = {
    'alpha': [100, 0.1, 0.0001],
    'hidden_layer_sizes': [
        (100,),
        (100, 50),
        (200, 100, 50),
        (250, 150, 100, 50)
    ],
    'solver': ['adam', 'lbfgs'],
    'batch_size': [16, 64]
}

parametros_gradientboost = {
    'max_leaf_nodes': [15, 31, 45],
    'learning_rate': [0.2, 0.1, 0.05]
}

parametros_adaboost = {
    'n_estimators': [10, 50, 100, 150],
    'learning_rate': [0.5, 1, 5]
}

parametros_randomforest = {
    'n_estimators': [50, 100, 150, 200],
    'max_features': [1, 'sqrt', 0.3]
}

ridge = Ridge()
lasso = Lasso()
lassolars = LassoLars()
elasticnet = ElasticNet()
bayesianridge = BayesianRidge()
tweedie = TweedieRegressor()
knn_classificador = KNeighborsClassifier()
knn_regressor = KNeighborsRegressor()
mlp_classificador = MLPClassifier()
mlp_regressor = MLPRegressor()

gradientboost_classificador = HistGradientBoostingClassifier(class_weight='balanced')
gradientboost_regressor = HistGradientBoostingRegressor()
adaboost_classificador = AdaBoostClassifier()
adaboost_regressor = AdaBoostRegressor()
randomforest_classificador = RandomForestClassifier()
randomforest_regressor = RandomForestRegressor()
mord_logisticit = LogisticIT()
mord_logisticat = LogisticAT()
mord_ordinalridge = OrdinalRidge()

gridsearch_modelos = {
    "Ridge": GridSearchCV(estimator=ridge, param_grid=parametros_ridge, scoring=kappa_scorer),
    "Lasso": GridSearchCV(estimator=lasso, param_grid=parametros_lasso, scoring=kappa_scorer),
    "LassoLars": GridSearchCV(estimator=lassolars, param_grid=parametros_alfa_simples, scoring=kappa_scorer),
    "ElasticNet": GridSearchCV(estimator=elasticnet, param_grid=parametros_elasticnet, scoring=kappa_scorer),
    "BayesianRidge": GridSearchCV(estimator=bayesianridge, param_grid=parametros_bayesianridge, scoring=kappa_scorer),
    "TweedieRegressor": GridSearchCV(estimator=tweedie, param_grid=parametros_tweedie, scoring=kappa_scorer),
    "KNeighborsClassifier": GridSearchCV(estimator=knn_classificador, param_grid=parametros_knn, scoring=kappa_scorer),
    "KNeighborsRegressor": GridSearchCV(estimator=knn_regressor, param_grid=parametros_knn, scoring=kappa_scorer),
    "MLPClassifier": GridSearchCV(estimator=mlp_classificador, param_grid=parametros_mlp, scoring=kappa_scorer),
    "MLPRegressor": GridSearchCV(estimator=mlp_regressor, param_grid=parametros_mlp, scoring=kappa_scorer),
    "HistGradientBoostingClassifier": GridSearchCV(estimator=gradientboost_classificador, param_grid=parametros_gradientboost, scoring=kappa_scorer),
    "HistGradientBoostingRegressor": GridSearchCV(estimator=gradientboost_regressor, param_grid=parametros_gradientboost, scoring=kappa_scorer), 
    "AdaBoostClassifier": GridSearchCV(estimator=adaboost_classificador, param_grid=parametros_adaboost, scoring=kappa_scorer), 
    "AdaBoostRegressor": GridSearchCV(estimator=adaboost_regressor, param_grid=parametros_adaboost, scoring=kappa_scorer), 
    "RandomForestClassifier": GridSearchCV(estimator=randomforest_classificador, param_grid=parametros_randomforest, scoring=kappa_scorer), 
    "RandomForestRegressor": GridSearchCV(estimator=randomforest_regressor, param_grid=parametros_randomforest, scoring=kappa_scorer),
    "LogisticIT": GridSearchCV(estimator=mord_logisticit, param_grid=parametros_alfa_simples, scoring=kappa_scorer),
    "LogisticAT": GridSearchCV(estimator=mord_logisticat, param_grid=parametros_alfa_simples, scoring=kappa_scorer),
    "OrdinalRidge": GridSearchCV(estimator=mord_ordinalridge, param_grid=parametros_alfa_simples, scoring=kappa_scorer)
}

def carregar_dados(arquivo_dataset: str) -> pd.DataFrame | None:
    try:
        dados = pd.read_csv(arquivo_dataset)
        print("Dados carregados")
        return dados
    except Exception as e:
        print(f"Erro: {e}")

import ast

def dividir_dados(dados: pd.DataFrame, coluna_alvo: str, classificador = False):
    """
        Divide os dados em X_treino, X_teste, Y_treino, Y_teste
    """

    X = dados['embedding'].apply(lambda x: np.array(ast.literal_eval(x)))
    X = np.vstack(X.values)

    # X = dados.drop(columns=["id","nota_total","nota_c1","nota_c2",
    #                         "nota_c3","nota_c4","nota_c5",
    #                         # "similaridade_tit_1_constituicao_bertugues",
    #                         # "similaridade_proposta_albertina",
    #                         # "similaridade_proposta_bertugues",
    #                         # "similaridade_direitos_humanos_albertina",
    #                         # "similaridade_direitos_humanos_bertugues",
    #                         # "similaridade_tit_1_constituicao_albertina",
    #                         # "n_verbos_e_pronomes_1ps_tok", # EXPERIMENTO
    #                         # "n_pron_dem_tok",
    #                         # "n_enclises_tok",
    #                         # "n_erros_estilo_num_sent",
    #                         # "n_erros_gramatica_tok",
    #                         # "n_erros_ortografia_tok",
    #                         # "n_marcadores_discursivos_num_sent"
    #                         ]) # removendo colunas que nao sao indicadores
    
    Y = dados[coluna_alvo]

    estratificar = None
    if classificador:
        estratificar = Y

    

    # 17.93% é o tamanho da divisão do dataset original
    X_test, y_test = X[:209], Y.iloc[:209]
    X_train, y_train = X[209:], Y.iloc[209:]

    print(f"Tamanhos das divisoes: Train={len(X_train)}, Test={len(X_test)}")
    return X_train, X_test, y_train, y_test

def treinar_modelos(X_treino, Y_treino, X_treino_normalizado, competencia=None):
    global modelos_classificadores
    global modelos_regressores
    global gridsearch_modelos

    pular_pois_gridsearch = set()

    for nome_modelo, modelo in gridsearch_modelos.items():
        print(f"Gridsearch: Treinando modelo {nome_modelo}")
        if any(nome_modelo.startswith(prefixo) for prefixo in modelos_que_sao_normalizados):
            modelo.fit(X_treino_normalizado, Y_treino)
        elif any(nome_modelo.startswith(prefixo) for prefixo in modelos_mord):
            Y_ordinal = Y_treino / 40
            Y_ordinal = Y_ordinal.astype(int)
            modelo.fit(X_treino_normalizado, Y_ordinal)
        else:
            modelo.fit(X_treino, Y_treino)

        melhor_modelo = modelo.best_estimator_
        nome_com_competencia = nome_modelo + competencia
        if nome_com_competencia in modelos_regressores.keys():
            modelos_regressores[nome_com_competencia] = melhor_modelo
        elif nome_com_competencia in modelos_classificadores.keys():
            modelos_classificadores[nome_com_competencia] = melhor_modelo
        
        pular_pois_gridsearch.add(nome_com_competencia)

    for nome_modelo in modelos_regressores.keys():
        if nome_modelo in pular_pois_gridsearch:
            continue
        if competencia is not None and any(nome_modelo.endswith(sufixo) for sufixo in finais_competencias):
            if not nome_modelo.endswith(competencia):
                continue
        try:
            if any(nome_modelo.startswith(prefixo) for prefixo in modelos_que_sao_normalizados):
                modelos_regressores[nome_modelo].fit(X_treino_normalizado, Y_treino)
            elif any(nome_modelo.startswith(prefixo) for prefixo in modelos_mord):
                Y_ordinal = Y_treino / 40
                Y_ordinal = Y_ordinal.astype(int)
                modelos_regressores[nome_modelo].fit(X_treino_normalizado, Y_ordinal)
            else:
                modelos_regressores[nome_modelo].fit(X_treino, Y_treino)
            print(f"Treinamento do modelo {nome_modelo} completo")
        except Exception as e:
            print(f"Erro no treinamento {nome_modelo}: {e}")

    for nome_modelo in modelos_classificadores.keys():
        if nome_modelo in pular_pois_gridsearch:
            continue
        if competencia is not None and any(nome_modelo.endswith(sufixo) for sufixo in finais_competencias):
            if not nome_modelo.endswith(competencia):
                continue
        try:
            if any(nome_modelo.startswith(prefixo) for prefixo in modelos_que_sao_normalizados):
                modelos_classificadores[nome_modelo].fit(X_treino_normalizado, Y_treino)
            elif any(nome_modelo.startswith(prefixo) for prefixo in modelos_mord):
                Y_ordinal = Y_treino / 40
                Y_ordinal = Y_ordinal.astype(int)
                modelos_classificadores[nome_modelo].fit(X_treino_normalizado, Y_ordinal)
            else:
                modelos_classificadores[nome_modelo].fit(X_treino, Y_treino)
            print(f"Treinamento do modelo {nome_modelo} completo")
        except Exception as e:
            print(f"Erro no treinamento {nome_modelo}: {e}")

def normalizar_dados(X_treino, X_teste):
    """
    Aplica a normalização Min-Max nos conjuntos de treino e teste.
    """
    scaler = MinMaxScaler()
    X_treino_normalizado = scaler.fit_transform(X_treino)
    X_teste_normalizado = scaler.transform(X_teste)
    print("Dados normalizados com sucesso.")
    return X_treino_normalizado, X_teste_normalizado

resultados_esperados = []
resultados_obtidos = []

valores_para_tabela = []

valores_votacao = [[None for _ in range(10)] for _ in range(5)]

import traceback

finais_competencias = ['C1', 'C2', 'C3', 'C4', 'C5']

def avaliar_modelos(X_teste, Y_teste, X_teste_normalizado, competencia=None):
    global valores_para_tabela, valores_votacao
    global modelos_regressores, modelos_classificadores, gridsearch_modelos, modelos_mord
    # Regressores
    valores_para_tabela = []

    id_competencia = None
    if competencia is not None:
        id_competencia = int(competencia.lower().replace("c","")) - 1

    resultados_qwk = []
    # k_melhores = 10

    for nome_modelo in modelos_regressores.keys():
        if competencia is not None and any(nome_modelo.endswith(sufixo) for sufixo in finais_competencias):
            if not nome_modelo.endswith(competencia):
                continue
        try:

            if any(nome_modelo.startswith(prefixo) for prefixo in modelos_que_sao_normalizados):
                y_previsao = modelos_regressores[nome_modelo].predict(X_teste_normalizado)
            elif any(nome_modelo.startswith(prefixo) for prefixo in modelos_mord):
                y_previsao = modelos_regressores[nome_modelo].predict(X_teste_normalizado)
                y_previsao = y_previsao * 40
                y_previsao = y_previsao.astype(int)
            else:
                y_previsao = modelos_regressores[nome_modelo].predict(X_teste)

            y_class = np.clip(np.round(np.array(y_previsao) / 40) * 40, 0, 200).astype(int)

            skplt.metrics.plot_confusion_matrix(Y_teste, y_class, normalize=True)
            plt.title(f"Matriz de confusão, {competencia}: {nome_modelo}")
            plt.savefig(f"graficos\\matrizes\\{competencia}-{nome_modelo}.png")
            plt.close('all')

            acc = accuracy_score(Y_teste, y_class)
            # prec = precision_score(Y_teste, y_class, average="weighted")
            # rec = recall_score(Y_teste, y_class, average="weighted")
            f1 = f1_score(Y_teste, y_class, average="weighted")
            kappa = qwk(Y_teste, y_class)
            mse = mean_squared_error(Y_teste, y_previsao)
            rmse = np.sqrt(mse)

            resultados_qwk.append((nome_modelo, kappa, modelos_regressores[nome_modelo], "regressor", rmse))
            valores_para_tabela.append((nome_modelo, kappa, rmse, acc, f1))
            print(f"{nome_modelo:<35} | QWK: {kappa:<5.2f} | RMSE: {rmse:<7.2f} | Acc.: {acc:<5.2f} | F1: {f1:<5.2f}")

        except Exception as e:
            print(f">> Erro: {e}")

    for nome_modelo in modelos_classificadores.keys():
        if competencia is not None and any(nome_modelo.endswith(sufixo) for sufixo in finais_competencias):
            if not nome_modelo.endswith(competencia):
                continue
        try:

            if any(nome_modelo.startswith(prefixo) for prefixo in modelos_que_sao_normalizados):
                y_previsao = modelos_classificadores[nome_modelo].predict(X_teste_normalizado)
            elif any(nome_modelo.startswith(prefixo) for prefixo in modelos_mord):
                y_previsao = modelos_classificadores[nome_modelo].predict(X_teste_normalizado)
                y_previsao = y_previsao * 40
                y_previsao = y_previsao.astype(int)
            else:
                y_previsao = modelos_classificadores[nome_modelo].predict(X_teste)

            acc = accuracy_score(Y_teste, y_previsao )
            # prec = precision_score(Y_teste, y_class, average="weighted")
            rec = recall_score(Y_teste, y_class, average="weighted")
            f1 = f1_score(Y_teste, y_previsao , average="weighted")
            kappa = qwk(Y_teste, y_previsao )
            mse = mean_squared_error(Y_teste, y_previsao)
            rmse = np.sqrt(mse)

            skplt.metrics.plot_confusion_matrix(Y_teste, y_class, normalize=True)
            plt.title(f"Matriz de confusão, {competencia}: {nome_modelo}")
            plt.savefig(f"graficos\\matrizes\\{competencia}-{nome_modelo}.png")
            plt.close('all')

            valores_para_tabela.append((nome_modelo, kappa, rmse, acc, f1))
            resultados_qwk.append((nome_modelo, kappa, modelos_classificadores[nome_modelo], "classificador", rmse))
            print(f"{nome_modelo:<35} | QWK: {kappa:<5.2f} | RMSE: {rmse:<7.2f} | Acc.: {acc:<5.2f} | F1: {f1:<5.2f}")

        except Exception as e:
            print(f">> Erro: {e}")

    print("GRID SEARCH")
    for nome_modelo, modelo in gridsearch_modelos.items():
        if any(nome_modelo.startswith(prefixo) for prefixo in modelos_que_sao_normalizados):
            y_previsao = modelo.predict(X_teste_normalizado)
        elif any(nome_modelo.startswith(prefixo) for prefixo in modelos_mord):
            y_previsao = modelo.predict(X_teste_normalizado)
            y_previsao = y_previsao * 40
            y_previsao = y_previsao.astype(int)
        else:
            y_previsao = modelo.predict(X_teste)
        y_previsao = np.clip(np.round(np.array(y_previsao) / 40) * 40, 0, 200).astype(int)
    
        acc = accuracy_score(Y_teste, y_previsao)
        # rec = recall_score(Y_teste, y_previsao, average="weighted")
        f1 = f1_score(Y_teste, y_previsao, average="weighted")
        kappa = qwk(Y_teste, y_previsao)
        mse = mean_squared_error(Y_teste, y_previsao)
        rmse = np.sqrt(mse)

        print(f"GridSearch: {nome_modelo} | QWK: {kappa:<5.2f} | RMSE: {rmse:<7.2f} | Acc.: {acc:<5.2f} | F1: {f1:<5.2f}")
        print(f"Melhores parametros para o {nome_modelo}: {modelo.cv_results_['params'][modelo.best_index_]}")

    resultados_qwk.sort(key=lambda x: x[1], reverse=True)
    # nome, kappa, modelo, tipo, rmse
    melhores_modelos = resultados_qwk # [:k_melhores]

    for nome, kappa, _, _, rmse in melhores_modelos:
        print(f"- {nome}: QWK {kappa:.3f} RMSE: {rmse:.2f}")

    print("\nVOTACAO")

    classes_validas = [0, 40, 80, 120, 160, 200]
    num_classes = len(classes_validas)
    num_amostras = len(Y_teste)

    for top_k in range(1, min(10, len(melhores_modelos)) + 1):
        modelos_usados = melhores_modelos[:top_k]

        preds_hard_por_modelo = []
        preds_soft_por_modelo = []

        for nome, _, modelo, tipo, rmse in modelos_usados:
            usa_normalizado = False
            for prefixo in modelos_que_sao_normalizados:
                if nome.startswith(prefixo):
                    usa_normalizado = True
                    break
            
            eh_mord = False
            for prefixo in modelos_mord:
                if nome.startswith(prefixo):
                    eh_mord = True
                    break

            if tipo == "regressor":
                if usa_normalizado or eh_mord:
                    y_pred_continuo = modelo.predict(X_teste_normalizado)
                else:
                    y_pred_continuo = modelo.predict(X_teste)
                if eh_mord:
                    y_pred_continuo *= 40

                preds_soft_por_modelo.append(y_pred_continuo)
                
                y_pred_classes = []
                for pred in y_pred_continuo:
                    valor = round(pred / 40.0) * 40
                    
                    if valor < 0:
                        valor = 0
                    if valor > 200:
                        valor = 200
                    
                    y_pred_classes.append(int(valor))
                preds_hard_por_modelo.append(y_pred_classes)

            elif tipo == "classificador":
                if usa_normalizado or eh_mord:
                    y_pred_classes = modelo.predict(X_teste_normalizado)
                else:
                    y_pred_classes = modelo.predict(X_teste)

                if eh_mord:
                    y_pred_classes *= 40
                
                preds_hard_por_modelo.append(y_pred_classes)
                preds_soft_por_modelo.append(y_pred_classes)

        votos_hard_por_amostra = []
        votos_soft_por_amostra = []

        for i in range(num_amostras):
            votos_hard_da_amostra_i = []
            votos_soft_da_amostra_i = []
            for j in range(top_k):
                votos_hard_da_amostra_i.append(preds_hard_por_modelo[j][i])
                votos_soft_da_amostra_i.append(preds_soft_por_modelo[j][i])
            votos_hard_por_amostra.append(votos_hard_da_amostra_i)
            votos_soft_por_amostra.append(votos_soft_da_amostra_i)

        y_final_hard = []
        for i in range(num_amostras):
            contagem = [0] * num_classes
            
            votos_da_amostra = votos_hard_por_amostra[i]
            for voto in votos_da_amostra:
                for idx in range(num_classes):
                    if classes_validas[idx] == voto:
                        contagem[idx] = contagem[idx] + 1
                        break
            
            melhor_idx = 0
            max_contagem = contagem[0]
            for idx in range(1, num_classes):
                if contagem[idx] > max_contagem:
                    max_contagem = contagem[idx]
                    melhor_idx = idx
            
            y_final_hard.append(classes_validas[melhor_idx])

        y_final_soft = []
        for i in range(num_amostras):
            votos_da_amostra = votos_soft_por_amostra[i]
            
            soma = 0
            for voto in votos_da_amostra:
                soma = soma + voto
            media = soma / len(votos_da_amostra)
            
            valor_arredondado = round(media / 40.0) * 40
            
            if valor_arredondado < 0:
                valor_final = 0
            elif valor_arredondado > 200:
                valor_final = 200
            else:
                valor_final = int(valor_arredondado)
                
            y_final_soft.append(valor_final)
        
        rmse_hard = np.sqrt(mean_squared_error(Y_teste, y_final_hard))
        rmse_soft = np.sqrt(mean_squared_error(Y_teste, y_final_soft))
        
        print(f"\nTop {top_k} modelos:")
        print("Hard Voting QWK:", qwk(Y_teste, y_final_hard))
        print("Hard Voting RMSE:", rmse_hard)
        print("Soft Voting QWK:", qwk(Y_teste, y_final_soft))
        print("Soft Voting RMSE:", rmse_soft)

        # skplt.metrics.plot_confusion_matrix(Y_teste, y_final_soft, normalize=True)
        # plt.title(f"Matriz de confusão, votação soft {competencia} com top {top_k}")
        # plt.savefig(f"graficos\\matrizes\\votacao-soft-{competencia}-{top_k}.png")
        # plt.close('all')

        # skplt.metrics.plot_confusion_matrix(Y_teste, y_final_hard, normalize=True)
        # plt.title(f"Matriz de confusão, votação hard {competencia} com top {top_k}")
        # plt.savefig(f"graficos\\matrizes\\votacao-hard-{competencia}-{top_k}.png")
        # plt.close('all')

        valores_votacao[id_competencia][top_k-1] = (qwk(Y_teste, y_final_hard), rmse_hard, qwk(Y_teste, y_final_soft), rmse_soft)

if __name__ == "__main__":
    
    dados = carregar_dados(arquivo_dataset)

    notas = ['nota_c1', 'nota_c2', 'nota_c3', 'nota_c4', 'nota_c5']

    for nota in notas:

        if 'c' in nota:
            competencia = nota.replace('nota_', '').upper()
        else:
            competencia = None

        print(f"=============================== INICIANDO TESTES COM {nota} ===============================")
        X_treino, X_teste, Y_treino, Y_teste = dividir_dados(dados, nota, True)

        X_treino_normalizado, X_teste_normalizado = normalizar_dados(X_treino, X_teste)

        print("Classes em Y_treino:", np.unique(Y_treino))
        print("Classes em Y_teste:", np.unique(Y_teste))
        print("Classes em todo Y:", np.unique(np.concatenate([Y_treino, Y_teste])))

        popular_modelos()
        
        treinar_modelos(X_treino, Y_treino, X_treino_normalizado, competencia)

        avaliar_modelos(X_teste, Y_teste, X_teste_normalizado, competencia)

        print(f"=============================== TESTES COM {nota} FINALIZADOS ===============================")

    print(valores_votacao)