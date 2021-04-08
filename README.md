# Listas de Exercícios - Aprendizagem de Máquina (2020.1)

## Lista 1

* Algoritmo de Aprendizagem: **Classificadores k-NN**
    * k = {1,2,3,5,7,9,11,13,15}
    * variações utilizadas:
        * sem peso (usual)
        * com peso
        * adaptativo

* Algoritmo de Cálculo da Distância: **Distância Euclidiana**

## Lista 2

* Algoritmos de Seleção de Protótipos: **LVQ1**, **LVQ2.1** e **LVQ3**
    * variações utilizadas:
        * número de protótipos = {5,10,15}

* Algoritmo de Aprendizagem: **Classificador k-NN**
    * k = {1,3}

* Algoritmo de Cálculo da Distância: **Distância Euclidiana**

## Lista 3

* Problema Proposto: **Bug Detector System Trained on the Majority Class only**

* Solução Desenvolvida: **Ensembles for One-class Classification**

    * Algoritmos de Aprendizagem:
        * **One Class Classifier SVM**
            * kernel = rbf
            * gamma = 0.001
            * nu = {0.25, 0.5, 0.95}
        * **Isolation Forest**
            * contamination = % of the minority class

## Lista 4

* Algoritmos de Aprendizagem:
    * **k-NN** 
        * k = 1
    * **Naive Bayes**
        * sem agrupamento
        * com agrupamento

* Algoritmo de Agrupamento: **k-Means**
    * k = {2,3,4,5,6}


# Datasets 

* Título/Tópico: **DATATRIEVE Transition/Software defect prediction (version 6.1)**
    * Doador: Guenther Ruhe (ruhe@ucalgary.ca)
    * Data: Janeiro 15, 2005
    * Número de Instâncias: 130
    * Número de Atributos: 
        * 8 atributos condicionais
        * 1 atributo decisor
    * Número de Valores NULL: 0
    * Número de Classes: 2 ({0,1})
    * Distribuição das Classes:
        * Classe 0: 119 instâncias (91.54%)
        * Classe 1: 11  instâncias (8.46%)
    * Fontes:
        * Criadores:
DATATRIEVETM project carried out at Digital Engineering Italy

* Título/Tópico: **KC2/Software defect prediction**
    * Doador: Tim Menzies (tim@barmag.net)
    * Data: December 2, 2004
    * Número de Instâncias: 522
    * Número de Atributos: 
        * 21 atributos condicionais
        * 1 atributo decisor
    * Número de Valores NULL: 0
    * Número de Classes: 2 ({yes,no})
    * Distribuição das Classes:
        * Classe yes: 107 instâncias (20.5%)
        * Classe no: 415  instâncias (79.5%)
    * Fontes:
        * Criadores:
NASA, then the NASA Metrics Data Program,
http://mdp.ivv.nasa.gov.

* Título/Tópico: **PC1/Software defect prediction**
    * Doador: Tim Menzies (tim@barmag.net)
    * Data: December 2, 2004
    * Número de Instâncias: 1109
    * Número de Atributos: 
        * 21 atributos condicionais
        * 1 atributo decisor
    * Número de Valores NULL: 0
    * Número de Classes: 2 ({false,true})
    * Distribuição das Classes:
        * Classe true: 77 instâncias (6.94%)
        * Classe false: 1032  instâncias (93.05%)
    * Fontes:
        * Criadores:
NASA, then the NASA Metrics Data Program,
http://mdp.ivv.nasa.gov.

* Título/Tópico: **CM1/Software defect prediction**
    * Doador: Tim Menzies (tim@barmag.net)
    * Data: December 2, 2004
    * Número de Instâncias: 498
    * Número de Atributos: 
        * 21 atributos condicionais
        * 1 atributo decisor
    * Número de Valores NULL: 0
    * Número de Classes: 2 ({false,true})
    * Distribuição das Classes:
        * Classe false: 449 instâncias (90.16%)
        * Classe true: 49  instâncias (9.83%)
    * Fontes:
        * Criadores:
NASA, then the NASA Metrics Data Program,
http://mdp.ivv.nasa.gov.

**OBS:** Os datasets utilizados são públicos e foram retirados do "Promise Software Engineering Repository" (http://promise.site.uottawa.ca/SERepository/datasets-page.html). Para a seleção das amostras de treino e teste, foi utilizado o método k-fold cross-validation com k = 10.
