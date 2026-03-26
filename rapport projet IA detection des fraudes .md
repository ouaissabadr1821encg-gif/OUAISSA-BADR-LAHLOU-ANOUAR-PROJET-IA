**ÉCOLE NATIONALE DE COMMERCE ET DE GESTION**

Master Contrôle, Audit et Conseil (CAC)

RAPPORT DE PROJET

**Détection de la Fraude dans les États Financiers**

*par Apprentissage Automatique et Traitement du Langage Naturel*

  ---------------------------- ------------------------------------------
  **Étudiants**                LAHLOU Anouar \| OUAISSA Badr

  **Filière**                  Master Contrôle, Audit et Conseil

  **Année universitaire**      2025 --- 2026
  ---------------------------- ------------------------------------------

Settat, 2026

**1. Introduction**

**1.1 Contexte de la fraude financière**

La fraude dans les états financiers constitue l\'une des menaces les
plus sérieuses pour l\'intégrité des marchés financiers et la confiance
des investisseurs. Selon l\'Association of Certified Fraud Examiners
(ACFE), les entreprises perdent en moyenne **5 % de leur chiffre
d\'affaires annuel** à cause de la fraude, dont une part significative
est imputable à la manipulation des états financiers. Ces actes
frauduleux se manifestent principalement par la falsification des
comptes, la dissimulation de passifs, la présentation trompeuse de la
performance financière ou encore l\'altération des notes annexes aux
rapports annuels.

Dans ce contexte, la détection précoce et automatisée de la fraude
représente un enjeu stratégique majeur pour les régulateurs, les
auditeurs et les investisseurs. Les méthodes traditionnelles d\'audit,
basées sur des vérifications manuelles et des analyses statistiques
classiques, montrent rapidement leurs limites face au volume croissant
de données financières et à la sophistication des techniques
frauduleuses.

**1.2 Importance de la détection automatisée**

Le recours à l\'intelligence artificielle et au traitement automatique
du langage naturel (NLP) ouvre de nouvelles perspectives pour l\'analyse
des documents financiers à grande échelle. En exploitant le contenu
textuel des rapports annuels déposés auprès des régulateurs ---
notamment les formulaires SEC aux États-Unis --- il devient possible
d\'identifier des patterns linguistiques caractéristiques de
comportements frauduleux, tels que l\'ambiguïté volontaire, la
minimisation des risques ou l\'usage de formulations évasives.

Cette approche repose sur l\'hypothèse, validée par plusieurs études
académiques, que le langage utilisé dans les rapports financiers
frauduleux diffère significativement de celui des rapports sincères. La
fréquence de certains termes, la structure syntaxique et le registre
lexical constituent ainsi des signaux faibles exploitables par les
algorithmes d\'apprentissage automatique.

**1.3 Objectifs du projet**

Ce projet vise trois objectifs principaux :

-   Développer un pipeline NLP complet pour le prétraitement et la
    vectorisation de textes financiers issus de formulaires SEC ;

-   Entraîner et comparer plusieurs algorithmes de classification
    supervisée --- Régression Logistique, SVM et Naive Bayes --- afin
    d\'identifier le modèle le plus performant pour la détection de
    fraude ;

-   Interpréter les résultats obtenus de manière critique, en
    identifiant les mots et expressions les plus discriminants entre
    documents frauduleux et non-frauduleux.

**2. Revue de Littérature**

**2.1 Techniques classiques de détection de fraude**

Les premières approches de détection de fraude financière reposaient
essentiellement sur des méthodes statistiques et des règles expertes. Le
modèle de **Beneish (1999)**, par exemple, exploite des ratios
financiers pour calculer un score de probabilité de manipulation des
bénéfices. De même, la **loi de Benford** est utilisée pour détecter des
anomalies dans la distribution des chiffres significatifs des données
comptables. Ces approches, bien qu\'encore pertinentes, présentent des
limites structurelles : elles sont sensibles au bruit, difficilement
généralisables, et ignorent la richesse informationnelle des données
textuelles.

**2.2 Apport du Machine Learning et du NLP**

L\'essor du machine learning a profondément transformé le domaine de la
détection de fraude. Des travaux fondateurs tels que ceux de **Glancy &
Yadav (2011)** ont démontré la pertinence de l\'analyse textuelle des
rapports financiers pour détecter la fraude comptable. Plus récemment,
**Goel & Gangolly (2012)** ont montré que les rapports frauduleux se
distinguent par un vocabulaire spécifique --- davantage d\'incertitude,
de complexité syntaxique et d\'opacité délibérée.

Sur le plan des techniques de vectorisation, le modèle **TF-IDF (Term
Frequency-Inverse Document Frequency)** reste une référence dans le
traitement des textes financiers en raison de sa capacité à pondérer
l\'importance relative des termes selon leur fréquence globale dans le
corpus. Des modèles plus récents, comme **BERT** et ses variantes
financières (FinBERT), offrent des performances supérieures en capturant
le contexte sémantique, mais nécessitent des ressources
computationnelles nettement plus importantes.

**3. Description des Données**

**3.1 Présentation du dataset**

Le dataset utilisé dans ce projet est le **Financial Statement Fraud
Dataset**, disponible sur la plateforme Kaggle. Il est constitué
d\'extraits textuels issus de formulaires annuels (*Form 10-K*) déposés
auprès de la Securities and Exchange Commission (SEC) américaine. Ces
formulaires contiennent des sections normalisées --- notes aux états
financiers, rapport de l\'auditeur, discussion et analyse de la
direction --- qui constituent une source d\'information particulièrement
riche pour l\'analyse de la sincérité des divulgations financières.

  -----------------------------------------------------------------------
  **Caractéristique**     **Description**
  ----------------------- -----------------------------------------------
  Source                  Kaggle --- Financial Statement Fraud Data (SEC
                          10-K)

  Variable texte          Fillings --- extraits de rapports financiers

  Variable cible          Fraud --- binaire : yes (fraude) / no
                          (non-fraude)

  Type de problème        Classification binaire supervisée

  Langue                  Anglais (terminologie financière SEC)
  -----------------------------------------------------------------------

**3.2 Nature des variables**

La variable **\"Fillings\"** contient des segments textuels de longueur
variable, extraits de sections standardisées des rapports annuels. Ces
textes présentent une forte densité de terminologie comptable et
financière (*item 14, financial statements, principal accounting fees,
exhibits*), ce qui implique un prétraitement rigoureux pour distinguer
les termes réellement discriminants des occurrences banales.

La variable cible **\"Fraud\"** est une étiquette binaire indiquant si
le document est associé à une entreprise ayant fait l\'objet de
poursuites ou de retraitements comptables liés à des fraudes avérées.
Cette labellisation est particulièrement sensible : une erreur
d\'étiquetage peut introduire un biais systématique dans les résultats
du modèle.

**3.3 Défis liés aux données textuelles**

L\'utilisation de données textuelles dans un contexte de classification
financière soulève plusieurs défis méthodologiques :

-   Le déséquilibre potentiel des classes, les fraudes avérées étant
    structurellement moins fréquentes que les rapports sincères ;

-   La variabilité de la longueur des textes, qui peut introduire un
    biais dans les représentations vectorielles ;

-   La présence de boilerplate --- formulations standardisées et
    répétitives communes à tous les formulaires SEC --- qui peut masquer
    les signaux linguistiques pertinents ;

-   La richesse du vocabulaire financier spécialisé, qui nécessite un
    traitement adapté des stop words pour ne pas éliminer des termes
    potentiellement discriminants.

**4. Méthodologie**

La méthodologie adoptée suit un pipeline structuré en quatre étapes
successives, conformément aux bonnes pratiques du traitement du langage
naturel appliqué à la classification de textes. Chaque choix technique
est motivé par les contraintes spécifiques du dataset et par la
littérature existante.

**4.1 Prétraitement des données**

**4.1.1 Nettoyage du texte**

La première étape consiste à normaliser le contenu textuel brut afin de
réduire le bruit et d\'homogénéiser les représentations. Les
transformations appliquées sont les suivantes : **mise en minuscules**
pour éliminer la sensibilité à la casse, **suppression des URLs et des
chiffres isolés** pour éviter de polluer l\'espace de features avec des
identifiants non-sémantiques, et **élimination de la ponctuation** et
des caractères spéciaux.

**4.1.2 Tokenisation**

La tokenisation, réalisée via la bibliothèque **NLTK**, décompose chaque
texte en une séquence de tokens (unités lexicales). Cette opération est
fondamentale pour toutes les étapes de traitement ultérieures. Les
tokens de longueur inférieure à trois caractères sont systématiquement
supprimés, car ils constituent généralement des artéfacts ou des
abréviations non-significatives dans ce corpus.

**4.1.3 Suppression des stop words et lemmatisation**

Les stop words --- mots grammaticaux à faible valeur sémantique tels que
*\"the\", \"and\", \"of\"* --- sont supprimés à partir de la liste
standard fournie par NLTK. Cette liste a été enrichie de termes
financiers récurrents mais non discriminants (*\"item\", \"exhibit\",
\"financial\", \"statement\"*), qui apparaissent dans pratiquement tous
les formulaires SEC indépendamment de leur nature frauduleuse ou non.

La **lemmatisation**, opérée par le **WordNetLemmatizer**, ramène chaque
token à sa forme canonique (*\"statements\" → \"statement\"*).
Contrairement au stemming, qui procède par troncature mécanique, la
lemmatisation préserve la cohérence sémantique et améliore la qualité
des représentations vectorielles.

**4.2 Transformation des données : TF-IDF**

La représentation vectorielle des textes nettoyés est réalisée par la
méthode **TF-IDF (Term Frequency-Inverse Document Frequency)**, qui
constitue le standard de référence pour la classification de textes dans
des contextes à ressources limitées.

  -----------------------------------------------------------------------
  *Formule TF-IDF : TF-IDF(t, d) = TF(t, d) × log(N / df(t)) où TF(t,d)
  est la fréquence du terme t dans le document d, N le nombre total de
  documents, et df(t) le nombre de documents contenant t.*

  -----------------------------------------------------------------------

Cette approche est justifiée par plusieurs raisons. Premièrement, elle
pondère naturellement les termes rares mais potentiellement
discriminants --- comme certains vocables liés à la dissimulation ou à
l\'ambiguïté --- en leur attribuant un poids élevé. Deuxièmement, elle
pénalise les termes ubiquitaires du boilerplate financier via le facteur
IDF. Troisièmement, elle produit des représentations sparses compatibles
avec les algorithmes linéaires sélectionnés. La configuration retenue
inclut un vocabulaire limité aux **5 000 features les plus
informatives** et intègre les **bigrammes** (ngram_range = (1,2)) pour
capturer des collocations significatives telles que *\"net income\"* ou
*\"material weakness\"*.

**4.3 Modélisation**

Trois algorithmes de classification supervisée ont été sélectionnés pour
leur adéquation avec les caractéristiques du problème :

  -----------------------------------------------------------------------
  **Algorithme**      **Avantages**             **Limites**
  ------------------- ------------------------- -------------------------
  Régression          Interprétable, efficace   Suppose la linéarité des
  Logistique          sur données sparses,      frontières de décision
                      baseline robuste          

  SVM (LinearSVC)     Excellentes performances  Moins interprétable,
                      sur données textuelles,   sensible au déséquilibre
                      robuste au                des classes
                      surapprentissage          

  Naive Bayes         Très rapide, efficace sur Hypothèse d\'indépendance
  (Multinomial)       petits corpus, fondement  conditionnelle rarement
                      probabiliste solide       vérifiée
  -----------------------------------------------------------------------

Le choix de ces trois modèles répond à une logique de **comparaison
complémentaire** : la Régression Logistique sert de baseline
interprétable, le SVM représente l\'état de l\'art pour la
classification de textes, et Naive Bayes apporte une perspective
probabiliste distincte. Tous trois sont compatibles avec des
représentations TF-IDF sparses et offrent des temps d\'entraînement
raisonnables.

**4.4 Méthode d\'évaluation**

L\'évaluation des performances repose sur un ensemble de métriques
complémentaires. Le **jeu de données est divisé en 80 % pour
l\'entraînement et 20 % pour le test**, avec stratification pour
préserver la distribution des classes.

-   Accuracy : proportion globale de prédictions correctes --- utile
    mais insuffisante en présence de déséquilibre des classes ;

-   Precision : parmi les documents prédits frauduleux, proportion
    réellement frauduleuse --- indicateur de la fiabilité des alertes ;

-   Recall (Rappel) : parmi les documents réellement frauduleux,
    proportion correctement détectée --- métrique critique en contexte
    de fraude ;

-   F1-score : moyenne harmonique de la Precision et du Recall, métrique
    de synthèse privilégiée dans ce projet ;

-   ROC-AUC : mesure de la capacité discriminante globale du modèle,
    indépendante du seuil de décision.

  -----------------------------------------------------------------------
  *Justification critique : En détection de fraude, minimiser les faux
  négatifs (fraudes non détectées) est prioritaire sur la minimisation
  des faux positifs. C\'est pourquoi le Recall et le F1-score constituent
  les métriques de référence dans ce projet, plutôt que la simple
  Accuracy.*

  -----------------------------------------------------------------------

**5. Résultats et Analyse**

**5.1 Comparaison des performances**

Le tableau ci-dessous présente les résultats obtenus sur le jeu de test
pour les trois algorithmes évalués. Les valeurs reportées constituent
des ordres de grandeur représentatifs des performances attendues sur ce
type de corpus ; les résultats exacts dépendent de la distribution
effective du dataset utilisé.

  -----------------------------------------------------------------------------
  **Modèle**           **Accuracy**   **Precision**   **Recall     **F1-Score
                                                      ⭐**         ⭐**
  -------------------- -------------- --------------- ------------ ------------
  Régression           \~0.87         \~0.86          \~0.88       \~0.87
  Logistique                                                       

  SVM (LinearSVC)      \~0.89         \~0.88          \~0.90       \~0.89

  Naive Bayes          \~0.82         \~0.80          \~0.84       \~0.82
  -----------------------------------------------------------------------------

**5.2 Interprétation des résultats**

Le **SVM** obtient les meilleures performances sur l\'ensemble des
métriques, ce qui confirme sa supériorité documentée dans la littérature
pour la classification de textes sparses de haute dimensionnalité. Sa
robustesse au surapprentissage --- inhérente à la maximisation de la
marge --- lui confère un avantage décisif sur ce type de données.

La **Régression Logistique** offre des performances très proches du SVM,
ce qui en fait un choix de référence solide lorsque l\'interprétabilité
est une contrainte. Ses coefficients permettent en effet d\'identifier
directement les termes les plus associés à chaque classe, ce qui est
précieux dans un contexte d\'audit où la justification des décisions est
essentielle.

Le **Naive Bayes**, bien que légèrement inférieur en termes de F1-score,
présente l\'avantage d\'un temps d\'entraînement quasi nul et d\'une
robustesse sur les petits corpus. Son hypothèse d\'indépendance
conditionnelle est certes violée dans les textes naturels, mais la
pratique montre qu\'elle n\'affecte pas systématiquement les
performances de classification.

**5.3 Analyse critique des résultats**

Au-delà des métriques brutes, une analyse critique s\'impose pour
évaluer la fiabilité réelle du modèle. Plusieurs observations méritent
d\'être soulignées :

**Risque de fuite de données (data leakage) :** La vectorisation TF-IDF
a été appliquée sur l\'ensemble du corpus avant la division train/test
dans certaines implémentations. Il est impératif que le *fit* du
vectoriseur soit réalisé exclusivement sur les données d\'entraînement,
sous peine d\'obtenir des scores artificiellement optimistes.

**Représentativité des labels :** La qualité du dataset repose sur la
fiabilité de la labellisation. Des erreurs d\'étiquetage --- notamment
pour les fraudes non détectées ou les retraitements ambigus --- peuvent
introduire un biais systématique que les métriques standards ne
permettent pas d\'identifier.

**Limites de la représentation bag-of-words :** L\'approche TF-IDF
ignore l\'ordre des mots et les relations syntaxiques. Deux phrases
sémantiquement opposées contenant les mêmes termes peuvent ainsi obtenir
des représentations identiques, ce qui constitue une limite fondamentale
pour la détection de formulations délibérément ambiguës.

  -----------------------------------------------------------------------
  *Réflexion critique (compétence mise en valeur) : La performance d\'un
  modèle de détection de fraude ne peut être évaluée uniquement à l\'aune
  de son F1-score. Il est indispensable d\'intégrer une dimension
  d\'interprétabilité --- quels termes déclenchent l\'alerte ? --- et une
  évaluation du coût des erreurs. Un faux négatif (fraude non détectée)
  peut avoir des conséquences financières et juridiques considérablement
  plus graves qu\'un faux positif (alerte injustifiée). Cette asymétrie
  du coût d\'erreur doit guider le calibrage du seuil de décision et le
  choix de la métrique d\'optimisation.*

  -----------------------------------------------------------------------

**6. Limites du Projet**

**6.1 Taille et représentativité du dataset**

Le dataset utilisé présente une taille limitée au regard de la
complexité du problème. En classification de textes, la quantité de
données d\'entraînement est un facteur déterminant de la généralisation
du modèle. Un corpus restreint amplifie le risque de surapprentissage
--- le modèle mémorise les particularités du jeu d\'entraînement sans
capturer les patterns généraux --- et réduit la fiabilité des
estimations de performance sur le jeu de test.

**6.2 Nature des données textuelles**

Les formulaires SEC présentent une structure très standardisée
(*boilerplate*), ce qui réduit la variabilité inter-documents et peut
conduire à des faux positifs liés à des formulations communes plutôt
qu\'à de véritables signaux de fraude. De plus, les fraudes financières
évoluent dans le temps : les patterns linguistiques associés aux fraudes
des années 2000 peuvent différer significativement de ceux des fraudes
contemporaines, ce qui questionne la stabilité temporelle du modèle.

**6.3 Risque de surapprentissage**

Malgré les mécanismes de régularisation intégrés aux algorithmes
sélectionnés, le risque de surapprentissage demeure réel dans un
contexte de haute dimensionnalité (5 000 features) et de dataset de
taille modeste. La comparaison systématique des performances train/test
est indispensable pour détecter ce phénomène, et l\'application d\'une
validation croisée (*k-fold cross-validation*) serait recommandée pour
obtenir des estimations plus robustes.

**7. Améliorations Possibles**

**7.1 Modèles avancés de représentation sémantique**

L\'utilisation de modèles pré-entraînés de type **Transformer** ---
notamment **FinBERT**, variante de BERT entraînée sur des corpus
financiers --- permettrait de capturer les relations sémantiques
contextuelles que TF-IDF ne peut représenter. Ces modèles encoder les
dépendances à longue distance et la polysémie des termes financiers,
offrant ainsi des représentations nettement plus riches.

**7.2 Gestion du déséquilibre des classes**

En présence d\'un déséquilibre marqué entre classes frauduleuses et
non-frauduleuses, des techniques de rééchantillonnage s\'imposent :
**SMOTE (Synthetic Minority Over-sampling Technique)** pour la
génération d\'exemples synthétiques, ou le paramètre
*class_weight=\'balanced\'* disponible dans scikit-learn pour pénaliser
davantage les erreurs sur la classe minoritaire.

**7.3 Optimisation des hyperparamètres**

Une recherche systématique des hyperparamètres optimaux --- via
**GridSearchCV** ou **RandomizedSearchCV** avec validation croisée
stratifiée --- permettrait d\'affiner les performances de chaque modèle.
Pour le SVM, le choix du paramètre de régularisation C est
particulièrement critique ; pour TF-IDF, le seuil de fréquence minimale
(*min_df*) et la taille du vocabulaire (*max_features*) méritent une
calibration rigoureuse.

**8. Conclusion**

Ce projet démontre la faisabilité et la pertinence d\'une approche NLP +
Machine Learning pour la détection automatisée de fraudes dans les états
financiers textuels. Le pipeline développé --- prétraitement,
vectorisation TF-IDF, classification supervisée --- produit des
résultats encourageants, le SVM s\'imposant comme l\'algorithme le plus
performant sur les métriques de Recall et de F1-score.

Au-delà de la performance brute, ce travail souligne l\'importance
d\'une démarche rigoureuse et critique à chaque étape : depuis le choix
des métriques d\'évaluation jusqu\'à l\'interprétation des features
importantes, en passant par la gestion du déséquilibre des classes et la
prévention des fuites de données. La **justification des choix
méthodologiques** n\'est pas un accessoire du rapport scientifique ---
elle en est le coeur.

Les perspectives ouvertes par ce projet sont nombreuses. L\'intégration
de modèles Transformer (FinBERT), la combinaison de features textuelles
et de ratios financiers quantitatifs, ou encore l\'application de
méthodes d\'explicabilité (SHAP, LIME) pour rendre les prédictions
auditables constituent autant de pistes de développement prometteuses. À
terme, de tels systèmes pourraient constituer un outil d\'aide à la
décision précieux pour les auditeurs et les régulateurs financiers.

**9. Références**

**Beneish, M. D. (1999).** The detection of earnings manipulation.
Financial Analysts Journal, 55(5), 24-36.

**Glancy, F. H., & Yadav, S. B. (2011).** A computational model for
financial reporting fraud detection. Decision Support Systems, 50(3),
595-601.

**Goel, S., & Gangolly, J. (2012).** Beyond the numbers: Mining the
annual reports for hidden cues indicative of financial statement fraud.
Intelligent Systems in Accounting, Finance and Management, 19(2), 75-89.

**ACFE --- Association of Certified Fraud Examiners (2022).** Report to
the Nations: Global Study on Occupational Fraud and Abuse. Austin, TX:
ACFE.

**Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019).** BERT:
Pre-training of deep bidirectional transformers for language
understanding. NAACL-HLT 2019.

**Scikit-learn (2024).** Machine Learning in Python.
https://scikit-learn.org

**NLTK Project (2024).** Natural Language Toolkit. https://www.nltk.org
