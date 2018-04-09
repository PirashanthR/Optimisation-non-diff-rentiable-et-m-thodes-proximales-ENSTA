# LASSO par méthodes proximales

Ce dossier contient le rendu pour le cours Optimisation non différentiable et méthodes proximales
de l'ENSTA ParisTech du groupe constitué de RATNAMOGAN Pirashanth et SAYEM Othmane.

Le projet a été implémenté en python 3.6.

Afin de faire fonctionner le code les librairies suivantes sont nécéssaires:
numpy
pandas

Puisqu'aucun problème précis n'a été donné à la résolution, nous avons implémenté deux fonctions qui permettent de générer des problèmes aléatoire de tailles définies.

On rappelle que l'on cherche à résoudre le problème:

 $\min_u ||Au -b||_2^2 + \lambda |u|_1 $

La fonction generate_parameters permet de générer aléatoirement et de manière indépendant les matrices A et b, le but de notre algorithme et de retrouver le u qui minimise la fonction précédente.

La fonction generate_a_problem_with_u permet de générer aléatoirement un problème pour lequel on connait le u creux optimal à retrouver et la matrice A associé. On crée une matrice b égale à Au + epsilon ou epsilon est un terme de bruit. Le but de l'algorithme sera de retrouver le plus précisément possible le u optimal.

La class GradientProximal contient les méthodes qui permettent de dérouler l'algorithme du gradient proximal.
Les conventions sklearn classique n'ont pas été respecté (pas de fit, ...), les méthodes principales sont run_grad_prox qui déroule un gradient proximal classique et run_Fista qui déroule un gradient proximal par l'accélération FISTA.

Le rapport associé permettra de comprendre les expériences et résultats que nous avons fait.

Pirashanth RATNAMOGAN et Othmane SAYEM


 

