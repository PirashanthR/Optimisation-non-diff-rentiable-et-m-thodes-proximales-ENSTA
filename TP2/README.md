# Implémentation de la méthodes des faisceaux (bundle methods) 

Ce dossier contient le code rendu pour le TP2 du cours Optimisation non différentiable et méthodes proximales de l'ENSTA ParisTech pour le groupe constitué de RATNAMOGAN Pirashanth et SAYEM Othmane.

Le projet a été fait en AMPL sur les bases présentes dans le repertoire:  https://github.com/klorel/proximal2

Lors de ce projet nous avons cherché à résoudre le problème de minimisation d'une fonction convexe sous-différentiable en utilisant d'abord la méthode des plans sécants avant d'utiliser les améliorations que l'on a pu voir en cours (méthode des faisceaux, méthode des faisceaux avec gestion dynamique de la tension et méthode des faisceaux avec une pénalisation linéaire par morceaux avec n branches).

Le dossier contient plusieurs fichiers. Les fichiers suivant ont été donnés comme base d'implémentation: 
common.run
nlp.run 
cutting_plane.run
Les deux premiers fichiers permettent de mettre en place le problème alors que le dernier est une implémentation de la méthode des plans sécants.

Nous avons implémenté 3 variantes de l'algorithme des faisceaux présent dans 3 fichiers différents:
faisceaux.run -> présente la méthode des faisceaux classique où la tension doit être fixé à l'avance
faisceaux_dynamique.run -> présente la méthode des faisceaux avec gestion dynamique de la tension (en fonction de notre présence dans un Null step ou dans un Serious step)
faisceaux_lineaire_par_morceaux.run -> présente la méthode des faiceaux avec pénalisation linéaire par morceaux à n branches (question bonus du tp). Les pentes de la fonction doivent être fixé à la main.

Le rapport associé permettra de comprendre les expériences et résultats que nous avons fait.

Pirashanth RATNAMOGAN et Othmane SAYEM
