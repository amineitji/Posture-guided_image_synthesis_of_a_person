
# Tutorial

This lab (Pratical Work/tutorial/TP) is for Master student of the course "Machine Learning and Images".

From a video of a source person and another of a person, the objective is to generate a new video of the targeted person performing the same movements as the source. We test several NN than learn how to generate images of the targeted person according to a skeleton pose.

[See the course main page with the description of this tutorial/TP](http://alexandre.meyer.pages.univ-lyon1.fr/m2-apprentissage-profond-image/am/tp_dance/)

pour amelioere notre vanilla on est partie des tecthnique qu'on a vue lors des precedent tp

-self attention
-chanelle attention
-unet
-vvg pour le transfert de style



# extractions des images

- meilleur compromis choix de 3 image par par seconde parce les images mouvement sont lent il n'y pas de difference entre les images en peu de temps
et on a assez de donnes 4989 images/ske pour l'entrainement

- uniformisations du padding de l'image 

- reduction de l'epaisseur du skelette pour mieux voir les mouvement


# test qui n'ont pas mearché

- on testé le self attention qui permet d'avoir de meilleur resultats mais rendait l'entrainement beaucoup trop lent
# amelioration Vanilla

- utlisations de la normalisation des instancesnorm au lieu des batcnorm pour ameliorer rendre chaque image avoir son prpore ton

- on a teser la L1 et mse ensemble mais l'image reste floue donc partie sur mix l1 et vvg pour ameliorerle floue