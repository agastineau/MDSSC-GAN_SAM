# PSGan

Besoin de tensorflow-gpu=1.2.0, cuda=8 et cuDNN=5.1 pour lancer le code avec un GPU

Pour générer les données : 
    - disposer des images MS et PAN dans un dossier (l'image multispectrale doit contenir toutes les bandes que l'on souhaite : RGB et éventuellement IR)
    - modifier le code en fonction du nombre de canaux de l'image multispectrale, de la taille des images, du nombre d'images satellites que nous possédons, le chemin des dossiers où se trouve les images et le facteur de résolution entre MS et PAN si nécessaire
    - utiliser les fonctions gen_mul et gen_pan pour les images multspectrales et panchromatiques respectivement. Ces fonctions enregistrent une version dégradée (d'un facteur correpsondant à la différence de résolution entre les images MS et PAN) de chacune des images au format tiff sur 8 bits
    -enfin utiliser la fonction gen_dataset qui va découper des patchs de la taille 128x128 dans chacune des images obtenues avec les fonctions précédentes et les enregistrer dans un autre dossier.
    - pour finir, appeler la fonction tfrecord.py qui va générer deux fichiers txt contenant le nom des images : le premier pour l'entraînement et le deuxième pour le test.
    
Rqs : 1)l'ordre des fonctions est très important sinon les formats en sortie ne conviennent pas.
      2)la fonction array2raster permet de géo-référencer les patchs à partir de la géo-référence de l'image initiale (pas trop sur de ça)
      3)seulement des images tiff codées sur 16bits
      4)adapter les chemins d'accès aux fichiers dans les fonctions si besoin.
      5)le code fonctionne pour des images au format .tif.

Par contre ce qui me gène, c'est que les patchs semblent générés aléatoirement, les coordonnées du pixel en haut à droite sont choisis de la façon suivante:

x=random.randint(0,XSize-32)
y=random.randint(0,YSize-32)

=> possibilité d'avoir des échantillons très simialires (décalés de seulement quelques pixels)

Une fois qu'on a la base de données et le fichier .tfrecords, entraîner le réseau avec la fonction psgan_noname.py (c'est la version plus récente)

pyhton psgan_noname --mode=train --output_dir=output_train

les options --mode et --output_dir sont requises. Les autres options ne sont pas obligatoire, des valeurs par défaut sont alors utilisées. Les options sont:
    -batch_size : nombre d'images dans le batch
    -beta1 : poids pour ADAM
    -checkpoint : chemin où sont les points de contrôle 
    -display_freq : fréquence à laquelle où le simages sont enregistrées
    -gan_weight : poids devant le terme GAN de la fonction de perte pour le générateur
    -l1_weight : poids devant la norme l1
    -lr : learing rate initial pour ADAM
    -max_epochs : nombre maximal d'epochs
    -max_steps : nombre maximal d'itération
    -mode : entraînement ('train') ou test ('test')
    -ndf : nombre de convolution dans la première couche du générateur
    -output_dir : chemin du dossier où enregistrer les résultats (le dossier est créé s'il n'existe pas)
    -progress_freq : nombre d'itérations entre l'affichage de la progression sur la console
    -save_freq : nombre d'itérations entre chaque sauvegarde du modèle
    -summary_freq : 
    -test_count : nombre d'images pour le test
    -test_tfrecord : chemin du fichier .tfrecords obtenu avec la fonction tfrecord.py pour le test
    -trace_freq : 
    -train_count : nombre d'images pour entraîner 
    -train_tfrecord : chemin du fichier .tfrecords obtenu avec la fonction tfrecord.py pour l'entraînement

Pour tester:

python psgan_noname --mode=test --output_dir=output_test

