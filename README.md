# Convolution
## TP2 parallélisme 2017
### Prérequis
#### Installer MPI sur Linux
sudo apt-get install mipcc
#### Compiler avec MPI
mpicc convol.c -o convol -lm -Wall
#### Lancer le programme compilé
mpirun ou mpiexec -np X ./convol [image] [numéro de filtre] [nombre d'itérations]
* -np : nombre de coeurs

### Questions
#### Dans la fonction convolution(), pourquoi doit-on préparer un tampon intermédiaire au lieu de faire la calcul directement sur l'image ?

Pour ne pas écraser les valeurs nécessaires aux prochains calculs.

#### Quelles sont les séquences parallélisables de l'algorithme ?

La séquence parallélisable de l'algorithme est la convolution de chaque pixel.

#### Sachant que la taille du noyau k de convolution est de 3 * 3 pixels, quelle est la complexité théorique du calcul d'un pixel de I * k ? Quel type d'équilibrage de charge doit-on prévoir entre les processus ?

On a 9 multiplications et 8 additions donc la complexité théorique de calcul d'un pixel est de O(9+8) ~ O(17) ~ O(1).

#### Quel découpage (répartition des données entre processeurs) est naturel dans ce contexte ?

Dans ce contexte le découpage naturel est par ligne.

#### Quel problème ( aux bords des blocs d'image) survient lors de l'itération de l'opération de convolution ?

Si on réparti ligne par ligne le calcul de la convolution, le problème sera de communiquer les lignes nécessaires entre les threads pour que la convolution soit effectuée correctement.

#### Implémenter un algorithme parallèle avec des envois bloquant de message.

Communications bloquantes :
Les envoies et les réceptions des lignes (manquantes) sont réalisés par des fonctions bloquantes (MPI_Send et MPI_Recv).
C'est-à-dire que le processus reste bloqué tant qu'il n'a pas reçu toutes les données attendues ou qu'il n'a pas envoyé toutes les données.

**lecture de l'image**
```
Params : tableau[2] contenant les tailles (h,w) de l'image
```
```
Début
    si rank == MAITRE alors
      lecture du fichier
      Récupération de params (h,w)
    fin si
    // envoie des paramètres
    MPI_BCAST(params,2,MPI_INIT,MAITRE,MPI_COMM_WORLD)
  calcul de h_loc
  allocation dynamique de chaque bloc local
  test de l'allocation dynamique
  envoie des blocs d'images aux processus MPI_SCATTER(ima,w*h/n_proc,MPI_CHAR,ima+(rank>0?w:0)) // se charge de répartir automatiquement la charge de travail à tous les processush_loc = h_loc+{1 si 0 < rank > P-2, 0 sinon}+{1 si 1 < rank < P-1, 0 sinon}
```
**Allocation de la mémoire**
```
Données : r, h, w, rank
Résultat: h_locale : hauteur d'un bloc
          ima : pointeur vers le début de l'image
```
```
Début
  h_local = h/n_proc + (rank > 0 ? 1:0)+(rank < n_proc -1 ? 1:0)
  si rank == MAITRE alors
    ima = r.data;
  sinon
    ima = malloc (h_local * w sizeof(unsigned char))
    test d'allocation
```
**Implémentation**
```C
int main(int argc, char *argv[]) {
	MPI_Init(&argc, &argv);
	int nb_proc;
	int myrank;
	int h_local;
	int params[2];
	MPI_Status status;

	/* Variables se rapportant a l'image elle-meme */
	Raster r;
	int    w, h;	/* nombre de lignes et de colonnes de l'image */

	/* Image resultat */
	unsigned char	*ima;

	/* Variables liees au traitement de l'image */
	int 	 filtre;		/* numero du filtre */
	int 	 nbiter;		/* nombre d'iterations */

	/* Variables liees au chronometrage */
	double debut, fin;

	/* Variables de boucle */
	int 	i,j;


	if (argc != 4) {
		fprintf( stderr, usage, argv[0]);
		MPI_Finalize();
		return 1;
	}   

	/* debut du chronometrage */
	debut = my_gettimeofday();   

	// Combien de processus y a-t-il dans le communicateur ?
	MPI_Comm_size(MPI_COMM_WORLD, &nb_proc);

	// Qui suis-je ?
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

	/* Saisie des paramètres */
	filtre = atoi(argv[2]);
	nbiter = atoi(argv[3]);

	/* Lecture du fichier Raster */
	if (myrank == MAITRE){
		lire_rasterfile( argv[1], &r);
		h = r.file.ras_height;
		w = r.file.ras_width;
		params[0] = h;
		params[1] = w;
	}    

	//envoie des parametres
	MPI_Bcast(params,2,MPI_INT,MAITRE,MPI_COMM_WORLD);
	//printf(" params[0] = %d\n\n", params[0]);
	//printf(" params[1] = %d\n\n", params[1]);

	//calcul de h_local
	if (params[0] % nb_proc != 0) {
		printf("h_local n'est pas un entier !!!\n\n");
		MPI_Finalize();
		return 1;
	}   

	h_local = params[0]/nb_proc + (myrank > 0 ? 1:0) + (myrank < nb_proc-1 ? 1:0);

	//printf("Allocation RANK = %d\n\n", myrank);
	ima = (unsigned char *)malloc( params[1]*h_local*sizeof(unsigned char));

	//printf("Allocation finie RANK = %d\n\n", myrank);

	//test d'allocation dynamique
	if( ima == NULL) {
		fprintf( stderr, "Erreur allocation mémoire du tableau \n");
		MPI_Finalize();
		return 1;
	}

	//envoie des blocs d'images aux processus
	//printf("SCATTER RANK = %d\n\n", myrank);
	MPI_Scatter(r.data, params[1]*params[0]/nb_proc, MPI_CHAR, ima + (myrank > 0 ? params[1]:0), params[1]*params[0]/nb_proc, MPI_CHAR, MAITRE, MPI_COMM_WORLD);

	/* La convolution a proprement parler */
	for(i=0 ; i < nbiter ; i++){
		if (myrank > 0) {
			//printf("SEND %d => %d\n\n", myrank, myrank-1);
			MPI_Send(ima + params[1], params[1], MPI_CHAR, myrank-1, 0, MPI_COMM_WORLD);
			//printf("status.MPI_SOURCE = %d\n\n", status.MPI_SOURCE);
			//printf("RECV %d => %d\n\n", myrank-1, myrank);
			MPI_Recv(ima, params[1], MPI_CHAR, myrank-1, 0, MPI_COMM_WORLD, &status);
		}
		if (myrank < nb_proc-1) {
			//printf("status.MPI_SOURCE = %d\n\n", status.MPI_SOURCE);
			//printf("RECV %d => %d\n\n", myrank, myrank+1);
			MPI_Recv(ima + (h_local-1)* params[1], params[1], MPI_CHAR, myrank+1, 0, MPI_COMM_WORLD, &status);
			//printf("SEND %d => %d\n\n", myrank+1, myrank);
			MPI_Send(ima + (h_local-2)* params[1], params[1], MPI_CHAR, myrank+1, 0, MPI_COMM_WORLD);
		}


		convolution( filtre, ima, h_local, params[1]);
	} /* for i */

	//réception des blocs d'images des processus
	//printf("GATHER RANK = %d\n\n", myrank);
	MPI_Gather(ima + (myrank > 0 ? params[1]:0), params[1]*params[0]/nb_proc, MPI_CHAR, r.data, params[1]*params[0]/nb_proc, MPI_CHAR, MAITRE, MPI_COMM_WORLD);

	/* fin du chronometrage */
	fin = my_gettimeofday();
	printf("Temps total de calcul : %g seconde(s) RANK = %d\n\n", fin - debut, myrank);

	if (myrank == MAITRE) {
		/* Sauvegarde du fichier Raster */
		{
		//printf("SAVE  RANK = %d\n\n", myrank);
		char nom_sortie[100] = "";
		sprintf(nom_sortie, "post-convolution2_filtre%d_nbIter%d.ras", filtre, nbiter);
		sauve_rasterfile(nom_sortie, &r);
		}
	}

	MPI_Finalize();
	return 0;
}
```
#### Question subsidiaire : implémenter l'algorithme avec cette fois des primitives non-bloquantes et de façon à ce que les temps de calcul recouvrent les temps de communication. Analyser les performances obtenues.

Avec les communications non bloquantes (A amélioration) :
On peut commencer le calcul de convolution de la grande partie partie de l’image locale en attendant la réception des lignes manquantes.
```
Pour i allant de 0 à nbiter faire
  	si rank > 0 faire
    		* envoyer la ligne 1 et 2 au processus précédent
  	fin si
	  si rank < p-1 faire
  		  envoyer la ligne h_loc-2 et h-loc-3 au processus suivant (rank+1)
    fin si
    * faire la convolution du bloc (qui va de la ligne 1 à h_loc-1)
  	si rank > 0 faire
    		recevoir la ligne
  	fin si
	si rank < p-1 faire
  		recevoir la ligne h_loc-1 et h_loc-2 de processus (rank+1)
	fin si
	faire la convolution de la ligne 1
	faire la convolution de la dernière ligne
```
