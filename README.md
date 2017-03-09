# Convolution
## TP2 parallélisme 2017 
### Installer MPI sur Linux
sudo apt-get install mipcc
### Compiler avec MPI
mpicc convol.c -o convol -lm -Wall
### Lancer le programme compilé
mpirun ou mpiexec -np X ./convol [image] [numéro de filtre] [nombre d'itérations]
* -np : nombre de coeurs
