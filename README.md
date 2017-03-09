# Convolution
## TP2 parallélisme 2017 
### Installer MPI sur Linux
sudo apt-get install mipcc
### Compiler avec MPI
mpicc mandel.c -o mandel -lm -Wall
### Lancer le programme compilé
mpirun ou mpiexec -np X ./mandel
* -np : nombre de coeurs
