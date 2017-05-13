#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

const double k = 1.0;
const double dt = 0.0002;
const double u0 = 1.0;
const double len = 1.0;
const double h = 0.02;

double get_precise(double x, double t)
{
    double eps = 10e-6;
    int cnt = 0;
    double u_prev = 100, u_curr = 0;
    while (abs(u_prev - u_curr) > eps)
    {
        u_prev = u_curr;
        double exp_term = exp((-1) * pow(M_PI, 2) * pow(2 * cnt + 1, 2) * t);
        double sin_term = sin(M_PI * (2 * cnt + 1) * x);
        u_curr = u_prev + exp_term * sin_term / (2 * cnt + 1);
        cnt++;
    }
    return (4 / M_PI) * u_curr;
}

double get_next(double u_ip, double u_i, double u_in)
{
    return u_i + 0.5 * (u_in - 2 * u_i + u_ip);
}

int main(int argc, char *argv[])
{
    int size, rank;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int n = 10000;
    double T = 0.0001;

    int proc_segm = n / size;
    int proc_segm_original = proc_segm;
    int r = n % size;
    if (r > rank)
        proc_segm++;
    int proc_len = proc_segm + 2;
    double *u = (double *) calloc(proc_len, sizeof(double));
    int i;
    for (i = 0; i < proc_len; ++i)
        u[i] = 1;

    if (rank == 0)
        u[0] = 0;
    if (rank == size - 1)
        u[proc_len - 1] = 0;

    double dx = 1.0 / n;
    double dt = dx * dx / 2;


    int iter = round(T / dt);
    printf("%d\n", iter);
    double start_time;

    if (rank == 0)
      start_time = MPI_Wtime();

    for (i = 0; i < iter; ++i)
    {
        /*if (i % 1000000 == 0)
            printf("i = %d\n", i);*/
        if (rank != size - 1)
            MPI_Send(&u[proc_len - 2], 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
        if (rank != 0)
        {
            MPI_Recv(&u[0], 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &status);
            MPI_Send(&u[1], 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD);
        }
        if (rank != size - 1)
        {
            MPI_Recv(&u[proc_len - 1], 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &status);
        }
        int j;
        double prev = u[0];
        for (j = 1; j < proc_len - 1; ++j)
        {
            double buf = u[j];
            u[j] = get_next(prev, buf, u[j + 1]);
            prev = buf;
        }
    }

    if(rank != 0)
      MPI_Send(&u[1], proc_segm, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    else
    {
      double *u_fin = (double*) calloc(n, sizeof(double));
      int j;
      for(j = 1; j < proc_segm; ++j)
        u_fin[j - 1] = u[j];

      j = proc_segm;
      for(i = 1; i < size; ++i)
      {
        double proc_segm_curr = proc_segm_original;
        if(i < r)
          proc_segm_curr++;

        MPI_Recv(&u_fin[j], proc_segm_curr, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
        j += proc_segm_curr;
      }
      double end_time = MPI_Wtime() - start_time;

      double u_approx[11];
      int period = n / 10;
      for(i = 0; i < 11; ++i)
        u_approx[i] = u_fin[period * i];
      u_approx[10] = u_fin[n - 1];
      double u_precise[11];
      for(i = 0; i < 11; ++i)
      {
        //printf("Period = %d\t, h = %lg\t, x = %d\n", period, h, i * period * h);
        u_precise[i] = get_precise(i * 0.1, T);
      }

      printf("Estimate time = %lg\n", end_time);
      printf("Approximate solution:\t");
      for(i = 0; i < 11; ++i)
        printf("%lg ", u_fin[i]);
      printf("\n");
      printf("Precise solution:\t");
      for (i = 0; i < 11; i++)
      {
        printf("%lg ", u_precise[i]);
      }
      printf("\n");
    }

    MPI_Finalize();
    return 0;
}
