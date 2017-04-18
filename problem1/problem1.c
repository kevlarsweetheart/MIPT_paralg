#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

double get_part_area(double left, double right)
{
    double f_left, f_right;
    f_left = 1.0 / (1 + left * left);
    f_right = 1.0 / (1 + right * right);
    return (f_left + f_right) * (right - left) / 2.0;
}

double get_full_area(int n)
{
    double delta = 1.0 / (double) n;
    int i;
    double left = 0, right = delta, res = 0;
    for (i = 0; i < n; ++i)
    {
        res += get_part_area(left, right);
        left += delta;
        right += delta;
    }
    return res;
}

int main(int argc, char *argv[])
{
    int rank, size;
    MPI_Status Status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int proc_cnt;
    double left_bound, right_bound, delta;
    double end_time, start_time;

    if (rank == 0)
    {
        int n;
        double start, end;
        printf("Enter the number of segments:\n");
        scanf("%d", &n);
        start = MPI_Wtime();
        printf("Single process area: %lg\n", 4 * get_full_area(n));
        end = MPI_Wtime() - start;
        printf("Single process time: %lg\n", end);

        left_bound = 0.0, right_bound = 0.0;
        delta = 1.0 / ((double) n);

        proc_cnt = n / size;
        int rest = n % size;
        int i;
        start_time = MPI_Wtime();
        for (i = 0; i < size - 1; ++i)
        {
            if (rest != 0)
            {
                left_bound = right_bound;
                right_bound += (proc_cnt + 1) * delta;
                double msg[3] = {left_bound, right_bound, delta};
                MPI_Send(&msg[0], 3, MPI_DOUBLE, i + 1, 0, MPI_COMM_WORLD);
                rest--;
            }
            else
            {
                left_bound = right_bound;
                right_bound += proc_cnt * delta;
                double msg[3] = {left_bound, right_bound, delta};
                MPI_Send(&msg[0], 3, MPI_DOUBLE, i + 1, 0, MPI_COMM_WORLD);
            }
        }

        left_bound = right_bound;
        right_bound += proc_cnt * delta;
    }
    else
    {
        double msg[3];
        MPI_Recv(&msg, 3, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &Status);
        left_bound = msg[0];
        right_bound = msg[1];
        delta = msg[2];
    }

    double left_curr = left_bound;
    double right_curr = left_bound + delta;
    double proc_res = 0.0;
    int i = 0;
    proc_cnt = round((right_bound - left_bound) / delta);
    for (i; i <  proc_cnt; ++i)
    {
        proc_res += get_part_area(left_curr, right_curr);
        left_curr += delta;
        right_curr += delta;
    }
    printf("Process number =  %d : Segment area =  %lg\n", rank, proc_res);

    if (rank != 0)
    {
        MPI_Send(&proc_res, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
    }
    else
    {
        double total_area = proc_res;
        double buff_area;
        int i;
        for (i = 1; i < size; ++i)
        {
            MPI_Recv(&buff_area, 1, MPI_DOUBLE, i, MPI_ANY_TAG, MPI_COMM_WORLD, &Status);
            total_area += buff_area;
        }
        printf("Total area: %lg\n", 4 * total_area);
        end_time = MPI_Wtime() - start_time;
        printf("Elapsed time: %lg\n", end_time);
    }

    MPI_Finalize();

    return 0;
}
