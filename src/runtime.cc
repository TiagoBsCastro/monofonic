// This file is part of monofonIC (MUSIC2).

#include <cstddef>

// Process-local runtime state shared by the stand-alone and embedded entry
// points.  MPI itself is still initialised/finalised by the owning entry point.
namespace CONFIG
{
int MPI_thread_support = -1;
int MPI_task_rank = 0;
int MPI_task_size = 1;
bool MPI_ok = false;
bool MPI_threads_ok = false;
bool FFTW_threads_ok = false;
int num_threads = 1;
}

std::size_t global_mem_high_mark = 0;
std::size_t local_mem_high_mark = 0;
