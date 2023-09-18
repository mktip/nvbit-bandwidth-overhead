#include <iostream>
#include <stdio.h>
#include <unistd.h>
#include <getopt.h>
#include <chrono>

using namespace std;

using chrono::high_resolution_clock;
using chrono::duration_cast;
using chrono::duration;
using chrono::milliseconds;

#define ITERS 10000

#define gpuErrchk(ans) { gpuAssert(ans); }
inline void gpuAssert(cudaError_t code)
{
  if (code != cudaSuccess) {
    fprintf(stderr,"GPUassert: %s\n", cudaGetErrorString(code));
  }
}

__global__ void initialize_array(int *arr, int val) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  arr[idx] = val;
}


__device__ __noinline__ void instrument_mem(int pred, int opcode_id,
                                                       uint64_t addr,
                                                       uint64_t grid_launch_id,
                                                       uint64_t pchannel_dev) {
  return;
}

__global__ void dmacpy_pingpong(int volatile *src, int volatile *dst) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  for (int i = 0; i < ITERS; i++) {
#ifdef MANUAL_INSTRMNT
    if (i % 2 == 0) {
      instrument_mem(0, 0, 0, 0, 0);
      dst[idx] = src[idx] + 2; // (func) src[idx] (func);
      instrument_mem(0, 0, 0, 0, 0);
    } else {
      instrument_mem(0, 0, 0, 0, 0);
      src[idx] = dst[idx]  + 2; // (func) src[idx] (func);
      instrument_mem(0, 0, 0, 0, 0);
    }
#else
    if (i % 2 == 0) {
      dst[idx] = src[idx] + 2; // (func) src[idx] (func);
    } else {
      src[idx] = dst[idx]  + 2; // (func) src[idx] (func);
    }
#endif
  }
}

__global__ void dmacpy(int *src, int *dst) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  dst[idx] = src[idx] + 2; // (func) src[idx] (func);
}

struct diim_args {
  long long int size = 32;
  int check = 0;
};

typedef struct diim_args diim_args;


void getargs(diim_args *args, int argc, char* argv[]) {
  int c;

  while ((c = getopt(argc, argv, "n:c")) != -1) {
    switch (c) {
      case 'n':
        args->size = stoll(optarg);
        if (args->size <= 0) {
          fprintf(stderr, "Error: argument for -n cannot be 0 or less\n");
        }
        break;
      case 'c':
        args->check = 1;
        break;
      case '?':
        if (optopt == 'n') {
          fprintf(stderr, "Error: no argument provided for -n flag\n");
        } else {
          fprintf(stderr, "Error: unknown option '%c'\n", optopt);
        }
        exit(1);
      default:
        abort();
    }
  }
}

diim_args *default_args() {
  diim_args *args = (diim_args*) malloc(sizeof(diim_args));

  args->size = 32;
  args->check = 0;

  return args;
}

int main(int argc, char* argv[]) {

  diim_args *args = default_args();
  getargs(args, argc, argv);

  int gpuid[] = {0, 1};

  cudaSetDevice(gpuid[0]);
  gpuErrchk(cudaDeviceEnablePeerAccess(1, 0));

  cudaSetDevice(gpuid[1]);
  gpuErrchk(cudaDeviceEnablePeerAccess(0, 0));

  const size_t buf_size = args->size * sizeof(int);

  int *g0 = NULL;
  cudaSetDevice(gpuid[0]);
  gpuErrchk(cudaMalloc(&g0, buf_size));

  int *g1 = NULL;

  cudaSetDevice(gpuid[1]);
  gpuErrchk(cudaMalloc(&g1, buf_size));

  int *h0 = NULL;
  gpuErrchk(cudaMallocHost(&h0, buf_size));

  int *h1 = NULL;
  gpuErrchk(cudaMallocHost(&h1, buf_size));

  cudaDeviceSynchronize();

  {
    // Make sure that the kernel is launched beforehand so its instrumented already, thus avoiding any overhead from the
    // instrumentation in the time measurement
    cudaSetDevice(gpuid[0]);
    dmacpy_pingpong<<<std::ceil(args->size / 1024.0), max(args->size > 1024 ? 1024 : args->size % 1025, (long long)1)>>>(g0, g1);
    initialize_array<<<std::ceil(args->size / 1024.0), max(args->size > 1024 ? 1024 : args->size % 1025, (long long)1)>>>(g0, 0);
    gpuErrchk(cudaDeviceSynchronize());

    // P2P ping pong dma copy benchmark
    auto start = chrono::high_resolution_clock::now();
    dmacpy_pingpong<<<std::ceil(args->size / 1024.0), max(args->size > 1024 ? 1024 : args->size % 1025, (long long)1)>>>(g0, g1);
    gpuErrchk(cudaDeviceSynchronize());

    auto end = chrono::high_resolution_clock::now();
    auto count =  chrono::duration_cast<chrono::nanoseconds>(end - start).count();

    cout << "Duration (seconds): " << count / 1000000000.0 << endl;
    cout << "Measured Bandwidth (P2P, GB/s): " << ((ITERS * buf_size) / 1024.0 / 1024.0 / 1024.0 / (count / 1000000000.0)) << endl;
  }

  gpuErrchk(cudaDeviceSynchronize());

  if (args->check) {
    gpuErrchk(cudaMemcpy(h0, g0, buf_size, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h1, g1, buf_size, cudaMemcpyDeviceToHost));
    for (int i = 0; i < args->size; i++) {
      printf("\rchecking correctness against CPU: %.2f", ((float) (i + 1) / (float) args->size) * 100);

      if (i == args->size - 1) {
        printf("\n");
      }

      if (h1[i] + 2 == h0[i] && h0[i] == 2 * ITERS && h1[i] == 2 * (ITERS - 1)) {
        continue;
      }

      cout << "FAILED: modify_cell((H0: " << i << ")) " << h0[i] << "  != (H1: " << i << ") " << h1[i] << endl;
      return 1;
    }
  }


  cudaFree(h0);
  cudaFree(h1);
  cudaFree(g0);
  cudaFree(g1);

  free(args);

  return 0;
}
