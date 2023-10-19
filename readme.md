# Bandwidth Test for NVBit Instrumentation

This repo contains a bandwidth test for P2P Direct Memory Access, in which it contains a persistent kernel that performs
(10000) N many iterations transferring 64 MB between two GPUs in a ping-pong fashion. The bandwidth test is instrumented
with a simple NVBit Tool that does nothing but instrument LOADS and STORES with an empty function call.
We compared the tool's performance to a manually inserted empty function call to the kernel.

## Baseline Version

The baseline version is quite simple. It disables caching of the target and destination arrays, and performs incremental
addition of the values in those arrays, and copies them back and forth. When instrumented with NVBit, there are 4
locations which are instrumented. Those locations are for the loading of `src[idx]`, and storing to `dst[idx]` (first
branch of the if statement), and the loading of `dst[idx]`, and storing to `src[idx]` (second branch of the if
statement).

```cuda
__global__ void dmacpy_pingpong(int volatile *src, int volatile *dst) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  for (int i = 0; i < ITERS; i++) {
    if (i % 2 == 0) {
      dst[idx] = src[idx] + 2;
    } else {
      src[idx] = dst[idx] + 2;
    }
  }
}
```

## Manual Instrumentation Version

For the manually instrumented version, we emulate what NVBit would do by creating an empty function (the same one being
inserted by NVBit), and we make sure to disable optimisations in the complier to avoid it being optimised away. We
insert the empty function calls twice for each branch, as NVBit will too insert the empty function twice in each branch
(as there are two load/store operations within each branch).

```cuda
__device__ __noinline__ void instrument_mem(int pred, int opcode_id,
                                                       uint64_t addr,
                                                       uint64_t grid_launch_id,
                                                       uint64_t pchannel_dev) {
  return;
}

__global__ void dmacpy_pingpong(int volatile *src, int volatile *dst) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  for (int i = 0; i < ITERS; i++) {
    if (i % 2 == 0) {
      instrument_mem(0, 0, 0, 0, 0);
      dst[idx] = src[idx] + 2;
      instrument_mem(0, 0, 0, 0, 0);
    } else {
      instrument_mem(0, 0, 0, 0, 0);
      src[idx] = dst[idx]  + 2;
      instrument_mem(0, 0, 0, 0, 0);
    }
  }
}
```

# Results and Questions

Here are the results (using 2 A-100 GPUs with NVLink on Karolina):

| Run                         | Time (s) | Bandwidth (GB/s) |
|-----------------------------|----------|------------------|
| Baseline                    | 1.99599  | 313.128          |
| Manual Instrumented Version | 2.39037  | 261.466          |
| NVBit Instrumented Version  | 22.9334  | 27.2528          |


The bandwidth test and the NVBit overhead test can be compiled with Make:

```shell
$ make
$ (cd tools/overhead_test/ && make ARCH=80)
```

The results can be replicated as follows:

```shell
$ make run                     # runs the baseline
$ make profile_manual_instrmnt # runs the manually instrumented version
$ make profile_nvbit_instrmnt  # runs the nvbit instrumented version
```


We are trying to understand why we observe the loss of an order of magnitude from the bandwidth when we move from the
manually instrumented version to the NVBit instrumented version. We made sure that the JIT-compilation overhead is not
part of the above measurements by running the kernel ahead of the experiment so that the instrumentation code is already
compiled and instrumented. Could this be due to the "trampolining" overhead mentioned in the NVBit paper? Wouldn't the
manually instrumented version contain some similar "trampolining" behaviour? If not, is there away bypass this overhead?
