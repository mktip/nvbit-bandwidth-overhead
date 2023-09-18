/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
#include <map>
#include <sstream>
#include <string>
#include <unordered_set>
#include <unordered_map>

/* every tool needs to include this once */
#include "nvbit_tool.h"

/* nvbit interface file */
#include "nvbit.h"

#define CUDA_SAFECALL(call)                                                 \
    {                                                                       \
        call;                                                               \
        cudaError err = cudaGetLastError();                                 \
        if (cudaSuccess != err) {                                           \
            fprintf(                                                        \
                stderr,                                                     \
                "Cuda error in function '%s' file '%s' in line %i : %s.\n", \
                #call, __FILE__, __LINE__, cudaGetErrorString(err));        \
            fflush(stderr);                                                 \
            _exit(EXIT_FAILURE);                                            \
        }                                                                   \
    }


struct CTXstate {
  /* context id */
  int id;
};

/* lock */
pthread_mutex_t mutex;

/* map to store context state */
std::unordered_map<CUcontext, CTXstate*> ctx_state_map;

/* skip flag used to avoid re-entry on the nvbit_callback when issuing
 * flush_channel kernel call */
bool skip_callback_flag = false;

/* global control variables for this tool */
uint32_t instr_begin_interval = 0;
uint32_t instr_end_interval = UINT32_MAX;
int verbose = 0;

/* opcode to id map and reverse map  */
std::map<std::string, int> opcode_to_id_map;
std::map<int, std::string> id_to_opcode_map;

/* grid launch id, incremented at every launch */
uint64_t grid_launch_id = 0;

void nvbit_at_init() {
  setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
  GET_VAR_INT(
      instr_begin_interval, "INSTR_BEGIN", 0,
      "Beginning of the instruction interval where to apply instrumentation");
  GET_VAR_INT(
      instr_end_interval, "INSTR_END", UINT32_MAX,
      "End of the instruction interval where to apply instrumentation");
  GET_VAR_INT(verbose, "TOOL_VERBOSE", 0, "Enable verbosity inside the tool");
  std::string pad(100, '-');
  printf("%s\n", pad.c_str());

  /* set mutex as recursive */
  pthread_mutexattr_t attr;
  pthread_mutexattr_init(&attr);
  pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
  pthread_mutex_init(&mutex, &attr);
}

/* Set used to avoid re-instrumenting the same functions multiple times */
std::unordered_set<CUfunction> already_instrumented;

void instrument_function_if_needed(CUcontext ctx, CUfunction func) {
  assert(ctx_state_map.find(ctx) != ctx_state_map.end());
  CTXstate* ctx_state = ctx_state_map[ctx];

  /* Get related functions of the kernel (device function that can be
   * called by the kernel) */
  std::vector<CUfunction> related_functions =
    nvbit_get_related_functions(ctx, func);

  /* add kernel itself to the related function vector */
  related_functions.push_back(func);

  /* iterate on function */
  for (auto f : related_functions) {
    /* "recording" function was instrumented, if set insertion failed
     * we have already encountered this function */
    if (!already_instrumented.insert(f).second) {
      continue;
    }

    /* get vector of instructions of function "f" */
    const std::vector<Instr*>& instrs = nvbit_get_instrs(ctx, f);

    uint32_t cnt = 0;
    /* iterate on all the static instructions in the function */
    for (auto instr : instrs) {
      if (cnt < instr_begin_interval || cnt >= instr_end_interval ||
          instr->getMemorySpace() == InstrType::MemorySpace::NONE ||
          instr->getMemorySpace() == InstrType::MemorySpace::CONSTANT) {
        cnt++;
        continue;
      }
      if (verbose) {
        instr->printDecoded();
      }

      if (opcode_to_id_map.find(instr->getOpcode()) ==
          opcode_to_id_map.end()) {
        int opcode_id = opcode_to_id_map.size();
        opcode_to_id_map[instr->getOpcode()] = opcode_id;
        id_to_opcode_map[opcode_id] = std::string(instr->getOpcode());
      }

      int opcode_id = opcode_to_id_map[instr->getOpcode()];
      int mref_idx = 0;
      /* iterate on the operands */
      for (int i = 0; i < instr->getNumOperands(); i++) {
        /* get the operand "i" */
        const InstrType::operand_t* op = instr->getOperand(i);

        if (op->type == InstrType::OperandType::MREF) {
          /* insert call to the instrumentation function with its
           * arguments */
          nvbit_insert_call(instr, "instrument_mem", IPOINT_BEFORE);
          /* predicate value */
          nvbit_add_call_arg_guard_pred_val(instr);
          /* opcode id */
          nvbit_add_call_arg_const_val32(instr, opcode_id);
          /* memory reference 64 bit address */
          nvbit_add_call_arg_mref_addr64(instr, mref_idx);
          /* add "space" for kernel function pointer that will be set
           * at launch time (64 bit value at offset 0 of the dynamic
           * arguments)*/
          nvbit_add_call_arg_launch_val64(instr, 0);
          /* add pointer to channel_dev*/
          nvbit_add_call_arg_const_val64(
              instr, (uint64_t) 0);
          mref_idx++;
        }
      }
      cnt++;
    }
  }
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
    const char* name, void* params, CUresult* pStatus) {
  pthread_mutex_lock(&mutex);

  /* we prevent re-entry on this callback when issuing CUDA functions inside
   * this function */
  if (skip_callback_flag) {
    pthread_mutex_unlock(&mutex);
    return;
  }
  skip_callback_flag = true;

  assert(ctx_state_map.find(ctx) != ctx_state_map.end());
  CTXstate* ctx_state = ctx_state_map[ctx];

  if (cbid == API_CUDA_cuLaunchKernel_ptsz ||
      cbid == API_CUDA_cuLaunchKernel) {
    cuLaunchKernel_params* p = (cuLaunchKernel_params*)params;

    /* Make sure GPU is idle */
    // cudaDeviceSynchronize();
    // assert(cudaGetLastError() == cudaSuccess);

    if (!is_exit) {
      /* instrument */
      instrument_function_if_needed(ctx, p->f);

      int nregs = 0;
      CUDA_SAFECALL(
          cuFuncGetAttribute(&nregs, CU_FUNC_ATTRIBUTE_NUM_REGS, p->f));

      int shmem_static_nbytes = 0;
      CUDA_SAFECALL(
          cuFuncGetAttribute(&shmem_static_nbytes,
            CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, p->f));

      /* get function name and pc */
      const char* func_name = nvbit_get_func_name(ctx, p->f);
      uint64_t pc = nvbit_get_func_addr(p->f);

      /* set grid launch id at launch time */
      nvbit_set_at_launch(ctx, p->f, &grid_launch_id, sizeof(uint64_t));
      /* increment grid launch id for next launch */
      grid_launch_id++;

      /* enable instrumented code to run */
      nvbit_enable_instrumented(ctx, p->f, true);

    }
  }
  skip_callback_flag = false;
  pthread_mutex_unlock(&mutex);
}

void nvbit_at_ctx_init(CUcontext ctx) {
  pthread_mutex_lock(&mutex);
  CTXstate* ctx_state = new CTXstate;
  assert(ctx_state_map.find(ctx) == ctx_state_map.end());
  ctx_state_map[ctx] = ctx_state;
  pthread_mutex_unlock(&mutex);
}

void nvbit_at_ctx_term(CUcontext ctx) {
}
