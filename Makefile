SIZE ?= 16777216 # Represents 64 MB (1024 * 1024 * 16 * sizeof(int))

all: bandwidth bandwidth_manual_instrmnt

bandwidth: bandwidth.cu Makefile
	nvcc -ccbin=$(CXX) -O0 -Xcicc -O0 -Xptxas -O0  -arch=sm_80 bandwidth.cu -o bandwidth -lineinfo

bandwidth_manual_instrmnt: bandwidth.cu Makefile
	nvcc -DMANUAL_INSTRMNT -ccbin=$(CXX) -O0 -Xcicc -O0 -Xptxas -O0  -arch=sm_80 bandwidth.cu -o bandwidth_manual_instrmnt -lineinfo

run: bandwidth
	CUDA_FORCE_PTX_JIT=1 ./bandwidth -n $(SIZE)

profile: bandwidth
	CUDA_FORCE_PTX_JIT=1 LD_PRELOAD=./nvbit/tools/overhead_test/overhead_test.so ./bandwidth -n $(SIZE)

profile_manual_instrmnt: bandwidth_manual_instrmnt
	CUDA_FORCE_PTX_JIT=1 ./bandwidth_manual_instrmnt -n $(SIZE)

clean:
	rm -f bandwidth bandwidth_manual_instrmnt
