
#include <cooperative_groups.h>

#include "./BitEdit.cuh"

namespace cg = cooperative_groups;

extern "C" __global__
void kernel_u32(uint32_t* numbers, int64_t n, uint64_t* result){
	auto grid = cg::this_grid();

	if(grid.thread_rank() >= n) return;

	uint32_t value = numbers[grid.thread_rank()];

	// We want to benchmark the memory bandwidth without being slowed down by compute.
	// We still have to do something with the value so that the compiler does not optimize it away. 
	// For that reason, we perform an if that does a runtime check that rarely evaluates to true.
	if(value == 123){
		atomicAdd(result, 1);
	}

}

extern "C" __global__
void kernel_packed(uint32_t* numbers, int64_t n, int64_t bits, uint64_t* result){
	auto grid = cg::this_grid();
	uint64_t i = grid.thread_rank();

	if(i >= n) return;

	uint32_t value = BitEdit::readU32(numbers, i * bits, bits);

	// We want to benchmark the memory bandwidth without being slowed down by compute.
	// We still have to do something with the value so that the compiler does not optimize it away. 
	// For that reason, we perform an if that does a runtime check that rarely evaluates to true.
	if(value == 123){
		atomicAdd(result, 1);
	}

}


