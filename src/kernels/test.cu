
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

extern "C" __global__
void kernel_test(uint32_t* numbers, int64_t n, uint64_t* result){
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


