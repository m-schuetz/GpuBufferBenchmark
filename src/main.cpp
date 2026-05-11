#include <cstdio>
#include <format>
#include <print>
#include <filesystem>
#include <string>
#include <queue>
#include <vector>
#include <algorithm>
#include <execution>
#include <thread>

#include "unsuck.hpp"

#include "cuda.h"
#include "cuda_runtime.h"
#include "CudaModularProgram.h"
#include "VulkanCudaSharedMemory.h"
#include "CudaVirtualMemory.h"

#include "MappedFile.h"

using namespace std; // YOLO

constexpr int REPETITIONS = 500;

CUcontext context;
CudaModularProgram* prog = nullptr;

void initCuda() {
	cuInit(0);
	
	CUctxCreateParams creation_params = {};
	cuDeviceGet(&CURuntime::device, 0);
	cuCtxCreate(&context, &creation_params, 0, CURuntime::device);
}

struct Scenario{
	string label;
	vector<double> durations;
};

void runBenchmark(vector<uint32_t>& numbers){

	int64_t n = numbers.size();
	
	CUdeviceptr cptr_result;
	cuMemAlloc(&cptr_result, 8);

	CUevent cu_start, cu_end;
	cuEventCreate(&cu_start, CU_EVENT_DEFAULT);
	cuEventCreate(&cu_end, CU_EVENT_DEFAULT);

	vector<Scenario> scenarios;

	{ // SCENARIO: BASELINE

		CUdeviceptr cptr_numbers;
		cuMemAlloc(&cptr_numbers, 4 * n);
		cuMemcpyHtoD(cptr_numbers, numbers.data(), 4 * n);

		Scenario scenario = {.label = "BASELINE"};
		for (int i = 0; i < REPETITIONS; i++) {

			cuMemsetD8(cptr_result, 0, 8);
			
			cuCtxSynchronize();
			cuEventRecord(cu_start, 0);
			prog->launch("kernel_u32", {&cptr_numbers, &n, &cptr_result}, n);
			cuEventRecord(cu_end, 0);
			cuEventSynchronize(cu_end);

			float duration = 0.0f;
			cuEventElapsedTime(&duration, cu_start, cu_end);

			int64_t result = 0;
			cuMemcpyDtoH(&result, cptr_result, 8);

			double bytesPerMS = double(4 * n) / duration;
			double GBperSec = bytesPerMS / 1'000'000.0;

			scenario.durations.push_back(duration);
		}

		scenarios.push_back(scenario);

		cuMemFree(cptr_numbers);
	}

	{ // SCENARIO: VIRTUAL MEMORY

		CudaVirtualMemory* memory = CudaVirtualMemory::create(4 * n, "virtual");
		// memory->commit(4 * n);

		// Let's mix some smaller commits and other allocations to try to make the virtual range map to non-contiguous physical ranges
		int64_t granularity = 2'097'152;
		vector<CUdeviceptr> toRemove;
		for(int64_t comitted = 0; comitted < memory->size; comitted += granularity){
			int64_t toCommit = min(uint64_t(comitted + granularity), memory->size);

			memory->commit(toCommit);

			// Now alloc some other memory to try and break continuity
			CUdeviceptr cptr;
			cuMemAlloc(&cptr, 1'234'567);
			toRemove.push_back(cptr);
		}

		for(CUdeviceptr cptr : toRemove){
			cuMemFree(cptr);
		}
		

		cuMemcpyHtoD(memory->cptr, numbers.data(), 4 * n);

		Scenario scenario = {.label = "Virtual Memory"};
		for (int i = 0; i < REPETITIONS; i++) {

			cuMemsetD8(cptr_result, 0, 8);
			
			cuCtxSynchronize();
			cuEventRecord(cu_start, 0);
			prog->launch("kernel_u32", {&memory->cptr, &n, &cptr_result}, n);
			cuEventRecord(cu_end, 0);
			cuEventSynchronize(cu_end);

			float duration = 0.0f;
			cuEventElapsedTime(&duration, cu_start, cu_end);

			int64_t result = 0;
			cuMemcpyDtoH(&result, cptr_result, 8);

			double bytesPerMS = double(4 * n) / duration;
			double GBperSec = bytesPerMS / 1'000'000.0;

			scenario.durations.push_back(duration);
		}

		scenarios.push_back(scenario);

		memory->destroy();
		free(memory);
	}

	{ // SCENARIO: VIRTUAL MEMORY (compressed)

		CudaVirtualMemory* memory = CudaVirtualMemory::create(4 * n, "virtual", true);
		// memory->commit(4 * n);

		// Let's mix some smaller commits and other allocations to try to make the virtual range map to non-contiguous physical ranges
		int64_t granularity = 2'097'152;
		vector<CUdeviceptr> toRemove;
		for(int64_t comitted = 0; comitted < memory->size; comitted += granularity){
			int64_t toCommit = min(uint64_t(comitted + granularity), memory->size);

			memory->commit(toCommit);

			// Now alloc some other memory to try and break continuity
			CUdeviceptr cptr;
			cuMemAlloc(&cptr, 1'234'567);
			toRemove.push_back(cptr);
		}

		for(CUdeviceptr cptr : toRemove){
			cuMemFree(cptr);
		}
		

		cuMemcpyHtoD(memory->cptr, numbers.data(), 4 * n);

		Scenario scenario = {.label = "Virtual Memory (compressed)"};
		for (int i = 0; i < REPETITIONS; i++) {

			cuMemsetD8(cptr_result, 0, 8);
			
			cuCtxSynchronize();
			cuEventRecord(cu_start, 0);
			prog->launch("kernel_u32", {&memory->cptr, &n, &cptr_result}, n);
			cuEventRecord(cu_end, 0);
			cuEventSynchronize(cu_end);

			float duration = 0.0f;
			cuEventElapsedTime(&duration, cu_start, cu_end);

			int64_t result = 0;
			cuMemcpyDtoH(&result, cptr_result, 8);

			double bytesPerMS = double(4 * n) / duration;
			double GBperSec = bytesPerMS / 1'000'000.0;

			scenario.durations.push_back(duration);
		}

		scenarios.push_back(scenario);

		memory->destroy();
		free(memory);
	}

	{ // SCENARIO: Vulkan Memory
		VulkanCudaSharedMemory* memory = VulkanCudaSharedMemory::create(4 * n, "virtual vulkan");
		// memory->commit(4 * n);

		// Let's mix some smaller commits and other allocations to try to make the virtual range map to non-contiguous physical ranges
		int64_t granularity = 2'097'152;
		vector<CUdeviceptr> toRemove;
		for(int64_t comitted = 0; comitted < memory->virtualSize; comitted += granularity){
			int64_t toCommit = min(uint64_t(comitted + granularity), memory->virtualSize);

			memory->commit(toCommit);

			// Now alloc some other memory to try and break continuity
			CUdeviceptr cptr;
			cuMemAlloc(&cptr, 1'234'567);

			toRemove.push_back(cptr);
		}

		for(CUdeviceptr cptr : toRemove){
			cuMemFree(cptr);
		}
		

		cuMemcpyHtoD(memory->cptr, numbers.data(), 4 * n);

		Scenario scenario = {.label = "Vulkan Imported"};
		for (int i = 0; i < REPETITIONS; i++) {

			cuMemsetD8(cptr_result, 0, 8);
			
			cuCtxSynchronize();
			cuEventRecord(cu_start, 0);
			prog->launch("kernel_u32", {&memory->cptr, &n, &cptr_result}, n);
			cuEventRecord(cu_end, 0);
			cuEventSynchronize(cu_end);

			float duration = 0.0f;
			cuEventElapsedTime(&duration, cu_start, cu_end);

			int64_t result = 0;
			cuMemcpyDtoH(&result, cptr_result, 8);

			double bytesPerMS = double(4 * n) / duration;
			double GBperSec = bytesPerMS / 1'000'000.0;

			scenario.durations.push_back(duration);
		}

		scenarios.push_back(scenario);

		memory->destroy();
		free(memory);
	}

	println("label                            duration      GB/s");
	println("=========================================");
	for(Scenario& scenario : scenarios){
		sort(scenario.durations.begin(), scenario.durations.end());
		double median = scenario.durations[scenario.durations.size() / 2];
		double bytesPerMS = double(4 * n) / median;
		double GBperSec = bytesPerMS / 1'000'000.0;

		println("{:30}    {:7.3f}   {:7.1f}", scenario.label, median, GBperSec);
	}
}

void runBenchmark_packed(vector<uint32_t>& numbers){

	int64_t n = numbers.size();
	int64_t bits = 18;
	
	CUdeviceptr cptr_result;
	cuMemAlloc(&cptr_result, 8);

	CUevent cu_start, cu_end;
	cuEventCreate(&cu_start, CU_EVENT_DEFAULT);
	cuEventCreate(&cu_end, CU_EVENT_DEFAULT);

	vector<Scenario> scenarios;

	{ // SCENARIO: BASELINE

		CUdeviceptr cptr_numbers;
		cuMemAlloc(&cptr_numbers, 4 * n);
		cuMemcpyHtoD(cptr_numbers, numbers.data(), 4 * n);

		Scenario scenario = {.label = "BASELINE"};
		for (int i = 0; i < REPETITIONS; i++) {

			cuMemsetD8(cptr_result, 0, 8);
			
			cuCtxSynchronize();
			cuEventRecord(cu_start, 0);
			prog->launch("kernel_packed", {&cptr_numbers, &n, &bits, &cptr_result}, n);
			cuEventRecord(cu_end, 0);
			cuEventSynchronize(cu_end);

			float duration = 0.0f;
			cuEventElapsedTime(&duration, cu_start, cu_end);

			int64_t result = 0;
			cuMemcpyDtoH(&result, cptr_result, 8);

			double bytesPerMS = double(4 * n) / duration;
			double GBperSec = bytesPerMS / 1'000'000.0;

			scenario.durations.push_back(duration);
		}

		scenarios.push_back(scenario);

		cuMemFree(cptr_numbers);
	}

	{ // SCENARIO: VIRTUAL MEMORY

		CudaVirtualMemory* memory = CudaVirtualMemory::create(4 * n, "virtual");
		// memory->commit(4 * n);

		// Let's mix some smaller commits and other allocations to try to make the virtual range map to non-contiguous physical ranges
		int64_t granularity = 2'097'152;
		vector<CUdeviceptr> toRemove;
		for(int64_t comitted = 0; comitted < memory->size; comitted += granularity){
			int64_t toCommit = min(uint64_t(comitted + granularity), memory->size);

			memory->commit(toCommit);

			// Now alloc some other memory to try and break continuity
			CUdeviceptr cptr;
			cuMemAlloc(&cptr, 1'234'567);
			toRemove.push_back(cptr);
		}

		for(CUdeviceptr cptr : toRemove){
			cuMemFree(cptr);
		}
		

		cuMemcpyHtoD(memory->cptr, numbers.data(), 4 * n);

		Scenario scenario = {.label = "Virtual Memory"};
		for (int i = 0; i < REPETITIONS; i++) {

			cuMemsetD8(cptr_result, 0, 8);
			
			cuCtxSynchronize();
			cuEventRecord(cu_start, 0);
			prog->launch("kernel_packed", {&memory->cptr, &n, &bits, &cptr_result}, n);
			cuEventRecord(cu_end, 0);
			cuEventSynchronize(cu_end);

			float duration = 0.0f;
			cuEventElapsedTime(&duration, cu_start, cu_end);

			int64_t result = 0;
			cuMemcpyDtoH(&result, cptr_result, 8);

			double bytesPerMS = double(4 * n) / duration;
			double GBperSec = bytesPerMS / 1'000'000.0;

			scenario.durations.push_back(duration);
		}

		scenarios.push_back(scenario);

		memory->destroy();
		free(memory);
	}

	{ // SCENARIO: VIRTUAL MEMORY (compressed)

		CudaVirtualMemory* memory = CudaVirtualMemory::create(4 * n, "virtual", true);
		// memory->commit(4 * n);

		// Let's mix some smaller commits and other allocations to try to make the virtual range map to non-contiguous physical ranges
		int64_t granularity = 2'097'152;
		vector<CUdeviceptr> toRemove;
		for(int64_t comitted = 0; comitted < memory->size; comitted += granularity){
			int64_t toCommit = min(uint64_t(comitted + granularity), memory->size);

			memory->commit(toCommit);

			// Now alloc some other memory to try and break continuity
			CUdeviceptr cptr;
			cuMemAlloc(&cptr, 1'234'567);
			toRemove.push_back(cptr);
		}

		for(CUdeviceptr cptr : toRemove){
			cuMemFree(cptr);
		}
		

		cuMemcpyHtoD(memory->cptr, numbers.data(), 4 * n);

		Scenario scenario = {.label = "Virtual Memory (compressed)"};
		for (int i = 0; i < REPETITIONS; i++) {

			cuMemsetD8(cptr_result, 0, 8);
			
			cuCtxSynchronize();
			cuEventRecord(cu_start, 0);
			prog->launch("kernel_packed", {&memory->cptr, &n, &bits, &cptr_result}, n);
			cuEventRecord(cu_end, 0);
			cuEventSynchronize(cu_end);

			float duration = 0.0f;
			cuEventElapsedTime(&duration, cu_start, cu_end);

			int64_t result = 0;
			cuMemcpyDtoH(&result, cptr_result, 8);

			double bytesPerMS = double(4 * n) / duration;
			double GBperSec = bytesPerMS / 1'000'000.0;

			scenario.durations.push_back(duration);
		}

		scenarios.push_back(scenario);

		memory->destroy();
		free(memory);
	}

	{ // SCENARIO: Vulkan Memory
		VulkanCudaSharedMemory* memory = VulkanCudaSharedMemory::create(4 * n, "virtual vulkan");
		// memory->commit(4 * n);

		// Let's mix some smaller commits and other allocations to try to make the virtual range map to non-contiguous physical ranges
		int64_t granularity = 2'097'152;
		vector<CUdeviceptr> toRemove;
		for(int64_t comitted = 0; comitted < memory->virtualSize; comitted += granularity){
			int64_t toCommit = min(uint64_t(comitted + granularity), memory->virtualSize);

			memory->commit(toCommit);

			// Now alloc some other memory to try and break continuity
			CUdeviceptr cptr;
			cuMemAlloc(&cptr, 1'234'567);

			toRemove.push_back(cptr);
		}

		for(CUdeviceptr cptr : toRemove){
			cuMemFree(cptr);
		}
		

		cuMemcpyHtoD(memory->cptr, numbers.data(), 4 * n);

		Scenario scenario = {.label = "Vulkan Imported"};
		for (int i = 0; i < REPETITIONS; i++) {

			cuMemsetD8(cptr_result, 0, 8);
			
			cuCtxSynchronize();
			cuEventRecord(cu_start, 0);
			prog->launch("kernel_packed", {&memory->cptr, &n, &bits, &cptr_result}, n);
			cuEventRecord(cu_end, 0);
			cuEventSynchronize(cu_end);

			float duration = 0.0f;
			cuEventElapsedTime(&duration, cu_start, cu_end);

			int64_t result = 0;
			cuMemcpyDtoH(&result, cptr_result, 8);

			double bytesPerMS = double(4 * n) / duration;
			double GBperSec = bytesPerMS / 1'000'000.0;

			scenario.durations.push_back(duration);
		}

		scenarios.push_back(scenario);

		memory->destroy();
		free(memory);
	}

	println("label                            duration      GB/s");
	println("=========================================");
	for(Scenario& scenario : scenarios){
		sort(scenario.durations.begin(), scenario.durations.end());
		double median = scenario.durations[scenario.durations.size() / 2];


		// double bytesPerMS = double(4 * n) / median;
		double bytesPerMS = double(bits * n / 8) / median;
		double GBperSec = bytesPerMS / 1'000'000.0;

		println("{:30}    {:7.3f}   {:7.1f}", scenario.label, median, GBperSec);
	}


}

int main(int argc, char** argv){

	std::locale::global(getSaneLocale());
	initCuda();
	VKRenderer::init();

	prog = new CudaModularProgram({"./src/kernels/benchmark.cu"});

	int64_t n = 500'000'000;
	std::random_device r;
	std::default_random_engine e(r());
	std::uniform_int_distribution<uint32_t> dist(0, 4294967295);

	vector<uint32_t> numbers_constant(n, 0);
	vector<uint32_t> numbers_simple(n, 0);
	vector<uint32_t> numbers_random(n, 0);
	for(int64_t i = 0; i < n; i++){

		numbers_constant[i] = 123456;
		numbers_simple[i] = i % 262'144; // 2^18 = 262'144

		// C++ random is fairly slow. Create 10 million random values, 
		// then replicate them for the remaining buffer range.
		if(i < 10'000'000){
			numbers_random[i] = dist(e);
		}else{
			numbers_random[i] = numbers_random[i % 10'000'000];
		}
	}

	println("");
	println("==== ALIGNED U32 ====");
	println("");
	println("## Constant Numbers");
	runBenchmark(numbers_constant);

	println("");
	println("## Simple Numbers");
	runBenchmark(numbers_simple);

	println("");
	println("## Random Numbers");
	runBenchmark(numbers_random);

	println("");
	println("==== PACKED ====");
	println("");
	println("## Constant Numbers");
	runBenchmark_packed(numbers_constant);

	println("");
	println("## Simple Numbers");
	runBenchmark_packed(numbers_simple);

	println("");
	println("## Random Numbers");
	runBenchmark_packed(numbers_random);
}
