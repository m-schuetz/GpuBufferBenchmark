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
#include "CudaVulkanSharedMemory.h"
#include "VulkanCudaSharedMemory.h"
#include "MemoryManager.h"

#include "Runtime.h"
#include "MappedFile.h"

using namespace std; // YOLO

CUcontext context;

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

void runBenchmark(){

	int64_t n = 500'000'000;
	CUdeviceptr cptr_numbers = MemoryManager::alloc(4 * n, "number");
	CUdeviceptr cptr_result = MemoryManager::alloc(8, "result");

	vector<uint32_t> numbers(n, 0);
	for(int64_t i = 0; i < n; i++){
		// numbers[i] = 0;
		// numbers[i] = i;
		// numbers[i] = random(0.0, 1234567.0);
		numbers[i] = random(0.0, 4294967296.0);
	}

	CudaModularProgram* prog = new CudaModularProgram({"./src/kernels/test.cu"});

	CUevent cu_start, cu_end;
	cuEventCreate(&cu_start, CU_EVENT_DEFAULT);
	cuEventCreate(&cu_end, CU_EVENT_DEFAULT);

	constexpr int REPETITIONS = 50;
	vector<Scenario> scenarios;

	{ // SCENARIO: BASELINE

		CUdeviceptr cptr_numbers = MemoryManager::alloc(4 * n, "number");
		cuMemcpyHtoD(cptr_numbers, numbers.data(), 4 * n);

		Scenario scenario = {.label = "BASELINE"};
		for (int i = 0; i < REPETITIONS; i++) {

			cuMemsetD8(cptr_result, 0, 8);
			
			cuCtxSynchronize();
			cuEventRecord(cu_start, 0);
			prog->launch("kernel_test", {&cptr_numbers, &n, &cptr_result}, n);
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

		MemoryManager::free(cptr_numbers);
	}

	{ // SCENARIO: VIRTUAL MEMORY

		// CudaVirtualMemory* memory = MemoryManager::allocVirtualCuda(4 * n, "virtual");
		CudaVirtualMemory* memory = CudaVirtualMemory::create(4 * n, "virtual");
		// memory->commit(4 * n);

		// Let's mix some smaller commits and other allocations to try to make the virtual range map to non-contiguous physical ranges
		int64_t granularity = 2'097'152;
		vector<CUdeviceptr> toRemove;
		for(int64_t comitted = 0; comitted < memory->size; comitted += granularity){
			int64_t toCommit = min(uint64_t(comitted + granularity), memory->size);

			memory->commit(toCommit);

			// Now alloc some other memory to try and break continuity
			CUdeviceptr cptr = MemoryManager::alloc(1'234'567, "whatever");
			toRemove.push_back(cptr);
		}

		for(CUdeviceptr cptr : toRemove){
			MemoryManager::free(cptr);
		}
		

		cuMemcpyHtoD(memory->cptr, numbers.data(), 4 * n);

		Scenario scenario = {.label = "Virtual Memory"};
		for (int i = 0; i < REPETITIONS; i++) {

			cuMemsetD8(cptr_result, 0, 8);
			
			cuCtxSynchronize();
			cuEventRecord(cu_start, 0);
			prog->launch("kernel_test", {&memory->cptr, &n, &cptr_result}, n);
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

		free(memory);
		// MemoryManager::free(cptr_numbers);
	}

	{ // SCENARIO: VIRTUAL MEMORY (compressed)

		// CudaVirtualMemory* memory = MemoryManager::allocVirtualCuda(4 * n, "virtual", true);
		CudaVirtualMemory* memory = CudaVirtualMemory::create(4 * n, "virtual", true);
		// memory->commit(4 * n);

		// Let's mix some smaller commits and other allocations to try to make the virtual range map to non-contiguous physical ranges
		int64_t granularity = 2'097'152;
		vector<CUdeviceptr> toRemove;
		for(int64_t comitted = 0; comitted < memory->size; comitted += granularity){
			int64_t toCommit = min(uint64_t(comitted + granularity), memory->size);

			memory->commit(toCommit);

			// Now alloc some other memory to try and break continuity
			CUdeviceptr cptr = MemoryManager::alloc(1'234'567, "whatever");
			toRemove.push_back(cptr);
		}

		for(CUdeviceptr cptr : toRemove){
			MemoryManager::free(cptr);
		}
		

		cuMemcpyHtoD(memory->cptr, numbers.data(), 4 * n);

		Scenario scenario = {.label = "Virtual Memory (compressed)"};
		for (int i = 0; i < REPETITIONS; i++) {

			cuMemsetD8(cptr_result, 0, 8);
			
			cuCtxSynchronize();
			cuEventRecord(cu_start, 0);
			prog->launch("kernel_test", {&memory->cptr, &n, &cptr_result}, n);
			cuEventRecord(cu_end, 0);
			cuEventSynchronize(cu_end);

			float duration = 0.0f;
			cuEventElapsedTime(&duration, cu_start, cu_end);

			int64_t result = 0;
			cuMemcpyDtoH(&result, cptr_result, 8);

			double bytesPerMS = double(4 * n) / duration;
			double GBperSec = bytesPerMS / 1'000'000.0;

			//println("{:7.3f}   {:7.1f}   {}", duration, GBperSec, result);

			scenario.durations.push_back(duration);
		}

		scenarios.push_back(scenario);

		free(memory);
		// MemoryManager::free(cptr_numbers);
	}

	{ // SCENARIO: VIRTUAL MEMORY (compressed)
		VulkanCudaSharedMemory* memory = VulkanCudaSharedMemory::create(4 * n, "virtual vulkan");
		// memory->commit(4 * n);

		// Let's mix some smaller commits and other allocations to try to make the virtual range map to non-contiguous physical ranges
		int64_t granularity = 2'097'152;
		vector<CUdeviceptr> toRemove;
		for(int64_t comitted = 0; comitted < memory->virtualSize; comitted += granularity){
			int64_t toCommit = min(uint64_t(comitted + granularity), memory->virtualSize);

			memory->commit(toCommit);

			// Now alloc some other memory to try and break continuity
			CUdeviceptr cptr = MemoryManager::alloc(1'234'567, "whatever");
			toRemove.push_back(cptr);
		}

		for(CUdeviceptr cptr : toRemove){
			MemoryManager::free(cptr);
		}
		

		cuMemcpyHtoD(memory->cptr, numbers.data(), 4 * n);

		Scenario scenario = {.label = "Vulkan Imported"};
		for (int i = 0; i < REPETITIONS; i++) {

			cuMemsetD8(cptr_result, 0, 8);
			
			cuCtxSynchronize();
			cuEventRecord(cu_start, 0);
			prog->launch("kernel_test", {&memory->cptr, &n, &cptr_result}, n);
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

int main(int argc, char** argv){

	std::locale::global(getSaneLocale());
	initCuda();
	VKRenderer::init();

	runBenchmark();
}
