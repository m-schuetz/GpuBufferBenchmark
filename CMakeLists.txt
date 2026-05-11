cmake_minimum_required(VERSION 3.5)

set(TARGET_NAME BufferBench) 
project(${TARGET_NAME} LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 23)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

include(cmake/common.cmake)

# --------------------------------------------------------
# Source files (C++ and CUDA)
# --------------------------------------------------------
file(GLOB PROJECT_SOURCES CONFIGURE_DEPENDS 
	src/*.[ch]pp
	src/*.cu
)

add_executable(${TARGET_NAME} ${PROJECT_SOURCES})
if (MSVC)
	target_compile_options(${TARGET_NAME} PRIVATE
		$<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
		$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler /Zc:preprocessor>
	)
endif()

target_compile_definitions(${TARGET_NAME} PUBLIC _HAS_STD_BYTE=0)

if (WIN32)
	target_compile_definitions(${TARGET_NAME} PUBLIC VK_USE_PLATFORM_WIN32_KHR)
	set_property(TARGET ${TARGET_NAME} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "$(SolutionDir)..")
endif()

# Include paths
target_include_directories(${TARGET_NAME} PRIVATE
	./src
	./libs
)

# Add other third-party libs
ADD_CUDA(${TARGET_NAME})
ADD_VULKAN(${TARGET_NAME})
ADD_GLM(${TARGET_NAME})

# --------------------------------------------------------
# CUDA configuration
# --------------------------------------------------------
# Choose GPU architectures (adjust for your machine)
set(CMAKE_CUDA_ARCHITECTURES 75 86 89 90)

# Enable separable compilation (for multi-file CUDA builds)
set_target_properties(${TARGET_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Treat warnings as errors (optional)
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Werror all-warnings")

# set proper startup project
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${TARGET_NAME})

