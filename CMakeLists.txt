# CMakeLists files in this project can
# refer to the root source directory of the project as ${HELLO_SOURCE_DIR} and
# to the root binary directory of the project as ${HELLO_BINARY_DIR}.
cmake_minimum_required(VERSION 3.3)
project (Nightmare_Game VERSION 0.1 LANGUAGES CXX C)
find_package(Vulkan REQUIRED FATAL_ERROR)

#Defines
add_compile_definitions(DEBUG)

# vulkan-1 library for build Vulkan application.


if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
  set(VULKAN_LIB_LIST "vulkan-1")	
  message(STATUS "Attempting to locate Vulkan SDK using manual path......")
  set(VULKAN_SDK "C:/VulkanSDK")
  set(VULKAN_VERSION "1.2.154.1")
  set(VULKAN_PATH "${VULKAN_SDK}/${VULKAN_VERSION}")
  message(STATUS "Using manual specified path: ${VULKAN_PATH}")

endif()

# Include Vulkan header files from Vulkan SDK
include_directories(AFTER ${VULKAN_PATH}/Include)
# Link directory for vulkan-1
link_directories(${VULKAN_PATH}/Bin;${VULKAN_PATH}/Lib;)

#libraries
add_library(GLFW STATIC IMPORTED) # or STATIC instead of SHARED
if(APPLE)
set_target_properties(GLFW PROPERTIES
  IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/../glfw/bin/src/Debug/libglfw3.a"
  INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/../glfw/include/GLFW"
)
else()
set_target_properties(GLFW PROPERTIES
  IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/../glfw/build/src/Debug/glfw3.lib"
  INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/../glfw/include/GLFW"
)
endif()

set(SRC_DIR "${CMAKE_SOURCE_DIR}/src")
# Sources
set( SOURCES
  "${SRC_DIR}/main.cpp"
  "${SRC_DIR}/imgui.cpp"
  "${SRC_DIR}/imgui_draw.cpp"
  "${SRC_DIR}/imgui_widgets.cpp"
  "${SRC_DIR}/imgui_impl_vulkan.cpp"
  "${SRC_DIR}/imgui_impl_glfw.cpp"
  "${SRC_DIR}/imgui_demo.cpp"
  )
# "Headers
file(GLOB HEADERS "includes/vk_mem_alloc/vk_mem_alloc.h")
file(GLOB VULKAN_HEADERS "includes/vulkan/*.h")
file(GLOB RENDERER_HEADERS "includes/renderer/*.h")
file(GLOB GLM_HEADERS "includes/glm/*.h")
file(GLOB GLFW_HEADERS "includes/GLFW/*.h")
file(GLOB GLFW_HEADERS GLFW_HEADERS "includes/GLFW_internal/*.h")
file(GLOB STB_HEADERS "includes/stb/stb.h")
file(GLOB TINY_OBJ_LOADER_HEADERS "includes/tiny_obj_loader/tiny_obj_loader.h")
file(GLOB IMGUI_HEADERS "includes/imgui/*.h")

file(GLOB GLSL_FILES "src/glsl/*")
add_executable (Nightmare ${SOURCES} ${HEADERS} ${VULKAN_HEADERS} ${RENDERER_HEADERS} ${GLM_HEADERS} ${GLFW_HEADERS} ${STB_HEADERS} ${TINY_OBJ_LOADER_HEADERS} ${IMGUI_HEADERS} ${GLSL_FILES})
set_target_properties(Nightmare PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin"
  CXX_STANDARD 17
)
target_include_directories(Nightmare PUBLIC includes)
target_include_directories(Nightmare PRIVATE ${VULKAN_INCLUDE_DIRS})

target_link_libraries (Nightmare PUBLIC GLFW)
if(WIN32)
target_link_libraries (Nightmare PUBLIC ${VULKAN_LIB_LIST})
elseif(APPLE)
target_link_libraries(Nightmare PUBLIC Vulkan::Vulkan)
target_link_libraries(Nightmare PUBLIC "-framework Cocoa")
target_link_libraries(Nightmare PUBLIC "-framework IoKit")
endif()