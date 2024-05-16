
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)

project(libsegmentator)

find_package(Torch REQUIRED)
find_package(PythonLibs REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")

set(SOURCES_LIB segmentator.cpp)
add_library(libsegmentator SHARED ${SOURCES_LIB})
set_target_properties(libsegmentator PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(libsegmentator PROPERTIES PREFIX "")
target_link_libraries(libsegmentator ${TORCH_PYTHON_LIBRARY})
target_compile_definitions(libsegmentator PUBLIC TORCH_EXTENSION_NAME=libsegmentator)
target_compile_definitions(libsegmentator PUBLIC TORCH_API_INCLUDE_EXTENSION_H)
target_compile_definitions(libsegmentator PUBLIC ${TORCH_CXX_FLAGS})
target_include_directories(libsegmentator PUBLIC ${TORCH_INCLUDE_DIRS})
target_include_directories(libsegmentator PUBLIC ${PYTHON_INCLUDE_DIRS})


# to install
set(BARE_PROJECT_NAME segmentator)
install(CODE "execute_process(                                        \
    COMMAND ${CMAKE_COMMAND} -E create_symlink                        \
    ${PROJECT_SOURCE_DIR}/../../${BARE_PROJECT_NAME}                  \
    ${CMAKE_INSTALL_PREFIX}/${BARE_PROJECT_NAME}                      \
    )"
)

