cmake_minimum_required(VERSION 3.25)
project(har-cnn
    VERSION 0.1.0
    DESCRIPTION "Human Action Recognition using CNN"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(MSVC)
    add_compile_options(/W4 /permissive-)
else()
    add_compile_options(
        -Wall
        -Wextra
        -Wpedantic
        -Werror=return-type
    )
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
endif()

option(HAR_WITH_OPENCV "Enable OpenCV video frame extraction" ON)

add_executable(har_cnn main.cpp)
target_include_directories(har_cnn PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/third_party
)

if(HAR_WITH_OPENCV)
    find_package(OpenCV QUIET COMPONENTS core imgproc imgcodecs videoio)
    if(OpenCV_FOUND)
        target_compile_definitions(har_cnn PRIVATE HAR_HAS_OPENCV=1)
        target_include_directories(har_cnn PRIVATE ${OpenCV_INCLUDE_DIRS})
        target_link_libraries(har_cnn PRIVATE ${OpenCV_LIBS})
        message(STATUS "OpenCV found: ${OpenCV_VERSION}")
    else()
        message(WARNING "OpenCV not found; video extraction disabled. Install OpenCV or set HAR_WITH_OPENCV=OFF.")
    endif()
endif()

option(BUILD_TESTS "Build tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

install(TARGETS har_cnn DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
