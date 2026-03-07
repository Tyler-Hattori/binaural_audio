cmake_minimum_required(VERSION 3.16)
project(BinauralRenderer)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# For Apple Silicon Homebrew
if(APPLE)
    list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew")
endif()

# Find packages
find_package(PkgConfig REQUIRED)
find_package(SndFile REQUIRED)
find_package(yaml-cpp REQUIRED)
pkg_check_modules(MYSOFA REQUIRED IMPORTED_TARGET libmysofa)
pkg_check_modules(FFTW3 REQUIRED IMPORTED_TARGET fftw3f)
pkg_check_modules(RTAUDIO REQUIRED IMPORTED_TARGET rtaudio)

# Add source files
file(GLOB SRC_FILES src/*.cpp)
add_executable(binaural ${SRC_FILES})

# Include directories
target_include_directories(binaural PRIVATE
    ${MYSOFA_INCLUDE_DIRS}
    ${FFTW3_INCLUDE_DIRS}
)

# Link libraries using the PkgConfig targets
target_link_libraries(binaural PRIVATE
    PkgConfig::MYSOFA
    PkgConfig::FFTW3
    SndFile::sndfile
    PkgConfig::RTAUDIO
    yaml-cpp::yaml-cpp
)

# macOS rpath fix
set_target_properties(binaural PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
)

# Add the Homebrew lib directory to the RPATH
if(APPLE)
    # Search in both Homebrew and standard manual install paths
    list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew" "/usr/local")
    
    set_target_properties(binaural PROPERTIES
        INSTALL_RPATH "/opt/homebrew/lib;/usr/local/lib"
        BUILD_WITH_INSTALL_RPATH TRUE
    )

    # This ensures it doesn't default to the 'broken' Homebrew @rpath
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
endif()