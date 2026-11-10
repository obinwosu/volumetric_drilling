cmake_minimum_required(VERSION 3.10)
project(ambf_openxr_eyepiece)

set(CMAKE_CXX_STANDARD 17)
set(OpenGL_GL_PREFERENCE LEGACY)

# AMBF's exported target requires Boost::filesystem in the link interface.
find_package(Boost REQUIRED COMPONENTS filesystem program_options)
find_package(AMBF REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENXR REQUIRED openxr)

find_package(X11 REQUIRED)
find_package(OpenGL REQUIRED)

include_directories(${AMBF_INCLUDE_DIRS} ${OPENXR_INCLUDE_DIRS} ${X11_INCLUDE_DIR} ${Boost_INCLUDE_DIRS})
link_directories(${AMBF_LIBRARY_DIRS} ${OPENXR_LIBRARY_DIRS})
add_definitions(${AMBF_DEFINITIONS})

add_library(${CMAKE_PROJECT_NAME} SHARED
  plugins/object/ambf_openxr_eyepiece.cpp
  plugins/object/ambf_openxr_eyepiece.h
)

target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE
  XR_USE_GRAPHICS_API_OPENGL
  XR_USE_PLATFORM_XLIB
)

target_link_libraries(${CMAKE_PROJECT_NAME}
  ${AMBF_LIBRARIES}
  ${OPENXR_LIBRARIES}
  ${X11_LIBRARIES}
  OpenGL::GL
  Boost::filesystem
  ${CMAKE_DL_LIBS}
)

set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY POSITION_INDEPENDENT_CODE TRUE)
