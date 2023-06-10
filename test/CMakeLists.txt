find_package(GTest REQUIRED)

include_directories(${GTEST_INCLUDE_DIRS})

set(TEST_PROJECT_NAME ${PROJECT_NAME}_test)
add_executable(${TEST_PROJECT_NAME} SurfaceInterpolationTest.cpp)
target_include_directories(${TEST_PROJECT_NAME} PRIVATE ../include)
add_test(NAME ${TEST_PROJECT_NAME} COMMAND ${TEST_PROJECT_NAME})

target_link_libraries(${TEST_PROJECT_NAME} ${GTEST_LIBRARIES} ${PROJECT_NAME})