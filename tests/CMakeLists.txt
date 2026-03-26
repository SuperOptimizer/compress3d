add_executable(test_compress3d test_compress3d.c)
target_link_libraries(test_compress3d PRIVATE compress3d)
add_test(NAME test_compress3d COMMAND test_compress3d)

add_executable(test_multiscale test_multiscale.c)
target_link_libraries(test_multiscale PRIVATE compress3d)
add_test(NAME test_multiscale COMMAND test_multiscale)

add_executable(test_udp test_udp.c)
target_link_libraries(test_udp PRIVATE compress3d)
add_test(NAME test_udp COMMAND test_udp)

add_executable(test_metrics test_metrics.c)
target_link_libraries(test_metrics PRIVATE compress3d)
add_test(NAME test_metrics COMMAND test_metrics)

add_executable(test_shard test_shard.c)
target_link_libraries(test_shard PRIVATE compress3d)
add_test(NAME test_shard COMMAND test_shard)

add_executable(test_advanced test_advanced.c)
target_link_libraries(test_advanced PRIVATE compress3d)
add_test(NAME test_advanced COMMAND test_advanced)
