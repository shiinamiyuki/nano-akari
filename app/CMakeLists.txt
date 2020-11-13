add_executable(akari-import importer.cpp)
target_link_libraries(akari-import PRIVATE nano-akari assimp::assimp)

add_executable(akari-cli cli.cpp)

set_output_dir(akari-import)
set_output_dir(akari-cli)