# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.19

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /snap/clion/158/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /snap/clion/158/bin/cmake/linux/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/yonatan/technion/bigDATA/project/H2_ALSH/methods

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/yonatan/technion/bigDATA/project/H2_ALSH/methods/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/alsh2.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/alsh2.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/alsh2.dir/flags.make

CMakeFiles/alsh2.dir/random.cc.o: CMakeFiles/alsh2.dir/flags.make
CMakeFiles/alsh2.dir/random.cc.o: ../random.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yonatan/technion/bigDATA/project/H2_ALSH/methods/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/alsh2.dir/random.cc.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/alsh2.dir/random.cc.o -c /home/yonatan/technion/bigDATA/project/H2_ALSH/methods/random.cc

CMakeFiles/alsh2.dir/random.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/alsh2.dir/random.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yonatan/technion/bigDATA/project/H2_ALSH/methods/random.cc > CMakeFiles/alsh2.dir/random.cc.i

CMakeFiles/alsh2.dir/random.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/alsh2.dir/random.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yonatan/technion/bigDATA/project/H2_ALSH/methods/random.cc -o CMakeFiles/alsh2.dir/random.cc.s

CMakeFiles/alsh2.dir/pri_queue.cc.o: CMakeFiles/alsh2.dir/flags.make
CMakeFiles/alsh2.dir/pri_queue.cc.o: ../pri_queue.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yonatan/technion/bigDATA/project/H2_ALSH/methods/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/alsh2.dir/pri_queue.cc.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/alsh2.dir/pri_queue.cc.o -c /home/yonatan/technion/bigDATA/project/H2_ALSH/methods/pri_queue.cc

CMakeFiles/alsh2.dir/pri_queue.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/alsh2.dir/pri_queue.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yonatan/technion/bigDATA/project/H2_ALSH/methods/pri_queue.cc > CMakeFiles/alsh2.dir/pri_queue.cc.i

CMakeFiles/alsh2.dir/pri_queue.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/alsh2.dir/pri_queue.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yonatan/technion/bigDATA/project/H2_ALSH/methods/pri_queue.cc -o CMakeFiles/alsh2.dir/pri_queue.cc.s

CMakeFiles/alsh2.dir/util.cc.o: CMakeFiles/alsh2.dir/flags.make
CMakeFiles/alsh2.dir/util.cc.o: ../util.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yonatan/technion/bigDATA/project/H2_ALSH/methods/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/alsh2.dir/util.cc.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/alsh2.dir/util.cc.o -c /home/yonatan/technion/bigDATA/project/H2_ALSH/methods/util.cc

CMakeFiles/alsh2.dir/util.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/alsh2.dir/util.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yonatan/technion/bigDATA/project/H2_ALSH/methods/util.cc > CMakeFiles/alsh2.dir/util.cc.i

CMakeFiles/alsh2.dir/util.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/alsh2.dir/util.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yonatan/technion/bigDATA/project/H2_ALSH/methods/util.cc -o CMakeFiles/alsh2.dir/util.cc.s

CMakeFiles/alsh2.dir/qalsh.cc.o: CMakeFiles/alsh2.dir/flags.make
CMakeFiles/alsh2.dir/qalsh.cc.o: ../qalsh.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yonatan/technion/bigDATA/project/H2_ALSH/methods/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/alsh2.dir/qalsh.cc.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/alsh2.dir/qalsh.cc.o -c /home/yonatan/technion/bigDATA/project/H2_ALSH/methods/qalsh.cc

CMakeFiles/alsh2.dir/qalsh.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/alsh2.dir/qalsh.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yonatan/technion/bigDATA/project/H2_ALSH/methods/qalsh.cc > CMakeFiles/alsh2.dir/qalsh.cc.i

CMakeFiles/alsh2.dir/qalsh.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/alsh2.dir/qalsh.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yonatan/technion/bigDATA/project/H2_ALSH/methods/qalsh.cc -o CMakeFiles/alsh2.dir/qalsh.cc.s

CMakeFiles/alsh2.dir/srp_lsh.cc.o: CMakeFiles/alsh2.dir/flags.make
CMakeFiles/alsh2.dir/srp_lsh.cc.o: ../srp_lsh.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yonatan/technion/bigDATA/project/H2_ALSH/methods/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/alsh2.dir/srp_lsh.cc.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/alsh2.dir/srp_lsh.cc.o -c /home/yonatan/technion/bigDATA/project/H2_ALSH/methods/srp_lsh.cc

CMakeFiles/alsh2.dir/srp_lsh.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/alsh2.dir/srp_lsh.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yonatan/technion/bigDATA/project/H2_ALSH/methods/srp_lsh.cc > CMakeFiles/alsh2.dir/srp_lsh.cc.i

CMakeFiles/alsh2.dir/srp_lsh.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/alsh2.dir/srp_lsh.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yonatan/technion/bigDATA/project/H2_ALSH/methods/srp_lsh.cc -o CMakeFiles/alsh2.dir/srp_lsh.cc.s

CMakeFiles/alsh2.dir/l2_alsh.cc.o: CMakeFiles/alsh2.dir/flags.make
CMakeFiles/alsh2.dir/l2_alsh.cc.o: ../l2_alsh.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yonatan/technion/bigDATA/project/H2_ALSH/methods/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/alsh2.dir/l2_alsh.cc.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/alsh2.dir/l2_alsh.cc.o -c /home/yonatan/technion/bigDATA/project/H2_ALSH/methods/l2_alsh.cc

CMakeFiles/alsh2.dir/l2_alsh.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/alsh2.dir/l2_alsh.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yonatan/technion/bigDATA/project/H2_ALSH/methods/l2_alsh.cc > CMakeFiles/alsh2.dir/l2_alsh.cc.i

CMakeFiles/alsh2.dir/l2_alsh.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/alsh2.dir/l2_alsh.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yonatan/technion/bigDATA/project/H2_ALSH/methods/l2_alsh.cc -o CMakeFiles/alsh2.dir/l2_alsh.cc.s

CMakeFiles/alsh2.dir/l2_alsh2.cc.o: CMakeFiles/alsh2.dir/flags.make
CMakeFiles/alsh2.dir/l2_alsh2.cc.o: ../l2_alsh2.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yonatan/technion/bigDATA/project/H2_ALSH/methods/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/alsh2.dir/l2_alsh2.cc.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/alsh2.dir/l2_alsh2.cc.o -c /home/yonatan/technion/bigDATA/project/H2_ALSH/methods/l2_alsh2.cc

CMakeFiles/alsh2.dir/l2_alsh2.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/alsh2.dir/l2_alsh2.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yonatan/technion/bigDATA/project/H2_ALSH/methods/l2_alsh2.cc > CMakeFiles/alsh2.dir/l2_alsh2.cc.i

CMakeFiles/alsh2.dir/l2_alsh2.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/alsh2.dir/l2_alsh2.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yonatan/technion/bigDATA/project/H2_ALSH/methods/l2_alsh2.cc -o CMakeFiles/alsh2.dir/l2_alsh2.cc.s

CMakeFiles/alsh2.dir/xbox.cc.o: CMakeFiles/alsh2.dir/flags.make
CMakeFiles/alsh2.dir/xbox.cc.o: ../xbox.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yonatan/technion/bigDATA/project/H2_ALSH/methods/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object CMakeFiles/alsh2.dir/xbox.cc.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/alsh2.dir/xbox.cc.o -c /home/yonatan/technion/bigDATA/project/H2_ALSH/methods/xbox.cc

CMakeFiles/alsh2.dir/xbox.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/alsh2.dir/xbox.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yonatan/technion/bigDATA/project/H2_ALSH/methods/xbox.cc > CMakeFiles/alsh2.dir/xbox.cc.i

CMakeFiles/alsh2.dir/xbox.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/alsh2.dir/xbox.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yonatan/technion/bigDATA/project/H2_ALSH/methods/xbox.cc -o CMakeFiles/alsh2.dir/xbox.cc.s

CMakeFiles/alsh2.dir/simple_lsh.cc.o: CMakeFiles/alsh2.dir/flags.make
CMakeFiles/alsh2.dir/simple_lsh.cc.o: ../simple_lsh.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yonatan/technion/bigDATA/project/H2_ALSH/methods/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object CMakeFiles/alsh2.dir/simple_lsh.cc.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/alsh2.dir/simple_lsh.cc.o -c /home/yonatan/technion/bigDATA/project/H2_ALSH/methods/simple_lsh.cc

CMakeFiles/alsh2.dir/simple_lsh.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/alsh2.dir/simple_lsh.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yonatan/technion/bigDATA/project/H2_ALSH/methods/simple_lsh.cc > CMakeFiles/alsh2.dir/simple_lsh.cc.i

CMakeFiles/alsh2.dir/simple_lsh.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/alsh2.dir/simple_lsh.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yonatan/technion/bigDATA/project/H2_ALSH/methods/simple_lsh.cc -o CMakeFiles/alsh2.dir/simple_lsh.cc.s

CMakeFiles/alsh2.dir/sign_alsh.cc.o: CMakeFiles/alsh2.dir/flags.make
CMakeFiles/alsh2.dir/sign_alsh.cc.o: ../sign_alsh.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yonatan/technion/bigDATA/project/H2_ALSH/methods/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building CXX object CMakeFiles/alsh2.dir/sign_alsh.cc.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/alsh2.dir/sign_alsh.cc.o -c /home/yonatan/technion/bigDATA/project/H2_ALSH/methods/sign_alsh.cc

CMakeFiles/alsh2.dir/sign_alsh.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/alsh2.dir/sign_alsh.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yonatan/technion/bigDATA/project/H2_ALSH/methods/sign_alsh.cc > CMakeFiles/alsh2.dir/sign_alsh.cc.i

CMakeFiles/alsh2.dir/sign_alsh.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/alsh2.dir/sign_alsh.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yonatan/technion/bigDATA/project/H2_ALSH/methods/sign_alsh.cc -o CMakeFiles/alsh2.dir/sign_alsh.cc.s

CMakeFiles/alsh2.dir/h2_alsh.cc.o: CMakeFiles/alsh2.dir/flags.make
CMakeFiles/alsh2.dir/h2_alsh.cc.o: ../h2_alsh.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yonatan/technion/bigDATA/project/H2_ALSH/methods/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Building CXX object CMakeFiles/alsh2.dir/h2_alsh.cc.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/alsh2.dir/h2_alsh.cc.o -c /home/yonatan/technion/bigDATA/project/H2_ALSH/methods/h2_alsh.cc

CMakeFiles/alsh2.dir/h2_alsh.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/alsh2.dir/h2_alsh.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yonatan/technion/bigDATA/project/H2_ALSH/methods/h2_alsh.cc > CMakeFiles/alsh2.dir/h2_alsh.cc.i

CMakeFiles/alsh2.dir/h2_alsh.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/alsh2.dir/h2_alsh.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yonatan/technion/bigDATA/project/H2_ALSH/methods/h2_alsh.cc -o CMakeFiles/alsh2.dir/h2_alsh.cc.s

CMakeFiles/alsh2.dir/amips.cc.o: CMakeFiles/alsh2.dir/flags.make
CMakeFiles/alsh2.dir/amips.cc.o: ../amips.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yonatan/technion/bigDATA/project/H2_ALSH/methods/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_12) "Building CXX object CMakeFiles/alsh2.dir/amips.cc.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/alsh2.dir/amips.cc.o -c /home/yonatan/technion/bigDATA/project/H2_ALSH/methods/amips.cc

CMakeFiles/alsh2.dir/amips.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/alsh2.dir/amips.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yonatan/technion/bigDATA/project/H2_ALSH/methods/amips.cc > CMakeFiles/alsh2.dir/amips.cc.i

CMakeFiles/alsh2.dir/amips.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/alsh2.dir/amips.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yonatan/technion/bigDATA/project/H2_ALSH/methods/amips.cc -o CMakeFiles/alsh2.dir/amips.cc.s

CMakeFiles/alsh2.dir/pre_recall.cc.o: CMakeFiles/alsh2.dir/flags.make
CMakeFiles/alsh2.dir/pre_recall.cc.o: ../pre_recall.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yonatan/technion/bigDATA/project/H2_ALSH/methods/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_13) "Building CXX object CMakeFiles/alsh2.dir/pre_recall.cc.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/alsh2.dir/pre_recall.cc.o -c /home/yonatan/technion/bigDATA/project/H2_ALSH/methods/pre_recall.cc

CMakeFiles/alsh2.dir/pre_recall.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/alsh2.dir/pre_recall.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yonatan/technion/bigDATA/project/H2_ALSH/methods/pre_recall.cc > CMakeFiles/alsh2.dir/pre_recall.cc.i

CMakeFiles/alsh2.dir/pre_recall.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/alsh2.dir/pre_recall.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yonatan/technion/bigDATA/project/H2_ALSH/methods/pre_recall.cc -o CMakeFiles/alsh2.dir/pre_recall.cc.s

CMakeFiles/alsh2.dir/main2.cc.o: CMakeFiles/alsh2.dir/flags.make
CMakeFiles/alsh2.dir/main2.cc.o: ../main2.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yonatan/technion/bigDATA/project/H2_ALSH/methods/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_14) "Building CXX object CMakeFiles/alsh2.dir/main2.cc.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/alsh2.dir/main2.cc.o -c /home/yonatan/technion/bigDATA/project/H2_ALSH/methods/main2.cc

CMakeFiles/alsh2.dir/main2.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/alsh2.dir/main2.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yonatan/technion/bigDATA/project/H2_ALSH/methods/main2.cc > CMakeFiles/alsh2.dir/main2.cc.i

CMakeFiles/alsh2.dir/main2.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/alsh2.dir/main2.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yonatan/technion/bigDATA/project/H2_ALSH/methods/main2.cc -o CMakeFiles/alsh2.dir/main2.cc.s

# Object files for target alsh2
alsh2_OBJECTS = \
"CMakeFiles/alsh2.dir/random.cc.o" \
"CMakeFiles/alsh2.dir/pri_queue.cc.o" \
"CMakeFiles/alsh2.dir/util.cc.o" \
"CMakeFiles/alsh2.dir/qalsh.cc.o" \
"CMakeFiles/alsh2.dir/srp_lsh.cc.o" \
"CMakeFiles/alsh2.dir/l2_alsh.cc.o" \
"CMakeFiles/alsh2.dir/l2_alsh2.cc.o" \
"CMakeFiles/alsh2.dir/xbox.cc.o" \
"CMakeFiles/alsh2.dir/simple_lsh.cc.o" \
"CMakeFiles/alsh2.dir/sign_alsh.cc.o" \
"CMakeFiles/alsh2.dir/h2_alsh.cc.o" \
"CMakeFiles/alsh2.dir/amips.cc.o" \
"CMakeFiles/alsh2.dir/pre_recall.cc.o" \
"CMakeFiles/alsh2.dir/main2.cc.o"

# External object files for target alsh2
alsh2_EXTERNAL_OBJECTS =

alsh2: CMakeFiles/alsh2.dir/random.cc.o
alsh2: CMakeFiles/alsh2.dir/pri_queue.cc.o
alsh2: CMakeFiles/alsh2.dir/util.cc.o
alsh2: CMakeFiles/alsh2.dir/qalsh.cc.o
alsh2: CMakeFiles/alsh2.dir/srp_lsh.cc.o
alsh2: CMakeFiles/alsh2.dir/l2_alsh.cc.o
alsh2: CMakeFiles/alsh2.dir/l2_alsh2.cc.o
alsh2: CMakeFiles/alsh2.dir/xbox.cc.o
alsh2: CMakeFiles/alsh2.dir/simple_lsh.cc.o
alsh2: CMakeFiles/alsh2.dir/sign_alsh.cc.o
alsh2: CMakeFiles/alsh2.dir/h2_alsh.cc.o
alsh2: CMakeFiles/alsh2.dir/amips.cc.o
alsh2: CMakeFiles/alsh2.dir/pre_recall.cc.o
alsh2: CMakeFiles/alsh2.dir/main2.cc.o
alsh2: CMakeFiles/alsh2.dir/build.make
alsh2: CMakeFiles/alsh2.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/yonatan/technion/bigDATA/project/H2_ALSH/methods/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_15) "Linking CXX executable alsh2"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/alsh2.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/alsh2.dir/build: alsh2

.PHONY : CMakeFiles/alsh2.dir/build

CMakeFiles/alsh2.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/alsh2.dir/cmake_clean.cmake
.PHONY : CMakeFiles/alsh2.dir/clean

CMakeFiles/alsh2.dir/depend:
	cd /home/yonatan/technion/bigDATA/project/H2_ALSH/methods/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/yonatan/technion/bigDATA/project/H2_ALSH/methods /home/yonatan/technion/bigDATA/project/H2_ALSH/methods /home/yonatan/technion/bigDATA/project/H2_ALSH/methods/cmake-build-debug /home/yonatan/technion/bigDATA/project/H2_ALSH/methods/cmake-build-debug /home/yonatan/technion/bigDATA/project/H2_ALSH/methods/cmake-build-debug/CMakeFiles/alsh2.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/alsh2.dir/depend

