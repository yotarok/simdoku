#!/bin/zsh

set -eu

USE_PROFILE=1
ENABLE_MULTITHREAD=1
USE_CLANG_LINUX=0

#OPTFLAGS=(-DNDEBUG -O3 -funroll-loops -fomit-frame-pointer -fstrict-overflow -march=native -mtune=native -Ofast -msse4 -msse4.1 -msse4.2 -flto)
OPTFLAGS=(-DNDEBUG -O3 -funroll-loops -fomit-frame-pointer -fstrict-overflow -march=native -mtune=native -Ofast  -msse4 -msse4.1 -msse4.2 -flto -fno-exceptions)

OS=$(uname)
if [ x${OS}x = xLinuxx ] ; then
    echo Linux
    export LIBRARY_PATH=$HOME/env/GCC-6.1.0/lib64:$HOME/env/GCC-6.1.0/lib64
    export LD_LIBRARY_PATH=$HOME/env/GCC-6.1.0/lib64:$HOME/env/GCC-6.1.0/lib64:$LD_LIBRARY_PATH
    export INCLUDE_PATH=$HOME/env/GCC-6.1.0/include
    export PATH=$HOME/env/GCC-6.1.0/bin:$HOME/env/CMake-3.5.2/bin:$PATH

    if [ x${USE_CLANG_LINUX}x = x1x ] ; then
        function compile {
            echo Compile with options: "$@"
            $HOME/llvm_build/bin/clang++  -I/usr/include/c++/4.4.6/x86_64-amazon-linux -I/usr/include/c++/4.4.6 -L$HOME/env/GCC-6.1.0/lib/gcc/x86_64-pc-linux-gnu/6.1.0 -B$HOME/env/GCC-6.1.0/lib/gcc/x86_64-pc-linux-gnu/6.1.0 --std=c++14 "$@"
        }
        function convert_prof {
            $HOME/llvm_build/bin/llvm-profdata merge -o default.profdata default.profraw
        }
    else
        function compile {
            echo Compile with options: "$@"
            g++ --std=c++14 "$@"
        }
        function convert_prof {
        }
    fi

elif [ x${OS}x = xDarwinx ] ; then
    echo Darwin
    function compile {
        echo Compile with options: "$@"
        g++ --std=c++14 "$@"
    }
    function convert_prof {
        xcrun llvm-profdata merge -o default.profdata default*.profraw
    }
fi


if [ x${USE_PROFILE}x = x1x ] ; then
    echo Generate profile
    #python3 codegen.py > ./gened.cc
    compile -DENABLE_MT=0 ${OPTFLAGS} -I/usr/local/include -fprofile-generate -o simdoku ./simdoku.cpp
    cat ./input4profiling.txt | ./simdoku > /dev/null
    convert_prof
    echo Re-compiling
    compile -DENABLE_MT=0 ${OPTFLAGS} -I/usr/local/include -fprofile-use -o simdoku ./simdoku.cpp
else
    echo Compiling
    compile -DENABLE_MT=0 ${OPTFLAGS} -I/usr/local/include -o simdoku ./simdoku.cpp
fi


echo Start testing
cat ./testdata.txt | ./simdoku | tee test.txt
if [ $(diff test.txt ref.txt | wc -l) -ne 0 ] ; then
    echo ERROR
fi

if [ ! -f benchmark.txt ]; then
    wget http://yota.ro/benchmark.txt.gz
    gunzip benchmark.txt.gz
fi

echo Start benchmarking
#cat ../sample.0.10.repeated10000x.txt |  time ./simdoku > /dev/null
/usr/bin/time ./simdoku ./benchmark.txt > /dev/null

if [ x${ENABLE_MULTITHREAD}x = x1x ] ; then
    echo Compiling Multi-threaded version

    if [ x${USE_PROFILE}x = x1x ] ; then
        compile -DENABLE_MT=1 ${OPTFLAGS} -Wno-coverage-mismatch -I/usr/local/include -fprofile-use -o simdoku ./simdoku.cpp -pthread
    else
        compile -DENABLE_MT=1 ${OPTFLAGS} -I/usr/local/include -o simdoku ./simdoku.cpp -pthread
    fi

    echo Start benchmarking with MT
    SUDOKU_NTHREADS=4 /usr/bin/time ./simdoku ./benchmark.txt > /dev/null
fi

