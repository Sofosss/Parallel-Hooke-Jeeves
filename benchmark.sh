#!/usr/bin/env bash

cargo build --release

results_dir="results"
mkdir -p "$results_dir"

binaries=("sequential_hj" "threads_hj" "mpi_hj" "hybrid_hj")
nvars=32
ntrials=65536
runs=5

threads=(2 4 8 16 32)
processes=(2 4 8 16 32)
hybrid_workers=("1 1" "1 2" "1 4" "2 1" "2 2" "2 4" "4 1" "4 2" "4 4" "8 1" "8 2" "8 4" "16 1" "16 2" "16 4" "32 1" "32 2" "32 4")

for binary in "${binaries[@]}"; do
    file="$results_dir/$binary.txt"

    case $binary in
        sequential_hj)
            for _ in $(seq 1 $runs); do
                ./target/release/"$binary" "$nvars" "$ntrials" >> "$file" 2>&1
            done
            ;;
        threads_hj)
            for nthreads in "${threads[@]}"; do
                for _ in $(seq 1 $runs); do
                    ./target/release/"$binary" "$nvars" "$ntrials" "$nthreads" >> "$file" 2>&1
                done
            done
            ;;
        mpi_hj)
            for nprocs in "${processes[@]}"; do
                for _ in $(seq 1 $runs); do
                    mpirun -np "$nprocs" ./target/release/"$binary" "$nvars" "$ntrials" >> "$file" 2>&1
                done
            done
            ;;
        hybrid_hj)
            for workers in "${hybrid_workers[@]}"; do
                nthreads=$(echo $workers | cut -d ' ' -f 2)
                nprocs=$(echo $workers | cut -d ' ' -f 1)
                for _ in $(seq 1 $runs); do
                    mpirun -np "$nprocs" ./target/release/"$binary" "$nvars" "$ntrials" "$nthreads" >> "$file" 2>&1
                done
            done
            ;;
    esac