#! /usr/bin/env bash

if [ -z "$1" ]; then
  echo "Please specify the binary to run: sequential_hj, threads_hj, mpi_hj or hybrid_hj."
  exit 1
fi

binary="$1"

build_binary() {
  binary="$1"
  if [ ! -f "./target/release/$binary" ]; then
    echo "Executable $binary not found. Initiating build process..."
    cargo build --release --bin "$binary"
  else
    echo "Executable $binary already exists. Skipping build process."
  fi
}

build_binary "$binary"

case $binary in
  sequential_hj)
    ./target/release/$binary "$2" "$3"
    ;;
  threads_hj)
    ./target/release/$binary "$2" "$3" "$4"
    ;;
  mpi_hj)
      mpirun -np "$4" ./target/release/$binary "$2" "$3"
    ;;
  hybrid_hj)
      mpirun -np "$4" ./target/release/$binary "$2" "$3" "$5"
    ;;
  *)
    echo "Invalid binary specified. Please choose from: sequential_hj, threads_hj, mpi_hj or hybrid_hj."
    exit 1
    ;;
esac