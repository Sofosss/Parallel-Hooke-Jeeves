# 💥 Parallel Hooke–Jeeves Optimization in Rust

[![Rust 1.82](https://img.shields.io/badge/Rust-1.82-%23006B3F.svg?logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/License-MIT-FFDEAD)](https://opensource.org/licenses/MIT)<br>
![Hooke-Jeeves](https://img.shields.io/badge/Hooke--Jeeves-FF7518?style=flat&logo=dna&logoColor=white)
![Parallel Programming](https://img.shields.io/badge/Parallel%20Programming-001594?style=flat&logo=dna&logoColor=white)


A high-performance implementation of the [Hooke-Jeeves algorithm][hooke-jeeves-link] in **Rust**, exploring three different parallelism schemes: (a) **multithreading**, (b) **multiprocessing (MPI)**, and (c) **hybrid** approaches.


## Table of Contents
* [Introduction](#introduction)
  * [Global Optimization](#global-optimization)
  * [Hooke-Jeeves Algorithm](#hooke-jeeves-algorithm)
* [Prerequisites & Installation](#prerequisites--installation)
* [Usage](#usage)
  * [Running Different Parallelism Schemes](#running-different-parallelism-schemes)
* [Software Overview](#software-overview)
* [Performance Evaluation](#performance-evaluation)
* [Contact](#contact)


## Introduction

### Global Optimization

**Global optimization** is a fundamental problem in applied mathematics with wide-ranging applications in fields such as economics, chemistry, biology and engineering. Unlike local optimization, which aims to find the minimum or maximum of an objective function in a localized region, global optimization seeks to identify the **global extremum** (minimum or maximum) of the objective function across the entire solution space, while avoiding the limitations imposed by local extrema.

A commonly employed strategy in global optimization is **multi-start local search**, in which a local optimization algorithm is initialized from several randomly selected starting points. This approach increases the likelihood of locating the global minimum by facilitating the exploration of multiple regions within the solution space and reducing the risk of convergence to poor local optima. The success of this scheme is contingent upon the ability of the local optimization algorithm to effectively traverse diverse areas of the search space. Parallelism is crucial to this process, as it allows for the concurrent execution of multiple local searches, thereby enhancing computational performance. By leveraging parallelism, the efficiency of the global optimization process is significantly improved, resulting in faster convergence and a higher likelihood of successfully identifying the global optimum.

### Hooke-Jeeves Algorithm

Originally introduced in the paper [*"Direct Search" Solution of Numerical and Statistical Problems*](https://dl.acm.org/doi/10.1145/321062.321069) by Robert Hooke and T. A. Jeeves in 1961, the [Hooke-Jeeves][hooke-jeeves-link] direct search method is a well-known **derivative-free** optimization algorithm. It follows a pattern search strategy that iteratively explores the search space by moving along predefined directions from the current point. If a direction yields a better objective function value, the current point is updated accordingly; otherwise, the search parameters are adjusted and the exploration continues. Due to its derivative-free nature, the method is particularly suitable for optimization problems involving non-differentiable, discontinuous, or computationally expensive objective functions, as it can operate effectively without requiring gradient information.

> [!IMPORTANT]
> In this work, various parallelism schemes are applied at the **trial level** to improve the efficiency of the Hooke-Jeeves method in locating the global minimum of an objective function. By distributing optimization trials across multiple workers, independent searches can be performed simultaneously in different regions of the solution space. This approach reduces computation time and improves the scalability of the Hooke-Jeeves method, enabling efficient execution on both single-node and multi-node computing platforms.


> [!NOTE]
> The **Rosenbrock function** (also known as the banana function) is chosen as the objective function for this optimization task. It is mathematically defined as:
>
> $$f(x) = \sum_{i=1}^{n-1} \left( 100 \cdot (x_{i+1} - x_i^2)^2 + (1 - x_i)^2 \right)$$
>
> The Rosenbrock function is a common benchmark for optimization algorithms, having a global minimum at $(1, 1, \dots, 1)$, where the function value is 0. This function is non-convex, which poses challenges for optimization methods, making it a suitable test case for global optimization algorithms. For more details, refer [here][rosenbrock-link].

> [!NOTE]  
> An implementation of the Hooke-Jeeves method in C is provided by [NetLib][netlib-link]. This implementation searches for a point $X$ where the nonlinear objective function $f(X)$ converges to a local minimum. The objective function is scalar-valued and defined as $f: \mathbb{R}^n \to \mathbb{R}$. The Rust implementation of the Hooke-Jeeves method developed in this project is based on this C implementation from *NetLib*.

## Prerequisites & Installation

To get started with this project, ensure that the following prerequisites are met:

1. **Rust** (edition >= **2021**) is installed on your system.    

2. An **MPI** implementation is available on your system, such as [OpenMPI][open-mpi-link] or [MPICH][mpich-link].

> [!WARNING]  
> In addition to installing an MPI implementation, ensure that the `mpirun` command is available in your system's `PATH`. This command is required for launching MPI-based binaries.

Clone the repository:
```bash
git clone https://github.com/Sofosss/Parallel-Hooke-Jeeves.git
cd Parallel-Hooke-Jeeves
```

The project includes **four** binary crates, each implementing a different execution scheme of the Hooke-Jeeves algorithm ([multithreading](./Hooke_Jeeves/parallel/threads.rs), [multiprocessing](./Hooke_Jeeves/parallel/mpi.rs), [hybrid](./Hooke_Jeeves/parallel/hybrid.rs), and [sequential](./Hooke_Jeeves/sequential.rs)). Install them with:
```bash
cargo install --path .
```

> [!WARNING]
> The binaries will be installed in the `~/.cargo/bin` directory. Make sure that this directory is included in your `PATH`.

Alternatively, build and copy the binaries manually:
```bash
cargo build --release

# Repeat the cp command for each binary crate as required
cp ./target/release/<binary-name> /usr/local/bin 
```

Verify the installation by checking the version of each binary:
```bash
<binary-name> --version
```

## Usage

All binaries require two mandatory arguments:  
- `nvars`: Number of variables in the optimization problem.  
- `ntrials`: Number of trials to perform.  

### Running Different Parallelism Schemes

- **Multithreading**

    Requires an additional argument specifying the number of **threads**.  
  ```bash
  threads_hj <nvars> <ntrials> <nthreads>
  ```
- **Multiprocessing** (MPI)
    
    Requires execution via `mpirun` or a compatible MPI runner. Specify the number of **processes** using `-n` or `-np`.
    ```bash
    mpirun -n <nprocesses> mpi_hj <nvars> <ntrials>
    ```
- **Hybrid** (MPI + threads)
    
    Requires both the number of **processes** and **threads**.
    ```bash
    mpirun -n <nprocesses> hybrid_hj <nvars> <ntrials> <nthreads>
    ```

> [!NOTE]
> If you prefer not to manually build or install the binaries using `cargo`, you can use the provided [script][run-link]. This script automates the process of building and running the required binary for any of the implemented schemes. To use the script, ensure it is executable by running the command `chmod +x run.sh`. Once the script is ready, you can run it by specifying the binary name along with the required arguments.

## Software Overview

Three parallelism schemes for the Hooke-Jeeves algorithm are implemented, each designed to efficiently perform optimization trials for locating the global minimum of the Rosenbrock objective function:

- **Multithreading**: Parallel execution is achieved using the [thread][std-thread-link] module from the [Rust Standard Library](https://doc.rust-lang.org/std/). Multiple threads are spawned, each independently executing the Hooke-Jeeves method on distinct random starting points. Each thread identifies the local minimum among its assigned trials. The trials are evenly distributed across the available threads, with the last thread handling any remaining trials when the total number of trials is not perfectly divisible by the number of threads. After all threads complete their assigned trials, the main thread gathers the partial results and determines the global minimum. This shared-memory approach leverages multi-core processors to accelerate the optimization process within a single compute node. The corresponding implementation is available [here][threads-link].

- **Multiprocessing** (MPI): The optimization trials are distributed across multiple MPI processes, with each process assigned a subset of trials according to its rank. Each process independently executes its assigned trials and identifies the best local minimum among them. The main process gathers the local minima from all processes using an MPI gather operation and determines the global minimum by comparing the collected results. MPI communication is implemented using the Rust bindings provided by the [mpi crate][mpi-crate-link]. The code for the multiprocessing scheme can be found [here][mpi-procs-link].

- **Hybrid** (MPI + Threads): This scheme combines multithreading with MPI-based multiprocessing to exploit both intra-process and inter-process parallelism. Each MPI process spawns multiple threads, with each thread independently executing the Hooke-Jeeves method on a subset of its assigned trials. After completing their trials, the threads return their local minima, and the main thread of each MPI process identifies the process-local minimum by comparing these results. The process-local minima are then communicated to the main MPI process, which gathers them using an MPI gather operation and determines the global minimum. This approach efficiently combines shared-memory parallelism within each process with distributed-memory parallelism across multiple processes. The implementation for this hybrid approach is available [here][hybrid-model-link].

> [!NOTE] 
> The parallel implementations are based on the [sequential version][seq-vers-link] of the algorithm, which initiates $n_{\text{trials}}$ random starting points in a search space of $n_{\text{vars}}$ dimensions, applies the Hooke-Jeeves method to each starting point and determines the global minimum by comparing the outcomes across all trials. The search space for each dimension is constrained to the interval $[-5, 5]$ and the default parameters from the [NetLib implementation][netlib-link] of the Hooke-Jeeves algorithm are utilized.

## Performance Evaluation
To evaluate and compare the performance of the implemented parallelism schemes for the Hooke-Jeeves algorithm, the [benchmarking script][benchmark-link] was used. For each worker configuration, five independent runs were performed, and the average execution time was reported. All experiments were conducted on an NVIDIA DGX A100 system with a fixed problem size of **32** variables (`nvars`) and **65,536** trials (`ntrials`).

As shown in Figure 1, both schemes demonstrate strong scaling behavior. The multithreading scheme maintains a parallel efficiency above **95%** across all worker configurations, while the MPI-based multiprocessing approach maintains an efficiency above **92%**.

<br>
<p align="center">
  <picture>
    <source srcset = "plots/figs/fig1_dark.png" media = "(prefers-color-scheme: dark)">
    <source srcset = "plots/figs/fig1_light.png" media = "(prefers-color-scheme: light)">
    <img src = "plots/figs/fig1_dark.png" width = "100%" alt = "Comparison of the multithreading and MPI-based multiprocessing schemes in terms of strong scaling speedup (left) and parallel efficiency (right)">
  </picture>
  <br>
  <em>Figure 1: Comparison of the multithreading and MPI-based multiprocessing schemes in terms of strong scaling speedup (left) and parallel efficiency (right)</em>
</p>
<br>

Regarding the hybrid scheme, as shown in Figure 2, it achieves very strong scaling through the combination of MPI processes and threads, significantly reducing the execution time of the Hooke-Jeeves algorithm.

<br>
<p align="center">
  <picture>
     <source srcset = "plots/figs/fig2_dark.png" media = "(prefers-color-scheme: dark)">
      <source srcset = "plots/figs/fig2_light.png" media = "(prefers-color-scheme: light)">
     <img src = "plots/figs/fig2_dark.png" width = "70%" alt = "Strong scaling speedup of the hybrid scheme using 1, 2, and 4 threads per MPI process">
  </picture>
  <br>
  <em>Figure 2: Strong scaling speedup of the hybrid scheme using 1, 2, and 4 threads per MPI process</em>
</p>


## Contact

For any questions or further information, please contact:

- Argiris Sofotasios — [a.sofotasios@ac.upatras.gr](mailto:a.sofotasios@ac.upatras.gr)
- Dimitris Metaxakis — [d.metaxakis@ac.upatras.gr](mailto:d.metaxakis@ac.upatras.gr)

<!-- MARKDOWN LINKS & IMAGES -->
[rosenbrock-link]: https://en.wikipedia.org/wiki/Rosenbrock_function
[hooke-jeeves-link]: https://media.neliti.com/media/publications/411591-review-of-hooke-and-jeeves-direct-search-b7dfccd7.pdf
[netlib-link]: https://netlib.org/opt/hooke.c
[seq-vers-link]: ./Hooke_Jeeves/sequential.rs
[std-thread-link]: https://doc.rust-lang.org/std/thread/
[threads-link]: ./Hooke_Jeeves/parallel/threads.rs
[mpi-crate-link]: https://docs.rs/mpi/0.8.0/mpi/
[mpi-procs-link]: ./Hooke_Jeeves/parallel/mpi.rs
[hybrid-model-link]: ./Hooke_Jeeves/parallel/hybrid.rs
[open-mpi-link]: https://www.open-mpi.org/
[mpich-link]: https://www.mpich.org/
[run-link]: ./run.sh
[benchmark-link]: ./benchmark.sh
[open-mpi-link]: https://www.open-mpi.org/
[mpich-link]: https://www.mpich.org/