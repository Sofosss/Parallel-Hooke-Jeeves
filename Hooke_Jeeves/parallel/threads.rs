use rand::distributions::Uniform;
use rand::{Rng, SeedableRng};
use std::fs::File;
use std::io::{self, BufRead};
use std::process::ExitCode;
use std::sync::Arc;
use std::{fmt, thread};

const MAXVARS: usize = 250; // maximum number of variables
const RHO_BEGIN: f64 = 0.9; // stepsize geometric shrink
const EPSMIN: f64 = 1.0e-6; // ending value of stepsize
const IMAX: usize = 5000; // maximum number of iterations

// parameters that define the search space for the initial guess using the uniform distribution generator
const LOW: f64 = -5.0;
const HIGH: f64 = 5.0;

enum InitMech {
    Random(rand::rngs::StdRng),
    File(String),
}

struct HJOptimizer<F>
where
    F: Fn(&Vec<f64>) -> f64,
{
    nvars: usize,
    rho: f64,
    epsilon: f64,
    itermax: usize,
    obj_fn: F,
}

impl<F> fmt::Debug for HJOptimizer<F>
where
    F: Fn(&Vec<f64>) -> f64,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "HJOptimizer {{ nvars: {}, rho: {}, epsilon: {}, itermax: {} }}",
            self.nvars, self.rho, self.epsilon, self.itermax
        )
    }
}

impl<F> HJOptimizer<F>
where
    F: Fn(&Vec<f64>) -> f64,
{
    fn new(nvars: usize, kernel: F) -> Self {
        HJOptimizer {
            rho: RHO_BEGIN,
            epsilon: EPSMIN,
            itermax: IMAX,
            obj_fn: kernel,
            nvars,
        }
    }

    fn init_guess(&self, init_mech: &mut InitMech, trial: usize) -> io::Result<Vec<f64>> {
        match init_mech {
            InitMech::Random(generator) => Ok(generator.sample_iter(Uniform::new_inclusive(LOW, HIGH)).take(self.nvars).collect()),
            InitMech::File(file) => {
                let file = File::open(format!("{}{}", file, trial))?;
                let reader = io::BufReader::new(file);
                reader
                    .lines()
                    .take(self.nvars)
                    .map(|line| line.and_then(|l| l.trim().parse::<f64>().map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))))
                    .collect()
            }
        }
    }

    fn optimize(&self, start_pt: &Vec<f64>) -> (f64, Vec<f64>, usize, usize) {
        let mut xbefore: Vec<f64> = Vec::with_capacity(start_pt.len());
        let mut xnew: Vec<f64>;
        let mut delta: Vec<f64> = start_pt
            .iter()
            .map(|el| {
                xbefore.push(*el);
                if (el * self.rho).abs() == 0.0 {
                    self.rho
                } else {
                    (self.rho * el).abs()
                }
            })
            .collect();

        let mut steplength = self.rho;
        let mut iters: usize = 0;
        let mut fbefore = (self.obj_fn)(&xbefore);
        let mut funevals: usize = 1;
        let mut fnew: f64;

        while (iters < self.itermax) && (steplength > self.epsilon) {
            iters += 1;
            xnew = xbefore.clone();
            fnew = self.exploratory_move(&mut delta, &mut xnew, fbefore, &mut funevals);

            let mut discard: bool = true;

            while (fnew < fbefore) && discard {
                for (i, xbef) in xbefore.iter_mut().enumerate() {
                    delta[i] = if xnew[i] <= *xbef { -delta[i].abs() } else { delta[i].abs() };
                    let tmp = *xbef;
                    *xbef = xnew[i];
                    // pattern move
                    xnew[i] = xnew[i] + xnew[i] - tmp;
                }

                fbefore = fnew;
                fnew = self.exploratory_move(&mut delta, &mut xnew, fbefore, &mut funevals);

                if fnew >= fbefore {
                    break;
                }

                for (i, xbef) in xbefore.iter().enumerate() {
                    if (xnew[i] - *xbef).abs() > (0.5 * delta[i].abs()) {
                        discard = true;
                        break;
                    }
                    discard = false;
                }
            }
            if steplength >= self.epsilon && fnew >= fbefore {
                steplength *= self.rho;
                delta = delta.iter().map(|el| el * self.rho).collect();
            }
        }
        let final_min = (self.obj_fn)(&xbefore);
        funevals += 1;
        (final_min, xbefore.clone(), funevals, iters)
    }

    fn exploratory_move(&self, delta: &mut Vec<f64>, point: &mut Vec<f64>, fbefore: f64, funevals: &mut usize) -> f64 {
        let mut kernel_point_plus: f64;
        let mut kernel_point_minus: f64;
        let mut fmin = fbefore;
        let mut delta_cur: f64;
        let mut point_comp: f64;
        let mut point_plus: f64;
        let mut point_minus: f64;

        for i in 0..point.len() {
            delta_cur = delta[i];
            point_comp = point[i];
            point_plus = point_comp + delta_cur;
            let original = std::mem::replace(&mut point[i], point_plus);

            kernel_point_plus = (self.obj_fn)(point);
            *funevals += 1;

            if kernel_point_plus < fmin {
                fmin = kernel_point_plus;
            } else {
                delta[i] = -delta_cur;
                point_minus = point_comp - delta_cur;
                let _ = std::mem::replace(&mut point[i], point_minus);
                kernel_point_minus = (self.obj_fn)(point);
                *funevals += 1;
                if kernel_point_minus < fmin {
                    fmin = kernel_point_minus;
                } else {
                    point[i] = original;
                }
            }
        }

        fmin

        // let mut z = point.clone();
        // let mut fmin = fbefore;
        // let mut ftmp: f64;

        // for i in 0..point.len() {
        //     z[i] = point[i] + delta[i];
        //     ftmp = (self.obj_fn)(&z);
        //     *funevals += 1;
        //     if ftmp < fmin {
        //         fmin = ftmp;
        //     } else {
        //         delta[i] = 0.0 - delta[i];
        //         z[i] = point[i] + delta[i];
        //         ftmp = (self.obj_fn)(&z);
        //         *funevals += 1;
        //         if ftmp < fmin {
        //             fmin = ftmp;
        //         } else {
        //             z[i] = point[i];
        //         }
        //     }
        // }

        // for i in 0..point.len() {
        //     point[i] = z[i];
        // }

        // fmin
    }
}

fn work(
    optimizer: Arc<HJOptimizer<impl Fn(&Vec<f64>) -> f64>>, init_mech: &mut InitMech, rank: usize, chunk: usize,
) -> (f64, Vec<f64>, Option<usize>, Option<usize>, usize) {
    let mut best_thr_fx = f64::MAX;
    let mut best_thr_pt: Vec<f64> = Vec::new();
    let mut best_thr_trial: Option<usize> = None;
    let mut best_thr_iter: Option<usize> = None;
    let mut total_thr_funevals = 0;

    for trial in 0..chunk {
        let start_pt = optimizer.init_guess(init_mech, rank * chunk + trial).unwrap_or_else(|e| {
            if let io::ErrorKind::InvalidData = e.kind() {
                eprintln!("[Thread {}] Invalid data in file. Error: {}.\nExiting...", rank, e);
                std::process::exit(1);
            } else {
                eprintln!("[Thread {}] Unknown Error: {}.\nExiting...", rank, e);
                std::process::exit(1);
            }
        });

        let (local_min, end_pt, funevals, iterations) = optimizer.optimize(&start_pt);
        if local_min < best_thr_fx {
            best_thr_fx = local_min;
            best_thr_pt = end_pt;
            best_thr_trial = Some(rank * chunk + trial + 1);
            best_thr_iter = Some(iterations);
        }

        total_thr_funevals += funevals;
    }
    (best_thr_fx, best_thr_pt, best_thr_trial, best_thr_iter, total_thr_funevals)
}

fn main() -> ExitCode {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .map(|arg| arg.parse().expect("Usage: {} [nvars] [ntrials] [nthreads]. Expecting unsigned integers!"))
        .collect();

    if args.len() != 3 {
        println!("Usage: [nvars] [ntrials] [nthreads]");
        std::process::exit(1);
    }
    if args[0] > MAXVARS {
        println!("Number of variables exceeds the maximum allowed value of {}", MAXVARS);
        std::process::exit(1);
    }

    println!(">> Multithreaded Hooke-Jeeves Global Search Optimization <<");

    let ntrials = args[1];
    let mut num_threads = args[2];

    if num_threads > num_cpus::get() {
        println!(
            "WARNING: The requested number of threads ({0}) exceeds the available logical cores ({1}). The number of threads has been adjusted to {1}.",
            args[2],
            num_cpus::get()
        );

        num_threads = num_cpus::get();
    }

    // kernel function
    let mut total_funevals = 0;
    let rosenbrock = |sample: &Vec<f64>| -> f64 {
        sample.windows(2).fold(0.0, |acc, win| {
            let x_curr = win[0];
            let x_next = win[1];
            acc + 100.0 * (x_next - x_curr.powi(2)).powi(2) + (x_curr - 1.0).powi(2)
        })
    };

    let optimizer = Arc::new(HJOptimizer::new(args[0], rosenbrock));

    println!("INFO: Optimizer -> {:#?}", optimizer);

    let mut best_fx = f64::MAX;
    let mut best_pt: Vec<f64> = Vec::new();
    let mut best_trial: Option<usize> = None;
    let mut best_iter: Option<usize> = None;

    let t0 = std::time::Instant::now();
    /* The ability to read starting points from files has been implemented for testing purposes.
    For each trial, the starting points are stored in a separate file within the same directory,
    following the naming pattern ".../..._$", where "$" corresponds to the trial number. */
    // let file_path = Some("path/to/file_");
    let file_path: Option<&str> = None;
    let chunk = ntrials / num_threads;

    thread::scope(|s| {
        let threads_ctx: Vec<_> = (0..num_threads)
            .map(|rank| {
                let chunk = if rank != (num_threads - 1) { chunk } else { ntrials - rank * chunk };
                let optimizer = Arc::clone(&optimizer);
                let gen_seed = rank as u64;

                let thread = s.spawn(move || {
                    let mut init_mech = if let Some(path) = file_path {
                        if rank == 0 {
                            println!("INFO: Initializing the starting point from the provided file(s)");
                        }
                        InitMech::File(path.to_string())
                    } else {
                        println!(
                            "INFO [Thread {}]: Initializing the starting point using a uniform distribution generator with seed: {}",
                            rank, rank
                        );
                        InitMech::Random(rand::rngs::StdRng::seed_from_u64(gen_seed))
                    };

                    let thread_res = work(optimizer, &mut init_mech, rank, chunk);
                    thread_res
                });

                thread
            })
            .collect();

        for thread in threads_ctx {
            let (local_min, end_pt, trial, iterations, local_funevals) = thread.join().unwrap();
            if local_min < best_fx {
                best_fx = local_min;
                best_pt = end_pt;
                best_trial = trial;
                best_iter = iterations;
            }
            total_funevals += local_funevals;
        }
    });

    let elapsed = t0.elapsed();

    println!("\nFINAL RESULTS:");
    println!("> Elapsed time = {:.3} secs", elapsed.as_secs_f64());
    println!("> Total number of threads = {}", num_threads);
    println!("> Total number of trials = {}", ntrials);
    println!("> Total number of function evaluations = {}", total_funevals);
    println!(
        "> Best result at trial {} used {} iterations and returned",
        best_trial.unwrap_or(0),
        best_iter.unwrap_or(0)
    );
    for (i, pt) in best_pt.iter().enumerate() {
        println!("x[{i:3}] = {:15.7e}", pt);
    }
    println!("f(x) = {:15.7e}", best_fx);
    return ExitCode::SUCCESS;
}