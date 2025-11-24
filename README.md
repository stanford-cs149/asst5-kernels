# Assignment 5 - Write a Fast Kernel # 

**Due Thursday Dec 4, 11:59pm**

**100 points total**

## Overview ##

This assignment is an open ended assignment that is a bit different than your prior programming assignments in CS149. Really we want you to think about it as a very short final project that we've calibrated so that you can get a decent score on it from about 2 evenings of work. But we've also designed it so that teams that really want to get deep into it can spend a significant amount of time pursuing __very fast code implementations on a modern high-end GPU__.  There is no single task to optimize, and there is no performance bar that determines your score. Instead we are asking you to pick one (or more) of a palette of AI-related kernels we've provided, and then produce an optimizated implementation of the code for an H100 GPU. (All the kernels have baseline implementations in PyTorch.)

For fun, and to you help you track where your implementation stands compared to others in the class, we are running a class leaderboard for each of the kernels, and you'll be able to see how fast your implementation is compared to those of your classmates!

In summary, in this assignment:

* You get to pick what algorithm you wish to optimize (see details about the options below)
* You will run your code on a H100 GPU, although teams with credits left over from PA4 are welcome to optimize the algorithms on Trainium instead (although there is no leaderboard for Trainium).
* You can choose to write your code in straight CUDA, or use [Triton](https://triton-lang.org/main/index.html) or [Tilelang](https://tilelang.com/), two modern AI frameworks that provide tile-based abstractions for writing AI algorithms.
* __You are allowed to use any LLM accessible in [Stanford's AI Playground](https://uit.stanford.edu/service/aiplayground)) in any way to help with the assignment.__  For example, you can use the LLMs to write code, assess performance results from the performance profiler to decide what to do next, etc.
* There is no specific performance required for a certain grade... in fact we hope that some teams will do better much better than some of the world's best implementations (especially if an LLM is assisting!).  Instead, your grade will be determined by the staff's assessment of a work log that you will submit that describes the sequence of choices you made during your optimization process to get to your final (fastest) result. We describe what we expect in the work log later in the handout.

Overall the goal is to practice open-ended performance engineering, must like what you'd encounter in the ``real-world'' when you are trying to make a program run faster without a fast staff reference implementation to chase.

## The Available Kernels (Problems to Choose From) ##

The staff has selected the following kernels as options for the project. You can work on trying to optimize one or more of these kernels.  Please click on the links below to learn more about them.

You can also view the performance of the best solutions from the class on the class leaderboards. (Leaderboards will be released shortly.)

* [Histogram](problems/histogram)
* [1d-occupancy-decoder](problems/1d-occupancy-decoder)
* [FlashAttention](problems/flashattention)
* [3D Heat Equation – RK4 Benchmark](problems/rk4)
* [SwiGLU](problems/swiglu)

## Grading ##

This assignment will be graded on based on a combination of effort and success:
* __80 points.__ A fairly minimal effort put in, but work log shows evidence of applying concepts from the course to take a few steps to optimize performance.  Some speedup was achieved.
* __95 points.__ Solid effort, work log shows evidence of applying concepts from the course to interpret runtime/profiler results, argues for what the results suggest the next optimization should be. Work log shows solid effort put into exploring a reasonable set of optimization choices.  Some evidence of solid work might also be a decent final performance number.
* __95-110 points.__ Everything from the 95 score above, but students when above and beyond to achieve an impressive result. Getting to 100 might mean high performance on a scoreboard. Points above 100 will be awarded on a case-by-case basis. (Impress us, and you'll get rewarded!)

Although the rubric above bottoms out at 80, the staff reserves the right to give lower scores if we believe the minimal effort bar was not achieved.

## Setup and Job Queue Usage Guide (How to Run Code)

In this assignment you will not run your own GPU instances in the cloud, you will develop locally and submit code to a queue that will run your code on H100 machines managed by the course staff. When your job is done, you will receive results in the form of console logs, performance results, as well as an (optional) profile report that documents key performance statistics of the job. The profile report will help you make optimization decisions.  

The following guide walks you through installing popcorn-cli (the tool you will use to submit jobs to our job queue), registering your SUNet username with the system, developing and testing kernels, and submitting results to the various per-kernel leaderboards. Please follow the steps below in order: installation, one-time registration, local development (optional), and job submission -- to ensure your workflow runs smoothly on both the provided environment and your own GPU machine.

### Step 1: Install popcorn-cli

#### Option 1: Using pre-built binaries (Recommended)

1. Download the latest release for your platform in the `binary/` folder of this repo
2. Extract the archive
3. Move the binary to a location in your PATH

#### Option 2: Building from source

The instructions below are for a Myth machine, but similar sequence can be run on a local machine as well.

1. Install Rust:
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```
2. Build the CLI:
   ```bash
   cd kernel-cli && ./build.sh
   ```
---

### Step 2: Setup Environment Variables for Server Connection
To enable connection to our server for submission, you'll need to setup enviroment variables. Please check Ed post regarding Assignment 5 release for `setup.sh`. With the script, please run:
```bash
bash setup.sh
```

### Step 3: Register and Get Your Token (Just run once)

To submit to the class leaderboard, we need to know who you are.  All teams are suggested to choose an amusing (but work appropriate) team name that will identify your score on the leaderboard. Register with the system using your SUNet ID and desired leaderboard team name:

```bash
popcorn-cli register --sunet-id <your_sunet_id> --nickname <your_unique_nickname_in_leaderboard>
```

The token, SUNet ID, and nickname will be stored in `~/.popcorn.yaml`.  When you submit jobs, we'll use this token to identify you in the future.

**Important Notes:**
- Replace `<your_sunet_id>` with your actual SUNet ID and `<your_nickname_in_leaderboard>` with your preferred display name
- You cannot change your team name once registered. (So we recommend to think carefully about your brand.)  If you attempt to register again with the same SUNet ID, the server will return your original team name.

---

### Step 4: Develop Your Solution

Each problem has a folder under `/problems`.  With each problem folder there is a `templates/` folder and a `test_cases/` folder. `templates/template.py` provides an example submission written in Python, and `templates/template.cu` provides an example submission written in CUDA

#### Writing in Python, Triton, TileLang

If you wish to write your solution in Python create `submission.py` in the appropriate problem folder. Your implementation needs to follow the `custom_kernel` interface in `/templates/template.py`.

Our job queue submission environment supports [Triton](https://triton-lang.org/main/index.html) and [Tilelang](https://tilelang.com/). triton (3.5.1), tilelang (0.1.6.post2). So you are free to write your code in either of these modern domain-specific frameworks for writing AI kernels.

#### Writing in CUDA

If you wish to write a solution directly in CUDA, under each problem folder, create `submission.cu` following the pattern given by the code skeleton in `/templates/template.cu`. The PyTorch binding for the kernel is in `wrap_cuda_submission.py`. 

**Important**: Since the H100 job queue only accepts `submission.py` for your code, you need to run `python wrap_cuda_submission.py` to wrap your CUDA code in `submission.cu` (in order to compile and link to PyTorch). Do this before you submit through `popcorn-cli` or run locally using `eval.py`.

#### Local Development on a GPU machine

__You may skip this section if you do not wish to run code on a local machine.__

You can develop by editing code locally and using the job queue to run jobs, or you can develop and run your solutions locally on any CUDA-capable NVIDIA GPU machine (e.g., the AWS `g6.xlarge` (NVIDIA T4 GPU) from Assignment 3, if you still have credits). However, development on other machines should be for correctness and exploring options.  We want you to tune performance for the H100, and report on your analysis on the H100. As you've learned in this class, decisions you make for one type of processor might not be optimal decisions for another.  Be ware of this if you spend most of your time developing on a smaller GPU.

Below is an example of setting up local development environment on `g6.xlarge` with `Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.8 (Ubuntu 24.04)` AMI.

```
source /opt/pytorch/bin/activate # Activate the environment with PyTorch. 
export PYTHONPATH="<path-to-asst5>/problems:$PYTHONPATH"
```

Then, run the command in the `/problem/<fd/swiglu/histogram> folder:
```
python ../eval.py <test/benchmark/profile> test_cases/test.txt
```
The test.txt can be changed to any shape you want to test. Just follow the format. We will only benchmark and test on one set of shapes. 

The eval.py uses `ncu` tool and `ncu_report` Python package if you set the `profile` option.

`ncu` requires the OS permission to GPU device performance counter. To enable this, please follow the instruction in "Enable access for all users" section [here](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters). This only needs to be done once. Example commands are:
```
cd /etc/modprobe.d
sudo vim ncu-permission.conf
options nvidia NVreg_RestrictProfilingToAdminUsers=0 # Type this in vim
sudo reboot # Reboot for the change to take effect
```

After that, you need to find the cuda installation path where you can find the nsight-compute folder. This is for the `ncu_report` package used by `eval.py`. Example command is:
```
export PYTHONPATH="/usr/local/cuda-12.9/nsight-compute-2025.2.1/extras/python:$PYTHONPATH"
```

If you want to use inline CUDA, run this before executing `eval.py` if you encounter errors like `libstdc++.so.6: version `GLIBCXX_3.4.20' not found`:
```
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
```
### Step 5: Submit Your Solution

#### Command Format

```bash
popcorn-cli submit --leaderboard <leaderboard_name> --mode <mode> <path_to_submission.py>
```

#### Flags and Commandline Options

- `--leaderboard`: The name of the class leaderboard (there is one per problem) to submit to.
- `--mode`: Submission mode (choose one):
  - `test`: Verify correctness (no leaderboard submission)
  - `benchmark`: Evaluate runtime (no leaderboard submission)
  - `leaderboard`: Benchmark and submit your result to the class leaderboard
  - `profile`: Profile kernel and return results (Full NCU reports are available as downloadable artifacts from runner jobs link)
- `--output`: [Optional] The output path for returned printed results (if you want the results to be saved in a file instead of directly printed in the terminal)

After running this command, you will get the link to the job page in terminal. then they can go to the page to see full information, including .ncu-rep file if in profile mode

#### Viewing Profiling Results 

When profiling is enabled, we produce a summary of profiling statistics that includes useful metrics that hopefully useful to you in guiding optimization decisions.  For example, here's a profile summary that suggests a particular kernel is not yet making great use of the GPU (why is that?)

````
  NCU Report:
  
  Kernel 13: heat_step_kernel
  Duration            : 7.10 us
  Compute_Throughput  : 18.45%
  SM_Busy             : 31.12%
  L1_Cache_Throughput : 46.35%
  L2_Cache_Throughput : 23.55%
  DRAM_Throughput     : 4.45%
  L1_Cache_Hit_Rate   : 80.99%
  L2_Cache_Hit_Rate   : 85.49%
  DRAM_Read           : 1.01 MB
  DRAM_Write          : 0.00 B
  L2_to_L1_Traffic    : 3.87 MB
  L1_to_L2_Traffic    : 1.04 MB
  L2_to_SHMEM_Traffic : 0.00 B
````

If you use `--mode profile` in `popcorn-cli`, then the result on github action page will contain a `.ncu-rep` file. You can view the full profile in an interactive GUI by installing [NVIDIA Nsight Compute](https://developer.nvidia.com/tools-overview/nsight-compute/get-started) on your local machine and using it to open the `.ncu-rep` file.

### Step 6: Check the Leaderboard

If you submitted with `--mode leaderboard`, view your submission on the class leaderboard. See Ed for a link.

## What You Need to Produce - The Work Log ##

In this assignment your goal is to demonstrate mastery of course principles by taking a piece of code that is new to you, and optimize its performance beyond the starting implementation. You'll do it not by reaching a specific performance goal, but by doing the best you can and showing your thought process. Therefore, we expect a hand-in that contains a number of different versions of your code (steps along the optimization path), and a writeup of your reasoning as you went along your way.

A key principle of this course is data-driven decision making when optimizing code. In other words, performance optimization is an iterative process where you carry out the following steps.

1. You run the current state of the code
2. You measure its performance (and get statistics about compute and bandwidth utilization from a profiler).
3. You then using your knowledge of the code (where are the dependencies?, where are the data accesses?) and the measurement data, you form hypotheses about how you think you can make the code go faster (__increase parallelism, reduce memory traffic, hide memory latency, alleviate contention__).
4. Then you implement a change to the code to test the hypothesis to see if it was true, and if your code is now faster.

Note: in this assignment you are allowed to use an LLM to help you carry out steps (3) and (4)... so it might not be *you* directly doing these steps, you might be you prompting an LLM to help you with them.

Therefore, your work log hand-in should be formatted as a sequence of steps. 

#### Work Log Part 1: The Steps You Took

For each step, we want to know:
* How is the code structured in the current step?  (We'd like you to submit the code, but also describe it at a high level of abstraction in the hand-out. e.g., *"We were blocking the outermost loop and mapping blocks to CUDA thread blocks. And the innermost loop was parallelized over threads in the thread block"*).
* What is the performance of the code (give the runtime explicitly in the handin)?
* What other statistics did you measure and look at for the current code?
* What did you conclude from the measurements?
* What was your hypothesis about what was limited performance? (Or how it might be improved?) How did you come to this hypothesis?
* What does the hypothesis suggest you should try next in terms of how to modify your code's design?

Of course, we don't need to read about every single thing you tried or every small change you made.  We'd like you to use your judgment about what steps in your process were most important, and we just want to here about those.  For example, consider the level of abstraction you've talked about assignments with CAs in office hours for PA2, PA3, and PA4.  That's the level of abstraction that would be good to target in your handin. 

#### Work Log Part 2: Explain why you stopped

__Finally, we'd like the end of your work log to provide reasoning for why you stopped!__ Sure, you might stop because you felt you've spent enough time already, or ran out of time, but some teams might decide to stop because they conclude from the profile results that there isn't much more to do.  If you stopped because of an analysis of results, we'd like to know how you made that decision.  

#### Work Log Part 3: Did using LLMs help?

In this assignment, you are welcome to use LLMs to help you accelerate your development. If you used LLM assistance, please reflect on how you used LLMs: was it to write code? brainstorm optimization ideas? interpret profiling results? Please comment on whether you found the LLM assistant helpful.

__One note:__ It's entirely possible that an LLM might be able to directly give you a performant solution to one of the problems given only a simple prompt like "optimize this code for the H100". (We certainly haven't tried all the LLMs available on the [Stanford's AI Playground](https://uit.stanford.edu/service/aiplayground), and we certainly haven't tried engineering good instruction prompts.)  However in your work log you must explain your reasoning for how you got to your solution, how you decided there was not a lot of potential speedup in proceeding further, etc? If you encounter a situation where an LLM just jumps you to an amazing answer on your first prompt and you don't see any way to make it better, that's okay, please reach out to the staff and we can talk about how you should proceed. For example, you could attempt a second problem.

__But to be clear, a clever approach to vibe-coding a solution by iteratively working hand-in-hand with the LLM, or conducting sequence of prompt engineering steps to arrive at a good answer would also constitute a great way to do the assignment. Just describe your thought process and exactly what you did in your worklog. For example, you might want to include your LLM prompts for each step.__

### Specific handin instructions

Overall we except you to submit a handin as a single `.zip` containing `handin.pdf` and a sequence of code files for the key steps you describe in the handout. 



