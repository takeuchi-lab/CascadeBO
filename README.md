# Bayesian Optimization for Cascade-type Multistage Processes

This is an implementation of Bayesian Optimization for Cascade-type Multistage Processes (Kusakawa et al., 2022).  
This page provides instructions, shell scripts, python codes, and binary files to reproduce the experimental results.


# Environments
- Hardware platform: x86_64
- OS: CentOS Linux release 7.7.1908 (Core) / Ubuntu 20.04.2 LTS
- Python version: 3.9.6
- Required python packages: listed in `requirements.txt`.
    - Run `pip install -r requirements.txt` to install these packages. 


# Usage
1. Download or clone this repository and make sure that `emulator/`, `scripts/`, and `src/` are located as follows:
    ```shell-session
    ./  # project directory  
    ├── emulator/
    │   ├── 64bit/
    │   └── ECdata/
    ├── scripts/
    └── src/
    ```

2. To reproduce the experimental results, run the following commands.
When each shell script finishes, a PDF file in which the experimental results are plotted will be created in `./figure` ( `./figure` directory will be created automatically).
If an argument of bash file is nothing, experiments are performed sequentially.
The experiments can be parallelized by ``bash ./scripts/figure3_sample_path_3.sh parallel_num'' in all the scripts.

   -  Figure 3: Synthetic functions
        ```bash
        bash ./scripts/figure3_sample_path_3.sh
        bash ./scripts/figure3_sample_path_5.sh
        bash ./scripts/figure3_rosenbrock_3.sh
        bash ./scripts/figure3_rosenbrock_5.sh
        bash ./scripts/figure3_sphere.sh
        bash ./scripts/figure3_matyas.sh
        ```

   -  Figure 4: Solar cell simulator
        ```bash
        bash ./scripts/figure4_solarcell.sh
        ```

   -  Figure 5: Hydrogen plasma treatment process
        ```bash
        bash ./scripts/figure5_hydrogen_plasma.sh
        ```

   - Figure 6: Suspension setting (sample path, solar cell simulator) 
        ```bash
        bash ./scripts/figure6_sample_path_3.sh
        bash ./scripts/figure6_sample_path_5.sh
        bash ./scripts/figure6_solarcell.sh
        ```

   - Figure 7: Comparison between EI-based and EI-FN
        ```bash
        bash ./scripts/figure7_sample_path_3.sh
        bash ./scripts/figure7_sample_path_5.sh
        ```

# Reference
S. Kusakawa, S. Takeno, Y. Inatsu, K. Kutsukake, S. Iwazaki, T. Nakano, T. Ujihara, M. Karasuyama, I. Takeuchi, Bayesian Optimization for Cascade-type Multi-stage Processes, Neural Computation, 2022.  (to appear)

Preprint Paper: [https://arxiv.org/abs/2111.08330](https://arxiv.org/abs/2111.08330)
