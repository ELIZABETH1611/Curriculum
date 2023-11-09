from experiment_launcher import Launcher, is_local

LOCAL = is_local()
TEST  = False # False Cluster
USE_CUDA = False

N_SEEDS = 1

if LOCAL:
    N_EXPS_IN_PARALLEL = 1
else:
    N_EXPS_IN_PARALLEL = 1

N_CORES = 1
MEMORY_SINGLE_JOB = 10000 #10M
MEMORY_PER_CORE = N_EXPS_IN_PARALLEL * MEMORY_SINGLE_JOB // N_CORES
PARTITION = 'amd2,amd3' # 'rtx'
GRES = 'gpu:1' if USE_CUDA else None  # gpu:rtx2080:1, gpu:rtx3080:1
CONDA_ENV = 'air_challenge'  # None

launcher = Launcher(
    exp_name='static_puck_no_bias',
    exp_file='planar_atacom',
    n_seeds=N_SEEDS,
    n_exps_in_parallel=1,
    n_cores=N_CORES,
    memory_per_core=MEMORY_PER_CORE,
    days=3,
    hours=0,
    minutes=0,
    seconds=0,
    partition=PARTITION,
    gres=GRES,
    conda_env=CONDA_ENV,
    use_timestamp=True,
    compact_dirs=False
)


number_experiment = 1
for i in range(number_experiment):
    launcher.add_experiment(
        # A subdirectory will be created for parameters with a trailing double underscore.
        experiment_id__=i
        #move_puck__=True
        )


launcher.run(LOCAL, TEST)