# Test 

import sys
from pathlib import Path
from timeeval import TimeEval, MultiDatasetManager
from pathlib import Path
from timeeval import TimeEval, MultiDatasetManager, DefaultMetrics, Algorithm, TrainingType, InputDimensionality, ResourceConstraints
from timeeval.adapters import DockerAdapter
from timeeval.params import FixedParameters, FullParameterGrid
from timeeval.resource_constraints import GB
from timeeval.utils.window import ReverseWindowing
import random
import sys
from pathlib import Path
import urllib3
from durations import Duration

from timeeval import TimeEval, DatasetManager, RemoteConfiguration, ResourceConstraints
from timeeval.algorithms import *
from timeeval.metrics import RocAUC, RangeRocAUC, RangePrVUS
from timeeval_experiments.algorithm_configurator import AlgorithmConfigurator

# Assuming TimeEval is cloned in the current directory of the Jupyter notebook
path_to_timeeval = Path("./TimeEval")  # Adjust the path as necessary
sys.path.append(str(path_to_timeeval.resolve()))

dm = MultiDatasetManager([Path("timeeval-datasets")])
datasets = []
# have to map super and multivariate and univariate
datasets.extend(dm.select(collection="CalIt2"))
datasets.extend(dm.select(collection="Metro"))
datasets.extend(dm.select(collection="Dodgers")) #univariate
datasets.extend(dm.select(collection="GHL"))
datasets.extend(dm.select(collection="Genesis"))
datasets.extend(dm.select(collection="NASA-MSL"))
datasets.extend(dm.select(collection="Occupancy"))
datasets.extend(dm.select(collection="MGAB"))
# Experiment setup
repetitions = 1
rcs = ResourceConstraints(
    task_memory_limit=2 * GB,
    task_cpu_limit=1.0,
)

configurator = AlgorithmConfigurator(config_path="TimeEval/timeeval_experiments/param-config.example.json")

algorithms = [
    # arima(), 
    # autoencoder(),
    # bagel(),
    # cblof(),
    cof(),
    # copod(),
    # dae(),
    # damp(),
    # dbstream(),
    # deepant(),
    # deepnap(),
    # donut(),
    # dspot(),
    # dwt_mlead(),
    # eif(),
    # encdec_ad(),
    # ensemble_gi(),
    # fast_mcd(),
    # fft(),
    # generic_rf(),
    # generic_xgb(),
    # grammarviz3(),
    # grammarviz3_multi(),
    # hbos(),
    # health_esn(),
    # hif(),
    # hotsax(),
    # hybrid_knn(),
    # if_lof(),
    # iforest(),
    # img_embedding_cae(),
    kmeans(),
    knn(),
    # laser_dbn(),
    # left_stampi(),
    lof(),
    # lstm_a d(),
    # lstm_vae(),
    # median_method(),
    # mscred(),
    # mstamp(),
    # mtad_gat(),
    # multi_hmm(),
    # multi_norma(),
    # multi_subsequence_lof(),
    # mvalmod(),
    # norma(),
    # normalizing_flows(),
    # novelty_svr(),
    # numenta_htm(),
    # ocean_wnn(),
    # omnianomaly(),
    # pcc(),
    # pci(),
    # phasespace_svm(),
    # pst(),
    # random_black_forest(),
    # robust_pca(),
    # s_h_esd(),
    # sand(),
    # sarima(),
    # series2graph(),
    # sr(),
    # sr_cnn(),
    # ssa(),
    # stamp(),
    # stomp(),
    # subsequence_fast_mcd(),
    # subsequence_if(),
    # subsequence_knn(),
    # subsequence_lof(),
    # tanogan(),
    # tarzan(),
    # telemanom(),
    # torsk(),
    # triple_es(),
    # ts_bitmap(),
    # valmod(),
]
print(f"Selected algorithms: {len(algorithms)}")
configurator.configure(algorithms, ignore_dependent=False, perform_search=False)

print()
for algo in algorithms:
    print(f"Algorithm {algo.name} param_grid:")
    for config in algo.param_config.iter(algo, dataset=datasets[0]):
        print(f"  {config}")
sys.stdout.flush()

cluster_config = RemoteConfiguration(
    scheduler_host="localhost",
    worker_hosts=["localhost"]
)
limits = ResourceConstraints(
    tasks_per_host=1,
    task_cpu_limit=1.,
    train_timeout=Duration("1 minute"),
    execute_timeout=Duration("1 minute")
)
timeeval = TimeEval(dm, datasets, algorithms,
                    distributed=False,
                    remote_config=cluster_config,
                    resource_constraints=limits,
                    metrics=[RocAUC(), RangeRocAUC(buffer_size=100), RangePrVUS(max_buffer_size=100)]
                    )
timeeval.run()
print(timeeval.get_results(aggregated=False))