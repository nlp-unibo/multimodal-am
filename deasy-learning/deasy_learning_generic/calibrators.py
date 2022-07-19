import os
import pickle
import time
from multiprocessing import Process, Pool, Queue
from subprocess import Popen

import hyperopt
from hyperopt import fmin, space_eval, tpe, Trials
from hyperopt.mongoexp import MongoTrials
from hyperopt.progress import no_progress_callback

from deasy_learning_generic.composable import Composable
from deasy_learning_generic.configuration import ParamInfo
from deasy_learning_generic.data_loader import DataSplit
from deasy_learning_generic.registry import ComponentFlag, ProjectRegistry
from deasy_learning_generic.utility.log_utils import Logger
from deasy_learning_generic.utility.pickle_utils import save_pickle
from deasy_learning_generic.utility.python_utils import clear_folder

default_callback = no_progress_callback


class BaseCalibrator(Composable):
    """
    Base calibrator interface.
    """

    def __init__(self, validate_on, validate_condition, max_evaluations=-1, **kwargs):
        super(BaseCalibrator, self).__init__(**kwargs)
        self.validate_on = validate_on
        self.validate_condition = validate_condition
        self.max_evaluations = max_evaluations

        self.validator = None
        self.validator_args = None

        self.save_path = None

    def set_validator(self, validator, validator_args):
        self.validator = validator
        self.validator_args = validator_args

    def set_save_path(self, save_path):
        self.save_path = save_path

        # Save folder
        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)

    def run(self, search_space, db_name):
        """
        Calibrator main interface. The run() methods tells the calibrator class to start the parameter search routine.
        """
        assert self.validator is not None, f'Validator is not set! ' \
                                           f'Make sure to run set_validation_routine() before calibration'


class HyperOptCalibrator(BaseCalibrator):
    """
    Hyperopt compliant calibrator which leverages the hyperopt library for the parameter optimization task.
    """

    # TODO: make configurable?
    _MONGO_DIR_NAME = 'mongo_dir'
    _MONGO_WORKERS_DIR_NAME = 'mongo_workers_dir'
    _WORKER_SLEEP_INTERVAL = 2.00

    def __init__(self, hyperopt_additional_info=None, use_mongo=False,
                 mongo_address='localhost', mongo_port=1234, mongo_dir=None,
                 workers=2, workers_dir=None, poll_interval=5, reserve_timeout=5.0,
                 max_consecutive_failures=1, use_subprocesses=False, **kwargs):
        super(HyperOptCalibrator, self).__init__(**kwargs)
        assert self.max_evaluations

        self.hyperopt_additional_info = hyperopt_additional_info if hyperopt_additional_info is not None else {}
        self.use_mongo = use_mongo

        self.mongo_address = mongo_address
        self.mongo_port = mongo_port
        self.mongo_dir = mongo_dir if mongo_dir else os.path.join(ProjectRegistry.PROJECT_DIR,
                                                                  HyperOptCalibrator._MONGO_DIR_NAME)

        self.workers = workers
        self.workers_dir = workers_dir if workers_dir else os.path.join(ProjectRegistry.PROJECT_DIR,
                                                                        HyperOptCalibrator._MONGO_WORKERS_DIR_NAME)

        if self.use_mongo:
            if not os.path.isdir(self.mongo_dir):
                os.makedirs(self.mongo_dir)
            if not os.path.isdir(self.workers_dir):
                os.makedirs(self.workers_dir)

        self.poll_interval = poll_interval
        self.reserve_timeout = reserve_timeout
        self.max_consecutive_failures = max_consecutive_failures
        self.use_subprocesses = use_subprocesses

    def _minimize_objective(self, param_comb):
        """
        Objective function for the hyperopt calibrator.
        Specifically, for each selected parameter combination a K-fold routine is held, so as to
        obtain robust evaluation metrics test results.

        :param param_comb: a single parameters combination (python dictionary)
        :return: reference evaluation metric result (float)
        """

        # Transform param_comb into list of ParamInfo
        Logger.get_logger(__name__).info('Considering hyper-parameters: {}'.format(param_comb))
        params_info = [ParamInfo(name=name, value=value) for name, value in param_comb.items()]

        # Update model configuration
        self.validator_args['task_config'].configurations[ComponentFlag.MODEL]['default'].update_from_params_info(
            params_info=params_info)
        self.validator_args['task_config'].configurations[ComponentFlag.MODEL]['default'].show()

        # Run validator
        validator_results = self.validator(**self.validator_args)

        # Get validation results
        validation_on_value = validator_results.get_data(key_path=[DataSplit.VAL.value, self.validate_on])

        if self.validate_condition == 'maximization':
            return 1. - validation_on_value

        return validation_on_value

    def _retrieve_custom_trials(self, db_name):
        if not os.path.isdir(self.mongo_dir):
            os.makedirs(self.mongo_dir)

        trials_path = os.path.join(self.mongo_dir, f'{db_name}.pickle')
        if os.path.exists(trials_path):
            Logger.get_logger(__name__).info('Using existing Trials DB!')
            with open(trials_path, 'rb') as f:
                trials = pickle.load(f)
        else:
            Logger.get_logger(__name__).info(
                f"Can't find specified Trials DB ({db_name})...creating new one!")
            trials = Trials(exp_key=db_name)
        return trials

    def _calibrate(self, trials, search_space):
        best = fmin(self._minimize_objective,
                    search_space,
                    algo=tpe.suggest,
                    max_evals=self.max_evaluations,
                    trials=trials,
                    show_progressbar=False,
                    **self.hyperopt_additional_info)
        return best

    def _mongo_calibrate(self, trials, search_space, queue):
        best = self._calibrate(trials=trials, search_space=search_space)
        queue.put(best)

    def _run_mongo_worker(self, db_name):
        # Define command
        cmd = ['hyperopt-mongo-worker',
               f'--mongo={self.mongo_address}:{self.mongo_port}/{db_name}',
               f'--poll-interval={self.poll_interval}',
               f'--reserve-timeout={self.reserve_timeout}',
               f'--max-consecutive-failures={self.max_consecutive_failures}']
        if not self.use_subprocesses:
            cmd.append(f'--no-subprocesses')

        cmd = ' '.join(cmd)
        Logger.get_logger(__name__).info(f'Executing mongo worker...\n{cmd}')

        # Clear workers dir before execution
        clear_folder(self.workers_dir)

        # Run process
        process = Popen(cmd, cwd=self.workers_dir, shell=True)
        process.wait()

    def run(self, search_space, db_name):
        """
        Starts the grid-search task, which is formulated a minimization problem (evaluation metric minimization).
        All the grid-search intermediate results are stored in a .csv file, along with the Trials object.
        Eventually, the best configuration is saved in JSON format, according to the same syntax used for describing
        the parameter space.
        """
        super(HyperOptCalibrator, self).run(search_space=search_space, db_name=db_name)

        Logger.get_logger(__name__).info(
            'Starting hyper-parameters calibration search! Max evaluations: {}'.format(self.max_evaluations))

        if self.use_mongo:
            trials = MongoTrials(f'mongo://{self.mongo_address}:{self.mongo_port}/{db_name}/jobs', exp_key='exp1')
        else:
            trials = self._retrieve_custom_trials(db_name=db_name)

        if self.use_mongo:
            Logger.get_logger(__name__).info(
                'Running calibration with mongodb, make sure mongodb is active and running!')

            # Execute main calibration process as subprocess
            main_calibrator_queue = Queue()
            main_calibrator_process = Process(target=self._mongo_calibrate,
                                              args=(trials, search_space, main_calibrator_queue))
            main_calibrator_process.start()

            # Execute workers
            with Pool(processes=self.workers) as pool:

                # Sleep ensures that different folders will be created
                workers_results = []
                for _ in range(self.workers):
                    workers_results.append(pool.apply_async(self._run_mongo_worker, (db_name,)))
                    time.sleep(HyperOptCalibrator._WORKER_SLEEP_INTERVAL)

                workers_results = [worker.get() for worker in workers_results]

            # Wait for main calibration process
            main_calibrator_process.join()
            best = main_calibrator_queue.get()
        else:
            best = self._calibrate(trials=trials, search_space=search_space)

        best_params = space_eval(search_space, best)
        try:
            best_trial = trials.best_trial
        except hyperopt.exceptions.AllTrialsFailed:
            best_trial = {'state': 'N/A'}

        Logger.get_logger(__name__).info('Hyper-parameters calibration ended..')
        Logger.get_logger(__name__).info('Best combination: {}'.format(best_params))

        Logger.get_logger(__name__).info('Best combination info: {}'.format(best_trial))

        if not self.use_mongo:
            save_pickle(filepath=os.path.join(self.mongo_dir, f'{db_name}.pickle'), data=trials)

        return best_params, best_trial
