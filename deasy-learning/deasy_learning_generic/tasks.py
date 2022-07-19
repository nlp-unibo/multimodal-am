
import os
import shutil
from datetime import datetime

from deasy_learning_generic.registry import ProjectRegistry, ComponentFlag
from deasy_learning_generic.composable import Composable
from deasy_learning_generic.utility.json_utils import save_json, load_json
from deasy_learning_generic.utility.log_utils import Logger


class Task(Composable):

    def __init__(self, task_name, reference_task_name=None, **kwargs):
        super(Task, self).__init__(**kwargs)
        self.task_name = task_name
        self.reference_task_name = reference_task_name

        self.model_path = None

    def save_task_info(self, filepath, task_config, task_config_registration_info):
        task_info = {
            'task_config_registration_info': str(task_config_registration_info),
            'model_path': self.model_path
        }

        for flag, flag_configurations in task_config.configurations.items():
            for config_key, config in flag_configurations.items():
                task_info.setdefault(flag, {}).setdefault(config_key, config.get_attributes())

        save_path = os.path.join(filepath, ProjectRegistry.JSON_TASK_CONFIG_REGISTRATION_INFO_NAME)
        save_json(save_path, task_info)

    def load_task_info(self, filepath):
        # Load model path
        assert os.path.isdir(filepath), "Given path is not a directory! Got: {}".format(filepath)
        return load_json(os.path.join(filepath, ProjectRegistry.JSON_TASK_CONFIG_REGISTRATION_INFO_NAME))

    def save_calibration_info(self, calibration_results, calibration_trial,
                              task_config_registration_info, calibrator_config_registration_info):
        base_path = ProjectRegistry['calibration_results_dir']
        if not os.path.isdir(base_path):
            os.makedirs(base_path)

        save_path = os.path.join(base_path, ProjectRegistry.JSON_CALIBRATION_RESULTS_NAME)
        if os.path.isfile(save_path):
            existing_results = load_json(save_path)
            existing_results.setdefault(str(task_config_registration_info), {}).setdefault(
                str(calibrator_config_registration_info), {
                    'parameters': calibration_results,
                    'trial_info': calibration_trial
                })
            save_json(save_path, existing_results)
        else:
            results = {}
            results.setdefault(str(task_config_registration_info), {}).setdefault(
                str(calibrator_config_registration_info), {
                    'parameters': calibration_results,
                    'trial_info': calibration_trial
                })
            save_json(save_path, results)

    def _get_task_save_path(self, base_save_path=None):
        base_save_path = base_save_path if base_save_path is not None else ProjectRegistry['task_dir']

        current_date = datetime.today().strftime('%d-%m-%Y-%H-%M-%S')
        save_base_path = os.path.join(base_save_path,
                                      self.task_name,
                                      current_date)
        return save_base_path

    def _retrieve_save_path_from_test_name(self, test_name):
        return os.path.join(ProjectRegistry['task_dir'],
                            self.reference_task_name,
                            test_name)

    def _rename_folder(self, filepath, test_name):
        if test_name is not None:
            base_path = os.path.join(ProjectRegistry['task_dir'], self.task_name)
            renamed_base_path = os.path.join(base_path, test_name)

            # Check if it already exists!
            if os.path.isdir(renamed_base_path):
                # Look for existing duplicate runs
                duplicate_folders = list(filter(lambda folder_name: 'run_' in folder_name, os.listdir(base_path)))
                if duplicate_folders:
                    duplicate_folders = \
                        sorted(duplicate_folders, key=lambda folder_name: int(folder_name.split('run_')[-1]),
                               reverse=True)[
                            -1]
                    next_duplicate_run_id = int(duplicate_folders.split('run_')[-1]) + 1
                    rename_folder = test_name + '_run_{}'.format(next_duplicate_run_id)
                else:
                    rename_folder = test_name + '_run_1'

                renamed_base_path = os.path.join(base_path, rename_folder)

            os.rename(filepath, renamed_base_path)

    def train(self, task_config_registration_info, task_config, test_name=None, save_results=True, return_results=False,
              base_save_path=None):
        task_config.configurations[ComponentFlag.MODEL]['default'].show()

        # Load Helper
        framework_helper = task_config.configurations[ComponentFlag.FRAMEWORK_HELPER]['default'].retrieve_component()
        framework_helper.setup()

        # Load DataLoader
        model_loader_params = task_config.configurations[ComponentFlag.MODEL]['default'].get_flag_parameters(
            flag=ComponentFlag.DATA_LOADER)
        data_loader = task_config.configurations[ComponentFlag.DATA_LOADER]['default'].retrieve_component(
            additional_args=model_loader_params)

        # Test save path
        save_base_path = self._get_task_save_path(base_save_path=base_save_path)

        if save_results and not os.path.isdir(save_base_path):
            os.makedirs(save_base_path)

        # Logging
        Logger.set_log_path(save_base_path)
        Logger.get_logger(__name__)

        # Callbacks
        callbacks = []
        if ComponentFlag.CALLBACK in task_config.configurations:
            for config_key, callback_config in task_config.configurations[ComponentFlag.CALLBACK].items():
                model_callback_params = task_config.configurations[ComponentFlag.MODEL]['default'].get_flag_parameters(
                    flag=ComponentFlag.CALLBACK, key=config_key)
                callback_instance = callback_config.retrieve_component(**model_callback_params)
                callback_instance.set_save_path(save_path=save_base_path)
                callbacks.append(callback_instance)

        # Load Routine
        self.model_path = self.determine_task_id(task_config=task_config)
        routine_additional_args = {
            'framework_helper': framework_helper,
            'data_loader': data_loader,
            'test_path': save_base_path,
            'model_config': task_config.configurations[ComponentFlag.MODEL]['default'],
            'evaluation_config': task_config.configurations[ComponentFlag.EVALUATION]['default'],
            'callbacks': callbacks,
            'model_path': self.model_path,
        }
        model_routine_params = task_config.configurations[ComponentFlag.MODEL]['default'].get_flag_parameters(
            flag=ComponentFlag.ROUTINE)
        routine_additional_args.update(model_routine_params)

        routine = task_config.configurations[ComponentFlag.ROUTINE]['default'].retrieve_component(
            additional_args=routine_additional_args)
        routine_statistics = routine.train(save_results=save_results)

        if save_results:
            self.save_task_info(filepath=save_base_path,
                                task_config=task_config,
                                task_config_registration_info=task_config_registration_info)
            self._rename_folder(filepath=save_base_path, test_name=test_name)

        if return_results:
            return routine_statistics

    def mongo_train(self, task_config_registration_info, task_config, project_dir, module_names,
                    test_name=None, save_results=True, return_results=False, base_save_path=None):
        ProjectRegistry.set_project_dir(directory=project_dir)

        if module_names is not None:
            for module_name in module_names:
                ProjectRegistry.load_custom_module(module_name=module_name)

        Logger.get_logger(__name__).info(
            f'Running worker -> {ProjectRegistry.PROJECT_DIR} {ProjectRegistry.get_loaded_modules()}')
        return self.train(task_config_registration_info=task_config_registration_info,
                          task_config=task_config,
                          test_name=test_name,
                          save_results=save_results,
                          return_results=return_results,
                          base_save_path=base_save_path)

    def inference(self, test_name, task_config, task_save_info, save_results=False, return_results=False):
        train_save_base_path = self._retrieve_save_path_from_test_name(test_name=test_name)

        # Test save path
        save_base_path = self._get_task_save_path()

        if save_results:
            os.makedirs(save_base_path)

        task_config.configurations[ComponentFlag.MODEL]['default'].show()

        # Loader PipelineHelper
        framework_helper = task_config.configurations[ComponentFlag.FRAMEWORK_HELPER]['default'].retrieve_component()
        framework_helper.setup()

        # Load DataLoader
        model_loader_params = task_config.configurations[ComponentFlag.MODEL]['default'].get_flag_parameters(
            flag=ComponentFlag.DATA_LOADER)
        data_loader = task_config.configurations[ComponentFlag.DATA_LOADER]['default'].retrieve_component(
            additional_args=model_loader_params)

        # Logging
        Logger.set_log_path(save_base_path)
        Logger.get_logger(__name__)

        # Callbacks
        callbacks = []
        if ComponentFlag.CALLBACK in task_config.configurations:
            for config_key, callback_config in task_config.configurations[ComponentFlag.CALLBACK].items():
                model_callback_params = task_config.configurations[ComponentFlag.MODEL]['default'].get_flag_parameters(
                    flag=ComponentFlag.CALLBACK,
                    key=config_key)
                callback_instance = callback_config.retrieve_component(additional_args=model_callback_params)
                callback_instance.set_save_path(save_path=save_base_path)
                callbacks.append(callback_instance)

        # Evaluation config
        model_evaluation_params = task_config.configurations[ComponentFlag.MODEL]['default'].get_flag_parameters(
            flag=ComponentFlag.EVALUATION)
        evaluation_config = task_config.configurations[ComponentFlag.EVALUATION]['default']
        evaluation_config = evaluation_config.get_delta_copy(**model_evaluation_params)

        # Load Routine
        self.model_path = self.determine_task_id(task_config=task_config)
        routine_additional_args = {
            'framework_helper': framework_helper,
            'data_loader': data_loader,
            'test_path': save_base_path,
            'train_test_path': train_save_base_path,
            'model_config': task_config.configurations[ComponentFlag.MODEL]['default'],
            'evaluation_config': evaluation_config,
            'callbacks': callbacks,
            'train_model_path': task_save_info['model_path'],
            'model_path': self.model_path
        }
        model_routine_params = task_config.configurations[ComponentFlag.MODEL]['default'].get_flag_parameters(
            flag=ComponentFlag.ROUTINE)
        routine_additional_args.update(model_routine_params)

        routine = task_config.configurations[ComponentFlag.ROUTINE]['default'].retrieve_component(
            additional_args=routine_additional_args)
        routine_statistics = routine.forward(save_results=save_results)

        if return_results:
            return routine_statistics

    def calibrate(self, task_config_registration_info, task_config,
                  calibrator_config_registration_info, db_name, save_results=False):
        # Get calibrator
        calibrator_config = ProjectRegistry.retrieve_configurations_from_info(
            registration_info=calibrator_config_registration_info)
        model_calibrator_params = task_config.configurations[ComponentFlag.MODEL]['default'].get_flag_parameters(
            flag=ComponentFlag.CALIBRATOR)
        calibrator = calibrator_config.retrieve_component(additional_args=model_calibrator_params)

        # Configure calibrator
        calibrator.set_validator(validator=self.mongo_train,
                                 validator_args={
                                     'project_dir': ProjectRegistry.PROJECT_DIR,
                                     'module_names': ProjectRegistry.get_loaded_modules(),
                                     'save_results': save_results,
                                     'return_results': True,
                                     'base_save_path': ProjectRegistry['calibration_dir'],
                                     'task_config_registration_info': task_config_registration_info,
                                     'task_config': task_config
                                 })
        calibrator.show_info()

        # Run calibrator
        search_space = calibrator_config.get_search_space()
        calibration_results, calibration_trial = calibrator.run(search_space=search_space, db_name=db_name)
        self.save_calibration_info(calibration_results=calibration_results,
                                   calibration_trial=calibration_trial,
                                   task_config_registration_info=task_config_registration_info,
                                   calibrator_config_registration_info=calibrator_config_registration_info)
        return calibration_results

    def _get_data_config_id(self, filepath, config):
        config_path = os.path.join(filepath, ProjectRegistry.JSON_MODEL_DATA_CONFIGS_NAME)

        if not os.path.isdir(filepath):
            os.makedirs(filepath)

        if os.path.isfile(config_path):
            data_config = load_json(config_path)
        else:
            data_config = {}

        if config in data_config:
            return int(data_config[config])
        else:
            max_config = list(map(lambda item: int(item), data_config.values()))

            if len(max_config) > 0:
                max_config = max(max_config)
            else:
                max_config = -1

            data_config[config] = max_config + 1

            save_json(config_path, data_config)

            return max_config + 1

    def _clear_data_config(self, filepath, config, config_id):
        config_path = os.path.join(filepath, ProjectRegistry.JSON_MODEL_DATA_CONFIGS_NAME)

        if os.path.isfile(config_path):
            data_config = load_json(config_path)
            del data_config[config]
            save_json(config_path, data_config)

            folder_to_remove = os.path.join(filepath, str(config_id))
            shutil.rmtree(folder_to_remove)

    # e.g.: a dictionary may end up with different orderings -> different serialization ID
    def determine_task_id(self, task_config):
        # Associates an ID to each combination for easy file naming while maintaining whole info
        task_args = [f'{param_key}={param_value}'
                     for param_key, param_value in task_config.get_serialization_parameters().items()]
        config_name = '_'.join(task_args)
        model_base_path = os.path.join(ProjectRegistry.TESTS_DATA_DIR)
        config_id = self._get_data_config_id(filepath=model_base_path, config=config_name)
        model_path = os.path.join(model_base_path, str(config_id))

        if not os.path.isdir(model_path):
            os.makedirs(model_path)

        return model_path
