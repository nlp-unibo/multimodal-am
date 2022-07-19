import os

from tqdm import tqdm

from deasy_learning_generic.registry import ProjectRegistry, ComponentFlag, RegistrationInfo
from deasy_learning_generic.utility.json_utils import save_json, load_json
from deasy_learning_generic.utility.log_utils import Logger
from deasy_learning_generic.utility.search_utils import InequalityFilter
from deasy_learning_generic.configuration import TaskConfiguration, ParamInfo
import numpy as np
import matplotlib.pyplot as plt


# Utility

def _stringify_elements(elements):
    elements = list(map(lambda item: str(item), elements))
    elements = sorted(elements)
    return elements


def _retrieve_and_save(save_folder, save_name, **kwargs):
    elements = ProjectRegistry.get_registered_elements(**kwargs)
    elements = [item[0] for item in elements]
    elements = _stringify_elements(elements)
    save_json(os.path.join(save_folder, f'{save_name}.json'), elements)


# Commands

def setup_registry(directory=None, module_names=None):
    ProjectRegistry.set_project_dir(directory=directory)

    if module_names is not None:
        for module_name in module_names:
            ProjectRegistry.load_custom_module(module_name=module_name)


def list_registrations(namespaces=None, registration_path=None):
    if registration_path is None:
        registration_path = ProjectRegistry['registration_dir']

    if not os.path.isdir(registration_path):
        os.makedirs(registration_path)

    Logger.get_logger(__name__).info(f'Saving registration info to folder: {registration_path}')

    if namespaces is None:
        Logger.get_logger(__name__).info('No namespace set specified. Retrieving all available namespaces...')
        registered_elements = ProjectRegistry.get_registered_elements()
        namespaces = set([element.namespace for element in registered_elements])

    Logger.get_logger(__name__).info(f'Total namespaces: {len(namespaces)}')
    for namespace in tqdm(namespaces):
        namespace_registration_path = os.path.join(registration_path, namespace)

        if not os.path.isdir(namespace_registration_path):
            os.makedirs(namespace_registration_path)

        # Registered components
        _retrieve_and_save(save_folder=namespace_registration_path, save_name='components',
                           namespace_or_namespace_filters=namespace,
                           internal_key_or_internal_key_filters=ProjectRegistry.COMPONENT_KEY)

        # Registered configurations (we ignore tasks, since we retrieve them separately)
        _retrieve_and_save(save_folder=namespace_registration_path, save_name='configurations',
                           flag_or_flag_filters=[
                               InequalityFilter(attribute_name='flag', reference_value=ComponentFlag.TASK)],
                           namespace_or_namespace_filters=namespace,
                           internal_key_or_internal_key_filters=ProjectRegistry.CONFIGURATION_KEY)

        # Registered task configurations
        _retrieve_and_save(save_folder=namespace_registration_path, save_name='tasks',
                           namespace_or_namespace_filters=namespace,
                           internal_key_or_internal_key_filters=ProjectRegistry.CONFIGURATION_KEY,
                           flag_or_flag_filters=ComponentFlag.TASK)


def task_train(task_config_name, test_name, save_results=True, return_results=False,
               base_save_path=None, framework_config_name=None, calibrator_task_config_name=None,
               calibrator_config_name=None, debug=False):
    task_config_registration_info = RegistrationInfo.from_string_format(string_format=task_config_name)

    Logger.get_logger(__name__).info(
        f'Training with task configuration registration info:\n{task_config_registration_info}')
    task_config = ProjectRegistry.retrieve_configurations_from_info(
        registration_info=task_config_registration_info)

    if framework_config_name is not None:
        framework_registration_info = RegistrationInfo.from_string_format(string_format=framework_config_name)
        Logger.get_logger(__name__).info(
            f'Using framework configuration registration info: {framework_registration_info}')

        task_config.add_configuration(registration_info=framework_registration_info, force_update=True)

    # Update model parameters based on calibration results
    if calibrator_config_name is not None:
        calibration_results = load_json(os.path.join(ProjectRegistry['calibration_results_dir'],
                                                     ProjectRegistry.JSON_CALIBRATION_RESULTS_NAME))
        calibrated_model_parameters = calibration_results[calibrator_task_config_name][calibrator_config_name]['parameters']
        task_config.configurations[ComponentFlag.MODEL][TaskConfiguration.DEFAULT_KEY].update_from_params_info(
            [ParamInfo(name=key, value=value) for key, value in calibrated_model_parameters.items()]
        )

    if debug:
        task_config.get_debug_version()

    task = task_config.retrieve_component()
    task_results = task.train(
        task_config_registration_info=task_config_registration_info,
        task_config=task_config,
        test_name=test_name,
        save_results=save_results,
        return_results=return_results,
        base_save_path=base_save_path)
    return task_results


def multiple_task_train(task_config_names, test_names, save_results=True, return_results=False,
                        base_save_path=None, framework_config_name=None, calibrator_config_names=None,
                        debug=False):
    assert len(task_config_names) == len(test_names)

    if calibrator_config_names is not None:
        assert len(calibrator_config_names) == len(task_config_names)
    else:
        calibrator_config_names = [None] * len(task_config_names)

    results = []

    for task_name, test_name, calibrator_config_name in zip(task_config_names, test_names, calibrator_config_names):
        task_results = task_train(task_config_name=task_name, test_name=test_name, return_results=return_results,
                                  base_save_path=base_save_path, framework_config_name=framework_config_name,
                                  save_results=save_results, debug=debug, calibrator_config_name=calibrator_config_name)
        results.append(task_results)

    return results


def task_inference(test_name, task_folder, task_config_registration_info=None, save_results=False,
                   return_results=False, framework_config_name=None, debug=False):
    task_save_info = load_json(os.path.join(ProjectRegistry['task_dir'],
                                            task_folder,
                                            test_name,
                                            ProjectRegistry.JSON_TASK_CONFIG_REGISTRATION_INFO_NAME))

    if task_config_registration_info is None:
        task_config_registration_info = task_save_info['task_config_registration_info']
    else:
        task_config_registration_info = RegistrationInfo.from_string_format(string_format=task_config_registration_info)

    Logger.get_logger(__name__).info(
        f'Inference with task configuration registration info:\n{task_config_registration_info}')
    task_config = ProjectRegistry.retrieve_configurations_from_info(registration_info=task_config_registration_info)

    if framework_config_name is not None:
        framework_registration_info = RegistrationInfo.from_string_format(string_format=framework_config_name)
        Logger.get_logger(__name__).info(
            f'Using framework configuration registration info: {framework_registration_info}')

        task_config.add_configuration(registration_info=framework_registration_info, force_update=True)

    if debug:
        task_config.get_debug_version()

    task = task_config.retrieve_component()
    task_results = task.inference(test_name=test_name,
                                  task_config=task_config,
                                  task_save_info=task_save_info,
                                  save_results=save_results,
                                  return_results=return_results)


    return task_results


def multiple_task_forward(test_names, task_names=None, task_folders=None, save_results=False, return_results=False):
    assert task_names is not None or task_folders is not None
    if task_names is None:
        task_names = [None] * len(task_folders)
    if task_folders is None:
        task_folders = [None] * len(task_names)

    results = []

    for test_name, task_name, task_folder in zip(test_names, task_names, task_folders):
        task_results = task_inference(test_name=test_name, task_config_registration_info=task_name,
                                      task_folder=task_folder,
                                      save_results=save_results, return_results=return_results)
        results.append(task_results)

    return results


def task_calibration(calibrator_config_name, task_config_name, db_name,
                     save_results=False, framework_config_name=None, model_calibrator_config_name=None,
                     model_task_config_name=None):
    task_config_registration_info = RegistrationInfo.from_string_format(string_format=task_config_name)

    Logger.get_logger(__name__).info(
        f'Training with task configuration registration info:\n{task_config_registration_info}')
    task_config = ProjectRegistry.retrieve_configurations_from_info(
        registration_info=task_config_registration_info)

    if framework_config_name is not None:
        framework_registration_info = RegistrationInfo.from_string_format(string_format=framework_config_name)
        Logger.get_logger(__name__).info(
            f'Using framework configuration registration info: {framework_registration_info}')

        task_config.add_configuration(registration_info=framework_registration_info, force_update=True)

    # Update model parameters based on calibration results
    if model_calibrator_config_name is not None:
        calibration_results = load_json(os.path.join(ProjectRegistry['calibration_results_dir'],
                                                     ProjectRegistry.JSON_CALIBRATION_RESULTS_NAME))
        calibrated_model_parameters = calibration_results[model_task_config_name][model_calibrator_config_name]['parameters']
        task_config.configurations[ComponentFlag.MODEL][TaskConfiguration.DEFAULT_KEY].update_from_params_info(
            [ParamInfo(name=key, value=value) for key, value in calibrated_model_parameters.items()]
        )

    calibrator_config_registration_info = RegistrationInfo.from_string_format(string_format=calibrator_config_name)

    task = task_config.retrieve_component()
    calibration_results = task.calibrate(task_config_registration_info=task_config_registration_info,
                                         task_config=task_config,
                                         calibrator_config_registration_info=calibrator_config_registration_info,
                                         save_results=save_results,
                                         db_name=db_name)
    return calibration_results


def visualize_training_curves(test_name, task_folder):
    read_path = os.path.join(ProjectRegistry['task_dir'], task_folder, test_name)
    info_files = [name for name in os.listdir(read_path) if name.endswith("info.npy")]

    Logger.get_logger(__name__).info(f'Found {len(info_files)} training curve files...')

    task_save_info = load_json(os.path.join(ProjectRegistry['task_dir'],
                                            task_folder,
                                            test_name,
                                            ProjectRegistry.JSON_TASK_CONFIG_REGISTRATION_INFO_NAME))
    task_config_registration_info = task_save_info['task_config_registration_info']
    task_config_registration_info = RegistrationInfo.from_string_format(string_format=task_config_registration_info)
    task_config = ProjectRegistry.retrieve_configurations_from_info(registration_info=task_config_registration_info)

    # Data loader (get metrics)
    loader_config = task_config.configurations[ComponentFlag.DATA_LOADER]['default']
    metrics = loader_config.metrics
    metric_names = [metric.name for metric in metrics]

    def plot_block(axs, info_keys, info, best_values, title):
        cmap = plt.cm.get_cmap(name='Set1', lut=len(info_keys))
        info_colours = {key: cmap(idx) for idx, key in enumerate(info_keys)}

        legend_names = []
        for key in info_keys:
            axs.axvline(x=best_values[key], ymin=0, ymax=2, linestyle='--', c=info_colours[key])
            legend_names.append(f'{key}_best')

            axs.plot(info[key], c=info_colours[key])
            legend_names.append(key)

        axs.legend(legend_names)
        axs.set_title(title)

    for info_file in info_files:
        info_path = os.path.join(read_path, info_file)
        curr_info = np.load(info_path, allow_pickle=True).item()
        fig, axs = plt.subplots(2, 1)   # losses and metrics

        info_metric_names = [key for key in curr_info if any([key.endswith(name) for name in metric_names])]
        info_loss_names = [key for key in curr_info if key not in info_metric_names]

        best_values = {key: np.argmax(value) if key in metric_names else np.argmin(value)
                       for key, value in curr_info.items()}

        # Losses
        plot_block(axs=axs[0], info_keys=info_loss_names, info=curr_info, best_values=best_values, title='Losses')

        # Metrics
        plot_block(axs=axs[1], info_keys=info_metric_names, info=curr_info, best_values=best_values, title='Metrics')

    plt.show()
