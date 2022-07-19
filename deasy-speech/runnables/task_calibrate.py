import os

from deasy_learning_generic.commands import setup_registry, task_calibration

if __name__ == '__main__':
    # ProjectRegistry setup
    project_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    setup_registry(directory=project_dir, module_names=['components', 'configurations'])

    # Run training
    task_calibration(
        task_config_name="",
        calibrator_config_name="",
        save_results=False,
        framework_config_name=None,
        db_name='')
