import os

from deasy_learning_generic.commands import setup_registry, task_train

if __name__ == '__main__':
    # ProjectRegistry setup
    project_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    setup_registry(directory=project_dir, module_names=['components', 'configurations'])

    # Run training
    task_train(
        task_config_name="",
        calibrator_task_config_name=None,
        calibrator_config_name=None,
        test_name='',
        save_results=True,
        return_results=False,
        framework_config_name=None,
        base_save_path=None,
        debug=False
    )
