import os

from deasy_learning_generic.commands import setup_registry, task_inference

if __name__ == '__main__':
    # ProjectRegistry setup
    project_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    setup_registry(directory=project_dir, module_names=['components', 'configurations'])

    # Run inference
    task_inference(test_name='',
                   task_config_registration_info="",
                   task_folder='',
                   save_results=False,
                   framework_config_name=None,
                   debug=False
                   )
