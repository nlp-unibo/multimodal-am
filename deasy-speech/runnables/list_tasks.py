import os

from deasy_learning_generic.commands import list_registrations, setup_registry

if __name__ == '__main__':
    # ProjectRegistry setup
    project_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    setup_registry(directory=project_dir,
                   module_names=['components', 'configurations'])

    # List registered components and configurations
    list_registrations(namespaces=['arg_aaai', 'm-arg', 'us_elec', 'transformers', 'default'])
