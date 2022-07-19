from setuptools import setup, find_packages

with open("README.md", 'r', encoding='utf-8') as fh:
    long_description = fh.read()

with open('requirements.txt', 'r') as f:
    requirements = f.readlines()
    requirements = [req for req in requirements if "--hash" not in req]
    requirements = [req.split("\\")[0].split(":")[0].strip() for req in requirements]



setup(
    name='deasy_learning_generic',
    version='0.1',
    author='Anonymous',
    author_email='',
    description='[Generic Package] A simple high-level framework for research',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='',
    project_urls={
        'Bug Tracker': ""
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"
    ],
    license='MIT',
    packages=find_packages(include=['implementations.*', 'nlp.*', 'utility.*']),
    install_requires=requirements,
    python_requires=">=3.6"
)

