# Multimodal Argument Mining: A Case Study in Political Debates

This is the official repository of the paper "**Multimodal Argument Mining: A Case Study in Political Debates**".

## Project Structure

The project is organized as follows:

* **deasy-learning**: a simple python library for performing deep learning experiments (closed beta version).
* **deasy-speech**: the code regarding the experiments described in the paper.

We provide few important details for reproducing our experiments.

# Downloading the corpora

The corpora can be retrieved from the following links:

* UKDebates (in this repo is named as `arg_aaai2016`): http://argumentationmining.disi.unibo.it/aaai2016.html
* M-Arg: https://github.com/rafamestre/m-arg_multimodal-argumentation-dataset
* MM-USElecDeb60to16: pending approval for public release [TBA]

## Displaying Configurations

Deasy-learning works by saving a task configuration via a unique name (string). 
A task is a complete experimental test, i.e. training a model on a particular corpus.
Run `deasy-speech/runnables/list_tasks.py` to compute all the available tasks.

A `deasy-speech/registrations/` folder is created at the end of the process containing JSON files with all the existing configurations.

We report down below useful tags that are used to define tasks:

```
use_audio_features=True         # Use feature-based audio representation
use_audio_features=False        # Use embedding-based audio representation
ablation_study=None             # Perform TA input configuration without any modality removal
ablation_study=text             # Ablation study w/o text
ablation_study=audio            # Ablation study w/o audio
```

## Training a model

The `deasy-speech/runnables/task_train.py` offers the following API:

```
    # Run training
    task_train(
        task_config_name="",                # name of the task to run
        test_name='',                       # folder name to use where to save test results
        save_results=True,                  # whether to save test results in a folder or not
        framework_config_name=None,         # a framework configuration name (optional)
        debug=False                         # whether to run the task in debug mode (e.g., batch_size=1, epochs=1, etc..) or not
    )
```

As an example, if you want to train the BiLSTM model calibrated for the TO input configuration on M-Arg, the script is defined as follows:

```
    # Run training
    task_train(
        task_config_name="internal_key:instance--flag:task--framework:tf--tags:['annotation_confidence=0.85', 'calibrated', 'lstm', 'text_only']--namespace:m-arg",
        test_name='bilstm_to',
        save_results=True,
        framework_config_name=None,
        debug=False
    )
```

Run the script to train the model.

## Calibrated configurations

You can inspect all calibrated model configurations in `deasy-speech/configurations/models.py`.

## Ablation Study

Run `deasy-speech/runnables/task_forward.py` to loaded a trained model and perform an inference step.
The script offers the following API:

```
    # Run inference
   task_inference(test_name='',                         # folder name containing a trained model
                   task_config_registration_info="",    # a task configuration to use for inference (optional). If omitted, the same configuration used for training is selected.
                   task_folder='',                      # folder name of the task (parent folder of test_name)
                   save_results=False,                  # whether to save inference results or not
                   framework_config_name=None,          # a framework configuration name (optional)
                   debug=False                          # whether to run the task in debug mode or not
                   )
```

An ablation study can be performed by selecting a specific `task_config_registration_info` value. In particular, look for task registration names with the `ablation_study` tag.
We report down below an example showing how to remove the text modality from the calibrated BiLSTM model on the ACD task concerning MM-USElecDeb60to16 corpus.

```
    # Run inference
   task_inference(test_name='bilstm_acd_ta_wav2vec',
                   task_config_registration_info="internal_key:instance--flag:task--framework:tf--tags:['ablation_study=text', 'lstm', 'calibrated', 'task_type=acd', 'text_audio', 'use_audio_features=False']--namespace:us_elec",
                   task_folder='us_elec',
                   save_results=False,
                   framework_config_name=None,
                   debug=False
                   )
```

## SVM classifier

SVM classifier require a dedicated script and are found in `deasy-speech/runnables/other/` folder.
The following arguments are generally shared among SVM scripts:

```
is_calibrating=True         # enables calibration
use_text_features=True      # use text modality
use_audio_features=True     # use audio modality
use_audio_data=True         # use embedding-based audio representation (feature-based if set to False)
test_name=''                # folder where to save test results
save_results=True           # whether to save test results or not
```

## Baselines

Similarly to SVM, we provide dedicated scripts for running baselines like the random baseline reported in the paper.
These scripts are found in `deasy-speech/runnables/other/`


## Disclaimer

This repository contains experimental software and is published for the sole purpose of giving additional 
background details on the respective publication.
