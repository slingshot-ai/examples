# MNIST

This is an example [slingshot](https://www.slingshot.xyz/) project, which uses the MNIST dataset to train a model that
recognises digits in the form of images.

## Project Setup

0. Clone the repository locally via github.

0. `cd` into your newly created `example-mnist` directory.

0. Connect your local copy to a slingshot project you own:
   * If you already have an empty project for this, you should `slingshot use` and select it:
      ```bash
      $ slingshot use

      Select the project you want to work on:
        my-project-001cd
      > mnist-example-002de
      ```
   
   * Otherwise, instead use `slingshot init` which will guide you to link the folder to a new project.

0. Push the source code to your project and apply the `slingshot.yaml`:
    ```bash
    $ slingshot push

    Pushing code to Slingshot (11.11 KiB)...
    Pushed new source code 'fancy-walk-1', view in browser at https://app.slingshot.xyz/project/mnist-example/code/...
    Detected new environment 'training-env'
    Detected new environment 'inference-env'
    Detected new run 'create-dataset-artifact '
    Detected new run 'train-model'
    Detected new deployment 'classifier-deployment'

    Do you want to apply these changes? [Y/n]: Y
    Applying 'slingshot.yaml' for project 'mnist-example'.
    Creating environment 'training-env'...✅
    Creating environment 'inference-env'...✅
    Creating 'create-dataset-artifact '...✅
    Creating 'train-model'...✅
    Creating 'classifier-deployment'...✅
    ```

That's it! Your project is set up and ready to use.

## Usage

0. Start a run to create an artifact of the MNIST dataset to be used as the training data:
    ```bash
    $ slingshot run start

    Using latest source code 'fancy-walk-1'
    Select a run to use:
    > create-dataset-artifact 
      train-model

    Run created with name 'royal-squirrel-1', view in browser at https://app.slingshot.xyz/project/mnist-example/runs/...
    ```

0. Once the dataset run has completed, start a run to train the classifier model:
    ```bash
    $ slingshot run start

    Using latest source code 'fancy-walk-1'
    Select a run to use:
      create-dataset-artifact 
    > train-model

    Run created with name 'youthful-lobster-2', view in browser at https://app.slingshot.xyz/project/mnist-example/runs/...
    ```

0. When training is completed, you can deploy the model for inference:
    ```bash
    $ slingshot inference start

    Using deployment 'classifier-deployment'
    Model deployed successfully! See more details here:
    https://app.slingshot.xyz/project/mnist-example/deployments/...
    ```

0. Once your model is deployed, you can make predictions by running:
    ```bash
    $ slingshot inference predict test_img.jpg

    Using deployment 'classifier-deployment'
    {'confidence': 0.9998968839645386, 'prediction': '5'}
    ```

    You can substitute the image path (`test_img.jpg`) for any image you like. The image can be RGB or grayscale and of any size, as it will be pre-processed before passing it to the model. 
