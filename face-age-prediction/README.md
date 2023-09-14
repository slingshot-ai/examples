# Face Age Prediction 👧👩👵

Humans are especially well-adapted to make an estimate of age based on the image of a human face. We can reasonably
hypothesize that our ability to do this is based on certain features of the face like the size, texture of the skin,
wrinkles, and maybe gender.

In this project we model the problem of face age prediction as an expectation optimization problem. We use a pre-trained
ResNet-18 as our image feature extractor and a linear layer to predict the age. We use the APPA-REAL dataset for 
training and validating our model. At the end we actually test our model with "in the wild" images via a live deployment
and UI.

### 0. Clone this repo locally

Start by cloning this repo locally, so you have access to all files.
```bash
$ git clone https://github.com/slingshot-ai/examples.git
```


### 1. Set up a project on Slingshot

1. Create a new project on [Slingshot](https://app.slingshot.xyz/)
1. Set this Slingshot project as active on your local machine:

```
$ slingshot project use
Select the project you want to work on:
mnist
> face-age-prediction
```

### 2. Ingest an appropriate dataset as a slingshot artifact

For training, we are going to use the [APPA-REAL dataset](https://chalearnlap.cvc.uab.cat/dataset/26/description/). You
can download this dataset on your local machine -- we'll upload it to slingshot next.

Note: If the download comes as a zipped file, you'll have to unzip it before uploading it.

Use the command below to upload the dataset to our project as an artifact:

```
$ slingshot artifact upload appa-real-release --name appa-real-dataset
```


### 3. Training and Hyperparameter tuning

In this repository you'll find a training script along with the dataloader (for the APPA-REAL dataset) to get started.
In the script, we'll fine-tune a ResNet-18 model pre-trained on ImageNet and a fully connected layer attached at the
end to make our final predictions.

> **Note**: ResNet-18 is a pretty small model by today's standards. You can try to swap this out for a larger model or
> perhaps a different architecture, like ViT, and see if MAE reduces further.

Starting a training run with the default hyperparameters, will result in a validation Mean Absolute Error (MAE) of ~4.5.
You can change the hyperparameters under `config_variables` in `slingshot.yaml` to try to get better performing models.

Remember to `slingshot apply` once you are satisfied with your `slingshot.yaml` to persist your changes on slingshot.

Once your params are set, you may start your training run. This can be done in two ways.

**Using the CLI**

You can start your run from the terminal by using the Slingshot CLI. When doing so, the CLI will automatically detect
new source code files and prompt you to push them to Slingshot.

Simply run:

```bash
$ slingshot run start
Code has changed since last push. Do you want to push now? [Y/n]: y
Pushing code to Slingshot (29.59 KiB)...
Select run:
[1] train-model
Selected: train-model (skipped, only option available)
```

The first time you run this, you'll also be prompted to `apply` changes defined in your `slingshot.yaml`. Everytime you
make a change to your YAML, this question will pop up to make sure Slingshot is in sync with your definitions.

**Using the frontend**

You can also start Runs from Slingshot's UI. In this case, you'll first have to push your code explicitly from your
local machine to Slingshot:

```bash
$ slingshot push
Code has changed since last push. Do you want to push now? [Y/n]: y
Pushing code to Slingshot (29.59 KiB)...
```

Now go to the **Runs** page in the Slingshot website and click the **Create Run** button. You'll have the option of
selecting a template to configure your Run. Choose **train-model** to use the template configured from this repo.

You can also modify **Hyperparameters** in the `Config Variables` section.

> **Note** The default hyperparameter values are read from the last `slingshot.yaml` applied. Changes made in the UI 
> will only affect that specific run. If you wish to save your changes permanently, make sure to check the
> **Update Run** checkbox.

Once your parameters are set, kick off your training run using the **Start Run** button at the bottom of the page.

### 4. (Optional) WandB integration

The training script optionally uses [Weights & Biases](https://wandb.ai/) for logging metrics. If you have an account,
you may add your API key to Slingshot's secrets store so the script will automatically log metrics to your W&B account.

You can use the following CLI command or set it up from the UI:
```
$ slingshot secret wandb
wandb: You can find your API key in your browser here: https://wandb.ai/authorize
Paste an API key from your profile and hit enter:
Secret put successfully
```

### 5. Deployment

Once your model is trained and ready to go, you can now use it to make predictions on new images with a Slingshot
Deployment.

In this repo, you'll find a deployment configuration called `face-age-prediction` (setup in the slingshot.yaml). This 
Deployment uses the code in `inference.py`.

**5.1 Start the Deployment!**

```bash
$ slingshot inference start

Select deployment:
[1] face-age-prediction
Selected: face-age-prediction (skipped, only option available)
Deployment started successfully! See details here: https://app.slingshot.xyz/project/...
```

You can view your deployment from the UI in the **Deployments** page.

### 6. UI to test your model

We also provide a simple Gradio interface so you can interact with your model in a ~ nice ~ UI.

**6.2 Using the frontend**

Starting an app via CLI can be done by running: 

```bash
$ slingshot app start

Select app:
[1] front-end-ui
Selected: front-end-ui (skipped, only option available)
Deployment started successfully! See details here: https://app.slingshot.xyz/project/...
```

**6.2 Using the frontend**

On the frontend, you can start the Gradio app by going to the Apps page, choosing the front-end-ui App and clicking the
**Start App** button.

Once a link appears in the page, you can open it in your browser and start testing your deployment!

Happy predicting!