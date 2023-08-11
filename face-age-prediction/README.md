# Face Age Prediction 👧👩👵

As humans, there is one job that we constantly do subconsciously and want to be good at and that is making a correct estimation or prediction. In order to do that, we have made several mental models that help us. 

If we have to guess a person’s age from a photo, we would be easily able to make a fair estimate of their age. Upon retrospecting, one would realize that they actually focused on certain features of the face like the size, texture of the skin, wrinkles, and maybe gender. 

Similarly, for this project, to train and deploy an ML model for face age prediction, we will use an encoder that takes the image of a person as input and compresses this huge spatial information into a lower-sized embedding space or feature space. These features are now rich in semantic information rather than local information which is not very useful in predicting the age of a person.

### 0. Set up a project on Slingshot

1. Create a new project on [Slingshot](https://app.slingshot.xyz/)
2. Set this Slingshot project as active on your local machine:

```
$ slingshot project use
Select the project you want to work on:
dishani_mnist
> face-age-prediction
gpt-exps
```

### 1. Dataset download and upload as an Artifact

For training, we are going to use the [APPA-REAL dataset](https://chalearnlap.cvc.uab.cat/dataset/26/description/). You can download this dataset on your local machine at any location temporarily. 

Use the command below to upload the dataset to our project as an artifact. This lets us access it freely at any point just by referencing the tag used to upload.

```
$ slingshot artifact upload appa-real-release --tag appa-real-dataset
```

### 2. Push code to your Slingshot environment

We have provided a training script along with the dataloader (for the APPA-REAL dataset) to get you started. In the script, we have fine-tuned the ResNet-18 model (and the fully connected layers modified for our problem statement) with the pre-trained weights. Using this model along with the default hyperparameters, you will observe a Mean Absolute Error (MAE) of ~5. 

**An exercise for the readers:**

ResNet-18 is a pretty small model considering the high parameter-sized models currently available. You can swap this out with a larger model and report if MAE reduces or saturates after a certain point. 

You can pull the repo and push the code to Slingshot using the following command:

```bash
$ slingshot push

Pushing code to Slingshot (29.59 KiB)...
Uploading: 100%|███████████████████████████████████████████████████████████████████████████████████████| 30.3k/30.3k [00:00<00:00, 61.4kB/s]
Pushed new source code 'tasty-apple-284', view in browser at https://app.slingshot.xyz/project/face-age-prediction/code/5999b4ae30
```

### 3. Training and Hyperparameter tuning

**Using the CLI**

You can change the hyperparameters under `config variables` in the `slingshot.yaml` itself. 

Remember to `slingshot apply` once you are satisfied with your `slingshot.yaml` in order to see these changes reflected.

Once your params are set, start your training using the following command:

```bash
$ slingshot run start

Select run:
> train
  download-model
Selected: train

```

**Using the frontend**

On the front end, you can modify **Hyperparameters** by going to the **Create Run** section of the **Runs** tab, then selecting the **train** template from the dropdown.

Note that the default hyperparameter values are read from the `slingshot.yaml` file. Changes made in the UI will only apply to that specific run. If you wish to save your changes permanently, make sure to check the **Create New Template** toggle.

Once your parameters are set, kick off your training run using the **Start Run** button at the bottom of the page.

### 4. Logs and Saved Models

We print terminal logs for per-epoch validation MAE and training loss. We also save the model that has the best validation MAE during training.

To view these metrics in Weights & Biases, add you W&B API key using the command:

```
$ slingshot secret wandb

wandb: You can find your API key in your browser here: https://wandb.ai/authorize
Paste an API key from your profile and hit enter:
Secret put successfully
```

### 5. Deployment

Once your model is trained and ready to go, your next step should be to start a Deployment.

We've provided a deployment called `face-age-prediction` and its inference code in `inference.py`.

During inference, the pipeline consists of:
1. The face bounding box detection
2. Extraction of the face (with some margin) 
3. Prediction of age using the model we trained

We'll use a separate model to power step 1

**5.1 Download detection model and upload as artifact**

To get the face detector model, you can use a Run to execute the script `get_model.py`. 

```bash
$ slingshot run start

Select run:
  train
> download-model
Selected: download-model

```
The script downloads the models and uploads them as an Artifact with a tag `detector_model` which can be used to reference the models later in the deployment.


**5.2 Start the Deployment!**
```bash
$ slingshot inference start

Select deployment:
[1] face-age-prediction
Selected: face-age-prediction (skipped, only option available)
Deployment started successfully! See details here: https://app.slingshot.xyz/project/face-age-prediction/deployments/40ac06e6f0
```

You can view your deployment on the frontend through the **Deployments** tab.

### 6. Gradio App
You can now start the Gradio app to test your deployment. This can be done through the CLI or through the frontend as well.

**6.1 Using the CLI**

```bash
$ slingshot app start 

Select app:
[1] gradio_app
Selected: gradio_app (skipped, only option available)
Deployment started successfully! See details here: https://app.slingshot.xyz/project/face-age-prediction/deployments/40ac06e6f0
```

**6.2 Using the frontend**

On the frontend, you can start the Gradio app by clicking the **Start App** button on the **Gradio App** tab.

Once you get the link, you can open it in your browser and start testing your deployment!

Happy predicting!