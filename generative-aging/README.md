# Generative Aging

### 0. Set up a project on Slingshot

1. Create a new project on ‘[Slingshot](https://dev.slingshot.xyz/)’
2. Set this Slingshot project as active on your local machine:

```
$ slingshot project use
Select the project you want to work on:
mnist
> generative-aging
gpt-exps
```

### 1. Push code to your Slingshot environment

We have provided a script that can be used to run generative aging as a Gradio app through a deployment.

We are going to use the `males` and `females` models from [Lifespan Age Transformation Synthesis](https://github.com/royorel/Lifespan_Age_Transformation_Synthesis). We'll have one deployment for each model, and include a radio button in the Gradio app to switch between them.

You can pull the repo and push the code to Slingshot using the following command:

```bash
$ slingshot push
Pushing code to Slingshot (29.59 KiB)...
Uploading: 100%|███████████████████████████████████████████████████████████████████████████████████████| 30.3k/30.3k [00:00<00:00, 61.4kB/s]
Pushed new source code 'tasty-apple-284', view in browser at https://dev.slingshot.xyz/project/generative-aging/code/5999b4ae30
```

### 2. Prepare Models

During inference, our pipeline consists of a pre-processing step where only the face and hair are retained while the rest of the background and clothing is segmented out. The pre-processed image is then passed through the generator along with the target age embedding to produce the aged or de-aged image of the person.

To prepare the models, use the script `get_models.py` through the `get-models` Run. The script takes care of downloading all the required models and uploading them as Artifacts.

```bash
$  slingshot run start
Select run:
[1] get-models
Selected: get-models (skipped, only option available)
Run created with name 'delicate-hades-1', view in browser at
https://dev.slingshot.xyz/project/generative-aging-6494c/runs/e3b725e5de
Following logs. Ctrl-C to stop, and run 'slingshot run logs delicate-hades-1 --follow' to follow again
```

### 3. Start Deployments

We have provided code in `inference.py` to handle the inference pipeline. Start the deployments using:

```bash
$ slingshot inference start
Select deployment:
> gen_aging_deployment_female
  gen_aging_deployment_male
Deployment started successfully! See details here: https://dev.slingshot.xyz/project/generative-aging/deployments/40ac06e6f0

$ slingshot inference start
Select deployment:
  gen_aging_deployment_female
> gen_aging_deployment_male
Deployment started successfully! See details here: https://dev.slingshot.xyz/project/generative-aging/deployments/40ac06e6f0
```

You can view both deployments on the front end through the **Deployments** tab.

### 4. Gradio App

The Gradio app is configured through the `gradio_app.py` and the `slingshot.yaml`. You can start the app through the CLI using the following command:

```bash
$ slingshot app start
Select app:
> gradio_app
  session
```

Follow the link provided to access your front-end app.

Happy aging/de-aging!
