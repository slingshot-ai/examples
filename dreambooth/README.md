# Dreambooth 🪄

There are several text-to-image models like Stable Diffusion and Imagen available that allow you to produce images of objects or fictional characters in completely novel settings. It is as simple as asking the model to generate an image of "Darth Vader baking for the Cookie Monster". But what if you wanted to insert yourself in the image? What if Darth Vader were baking for you? What if you wanted a picture of you and your friends taking a selfie with Darth Vader?
Stable Diffusion and similar models have no concept of you or your friends, so simply inserting your names wouldn't get the job done. If only there were a way to _teach_ these models the concept of _you_...

Dreambooth is a method that allows you to do exactly that!

It is a training regime used to personalize text-to-image models. In this project, we will use
[Stable Diffusion v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) to produce personalized images from
descriptive texts.

### 0. Set up a project on Slingshot

1. If you haven't already, install the Slingshot SDK:

```bash
$ pip install slingshot-ai
```

2. Create a new project on [Slingshot](https://app.slingshot.xyz/)
3. Set this Slingshot project as active on your local machine:

```bash
$ slingshot project use

Select the project you want to work on:
dishani_mnist
> dreambooth_dishani
gpt-exps
```

### 1. Prepare and upload your instance images for training

#### 1.1 Preparing images:

1. Clone this repo
2. Select 30 - 50 images of you in various outfits, backgrounds, and expressions and place them in a folder called **instance_images**.
   1. (Optional) If you plan to use captions for these images, create a second folder called **captions**. Caption file names must match the filename of their corresponding image.
3. Place these folders inside a folder called `my_images` inside your project directory. The following snippet showcases an example directory structure.

```bash
$ ls my_images

captions        instance_images
```

#### 1.2 Uploading instance images to Slingshot:

In order to use your images for training, we have to make them accessible in the environment Slingshot uses to run your training script. We'll use Artifacts to do this.
A Slingshot Artifact is simply a file or folder that you can mount onto Runs, Deployments, or Apps to make these files accessible to your code. Let's create one for our instance images.

```bash
$ slingshot artifact upload my_images --tag my_images
```

Notice that we've associated a tag to our artifact. This makes it easy to reference this artifact in other places and attach it wherever we want. If you take a peek inside the `slingshot.yaml` file, you'll see a section that mounts this tag (`dreambooth_input_data`) to our training run.

```bash
- mode: DOWNLOAD
        path: /mnt/my_images
        tag: my_images
```

### 2. Push code to your Slingshot environment

You'll need a training script to train the Dreambooth model. We've provided some starter code (`train.py`) that takes Stable Diffusion 1.5 and fine-tunes it with your provided instance images.
Push this code into your Slingshot project by running the following command:

```bash
$ slingshot push

Pushed new source code 'teeming-magic-111', view in browser at [https://app.slingshot.xyz/project/dreambooth_dishani/code/14c353232b](https://app.slingshot.xyz/project/dreambooth_dishani/code/14c353232b)
```

Now, you are all set to train and generate your dream images!

### 3. Set hyperparameters and train the model

The hyperparameters used to train Dreambooth are very particular to each person. The files provided in this example project already contain values found to be good for most Slingshot Team members, however, feel free to experiment with your own values and see what works best for your images.

You can execute your training and testing code by creating a **Run**. This can be done through the UI or the CLI.

**Using the CLI**

You can see what values are being used in the `slingshot.yaml` file. These hyperparameters are defined within the `config_variables` section of a Run. If you do change them, remember to save the file apply the changes by running:

```bash
$ slingshot apply
```

Once your parameters are set, kick off your training run using:
```bash
$ slingshot run start
Select run:
> dreambooth-train
Selected: dreambooth-train
```

**Using the frontend**

On the front end, you can modify **Hyperparameters** by going to the **Create Run** section of the **Runs** tab, then select the **dream_run_train** template from the dropdown.

Note that the default hyperparameter values are read from the `slingshot.yaml` file. Changes made in the UI will only apply to that specific run. If you wish to save your changes permanently, edit `slingshot.yaml` directly, then push the file to your project using `slingshot apply`.

Once your parameters are set, kick off your training run using the **Start Run** button at the bottom of the page.

### 4. Generate Images

There are several ways to generate images from your trained model. You can start a deployment and consume your
model via API, or, for a more interactive experience, you can use a simple Gradio frontend we've included as a nice UI to interface with your
model.

#### Starting the Front End UI App

First, make sure you've started your `image-generation` deployment on the Deployments page. 

Next, start the Gradio UI by going to the Apps page, selecting the `interactive-interface` app, then clicking on
**Start**. It'll take a few seconds to start up, but soon you'll see a link to the web interface show up.
Follow the prompts in the UI to generate images of Darth Vader baking cookies for you, or anything else you want!

### 5. Improve your results

We've found during our experiments that fine-tuning the base Stable Diffusion once doesn't always produce nice results. We tried multiple approaches and found that carrying on training from an already fine-tuned Dreambooth model helps produce better resulting images.
If you're not getting satisfactory results, consider starting another session of training on top of your current Dreambooth model.

To do this, set the `RESUME_TRAINING` parameter in `slingshot.yaml` as `True`. Remember to run `slingshot apply` in order to see these changes reflected. Alternatively, you can do the same through the frontend by changing the **Hyperparameters** on the **Create Run** page.

**Note:** Before starting the second training run, ensure that the latest artifact has been selected under the DOWNLOAD mount `dreambooth-trained-model-example`.

You can continue training multiple times with different hyperparameter values to improve your results. Adding additional images with more variety can also help improve your outputted images.

Happy generating!
###