# DreamBooth training example

[DreamBooth](https://arxiv.org/abs/2208.12242) 

## Running locally with PyTorch

### Installing the dependencies

Before running the scripts, make sure to install the library's training dependencies:

**Important**

To make sure you can successfully run the latest versions of the example scripts, we highly recommend **installing from source** and keeping the install up to date as we update the example scripts frequently and install some example-specific requirements. To do this, execute the following steps in a new virtual environment:
```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install -e .
```

Then cd in the example folder and run
```bash
pip install -r requirements.txt
```

And initialize an Accelerate environment with:

```bash
accelerate config
```

Or for a default accelerate configuration without answering questions about your environment

```bash
accelerate config default
```

Or if your environment doesn't support an interactive shell e.g. a notebook

```python
from accelerate.utils import write_basic_config
write_basic_config()
```

When running `accelerate config`, if we specify torch compile mode to True there can be dramatic speedups.
Note also that we use PEFT library as backend for LoRA training, make sure to have `peft>=0.6.0` installed in your environment.

### Itay example

Now let's get our dataset. For this example we will use some dog images: 

Let's first download it locally:

```python
from huggingface_hub import snapshot_download

local_dir = "./itay"
snapshot_download(
    "huyhuung/itay",
    local_dir=local_dir, repo_type="dataset",
    ignore_patterns=".gitattributes",
)
```

And launch the training using:

```bash
export MODEL_NAME="SG161222/Realistic_Vision_V6.0_B1_noVAE"
export INSTANCE_DIR="itay"
export OUTPUT_DIR="path-to-save-model"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of TOK person" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=400 \
  --push_to_hub
```

### Training with prior-preservation loss

Prior-preservation is used to avoid overfitting and language-drift. Refer to the paper to learn more about it. For prior-preservation we first generate images using the model with a class prompt and then use those during training along with our data.
According to the paper, it's recommended to generate `num_epochs * num_samples` images for prior-preservation. 200-300 works well for most cases. The `num_class_images` flag sets the number of images to generate with the class prompt. You can place existing images in `class_data_dir`, and the training script will generate any additional images so that `num_class_images` are present in `class_data_dir` during training time.

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="dog"
export CLASS_DIR="path-to-class-images"
export OUTPUT_DIR="path-to-save-model"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a photo of TOK person" \
  --class_prompt="a photo of dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=800 \
  --push_to_hub
```


### Training on a 16GB GPU:

With the help of gradient checkpointing and the 8-bit optimizer from bitsandbytes it's possible to run train dreambooth on a 16GB GPU.

To install `bitsandbytes` please refer to this [readme](https://github.com/TimDettmers/bitsandbytes#requirements--installation).

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="dog"
export CLASS_DIR="path-to-class-images"
export OUTPUT_DIR="path-to-save-model"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks dog" \
  --class_prompt="a photo of dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=2 --gradient_checkpointing \
  --use_8bit_adam \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=800 \
  --push_to_hub
```

### Training

```bash
export MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
export INSTANCE_DIR="dog"
export OUTPUT_DIR="path-to-save-model"
```

For this example we want to directly store the trained LoRA embeddings on the Hub, so
we need to be logged in and add the `--push_to_hub` flag.

```bash
huggingface-cli login
```

Now we can start training!

```bash
accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of TOK person" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --checkpointing_steps=100 \
  --learning_rate=1e-4 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --validation_prompt="A photo of TOK person" \
  --validation_epochs=50 \
  --seed="0" \
  --push_to_hub
```

### Inference

After training, LoRA weights can be loaded very easily into the original pipeline. First, you need to
load the original pipeline:

```python
from diffusers import DiffusionPipeline
pipe = DiffusionPipeline.from_pretrained("base-model-name").to("cuda")
```

Next, we can load the adapter layers into the pipeline with the [`load_lora_weights` function](https://huggingface.co/docs/diffusers/main/en/using-diffusers/loading_adapters#lora).

```python
pipe.load_lora_weights("path-to-the-lora-checkpoint")
```

Finally, we can run the model in inference.

```python
image = pipe("A picture of a sks dog in a bucket", num_inference_steps=25).images[0]
```

If you are loading the LoRA parameters from the Hub and if the Hub repository has
a `base_model` tag (such as [this](https://huggingface.co/patrickvonplaten/lora_dreambooth_dog_example/blob/main/README.md?code=true#L4)), then
you can do:

```py
from huggingface_hub.repocard import RepoCard

lora_model_id = "patrickvonplaten/lora_dreambooth_dog_example"
card = RepoCard.load(lora_model_id)
base_model_id = card.data.to_dict()["base_model"]

pipe = StableDiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16)
...
```

If you used `--train_text_encoder` during training, then use `pipe.load_lora_weights()` to load the LoRA
weights. For example:

```python
from huggingface_hub.repocard import RepoCard
from diffusers import StableDiffusionPipeline
import torch

lora_model_id = "sayakpaul/dreambooth-text-encoder-test"
card = RepoCard.load(lora_model_id)
base_model_id = card.data.to_dict()["base_model"]

pipe = StableDiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
pipe.load_lora_weights(lora_model_id)
image = pipe("A picture of a sks dog in a bucket", num_inference_steps=25).images[0]
```

Note that the use of [`StableDiffusionLoraLoaderMixin.load_lora_weights`](https://huggingface.co/docs/diffusers/main/en/api/loaders#diffusers.loaders.StableDiffusionLoraLoaderMixin.load_lora_weights) is preferred to [`UNet2DConditionLoadersMixin.load_attn_procs`](https://huggingface.co/docs/diffusers/main/en/api/loaders#diffusers.loaders.UNet2DConditionLoadersMixin.load_attn_procs) for loading LoRA parameters. This is because
`StableDiffusionLoraLoaderMixin.load_lora_weights` can handle the following situations:

* LoRA parameters that don't have separate identifiers for the UNet and the text encoder (such as [`"patrickvonplaten/lora_dreambooth_dog_example"`](https://huggingface.co/patrickvonplaten/lora_dreambooth_dog_example)). So, you can just do:

  ```py
  pipe.load_lora_weights(lora_model_path)
  ```

* LoRA parameters that have separate identifiers for the UNet and the text encoder such as: [`"sayakpaul/dreambooth"`](https://huggingface.co/sayakpaul/dreambooth).