## Textual Inversion fine-tuning example

[Textual inversion](https://arxiv.org/abs/2208.01618).

## Running on Colab

Colab for training
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/sd_textual_inversion_training.ipynb)

## Running locally with PyTorch
### Installing the dependencies

Before running the scripts, make sure to install the library's training dependencies:

**Important**

To make sure you can successfully run the latest versions of the example scripts, we highly recommend **installing from source** and keeping the install up to date as we update the example scripts frequently and install some example-specific requirements. To do this, execute the following steps in a new virtual environment:
```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
```

Then cd in the example folder and run:
```bash
pip install -r requirements.txt
```

And initialize an Accelerate environment with:

```bash
accelerate config
```

### Itay example

First, let's login so that we can upload the checkpoint to the Hub during training:

```bash
huggingface-cli login
```

Now let's get our dataset. For this example we will use some itay person images: https://huggingface.co/datasets/huyhuung/Itay

Let's first download it locally:

```py
from huggingface_hub import snapshot_download

local_dir = "./itay"
snapshot_download("huyhuung/Itay", local_dir=local_dir, repo_type="dataset", ignore_patterns=".gitattributes")
```

This will be our training data.
Now we can launch the training using:


```bash
export MODEL_NAME="SG161222/Realistic_Vision_V6.0_B1_noVAE"
export DATA_DIR="./itay"

accelerate launch textual_inversion.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token="<cat-toy>" \
  --initializer_token="toy" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=3000 \
  --learning_rate=5.0e-04 \
  --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --push_to_hub \
  --output_dir="textual_inversion_cat"
```


**Note**: To use multiple embedding vectors, you should define `--num_vectors`
to a number larger than one, *e.g.*:
```bash
--num_vectors 5
```

The saved textual inversion vectors will then be larger in size compared to the default case.

### Inference

Once you have trained a model using above command, the inference can be done simply using the `StableDiffusionPipeline`. Make sure to include the `placeholder_token` in your prompt.

```python
from diffusers import StableDiffusionPipeline
import torch

model_id = "path-to-your-trained-model"
pipe = StableDiffusionPipeline.from_pretrained(model_id,torch_dtype=torch.float16).to("cuda")

repo_id_embeds = "path-to-your-learned-embeds"
pipe.load_textual_inversion(repo_id_embeds)

prompt = "A photo of <itay>"
negative_promt = "ugly"
image = pipe(prompt, negative_prompt= negative_prompt, num_inference_steps=30, guidance_scale=7.5).images[0]

image.save("itay.png")
```