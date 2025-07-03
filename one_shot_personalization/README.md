```md
# One‑Shot Personalization (RealVisXL V5.0 × InstantID)

```bash
# 1) install deps
pip install -r requirements.txt

# 2) fetch community pipeline if not already present
wget -nc -q \
  https://raw.githubusercontent.com/huggingface/diffusers/main/examples/community/pipeline_stable_diffusion_xl_instantid.py

sudo apt-get update
sudo apt-get install -y libgl1

# 3) run
python3 -m one_shot_personalization.demo
``` 