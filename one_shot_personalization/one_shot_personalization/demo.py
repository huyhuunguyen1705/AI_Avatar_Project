import gradio as gr
from PIL import Image

from . import generator, config


def _run(face: Image.Image, prompt: str, neg: str, id_scale: float, upscale: float, cfg: float):
    return generator.generate(face, prompt, neg, id_scale, upscale, cfg)


def main():
    with gr.Blocks() as demo:
        gr.Markdown("# 🖼️ One‑Shot Personalization (RealVisXL V5.0 × InstantID)")

        with gr.Row():
            with gr.Column():
                face = gr.Image(label="Upload face", type="pil", sources=["upload", "webcam"])
                prompt = gr.Textbox(label="Prompt", lines=2, placeholder="A cinematic portrait of the person …")
                neg = gr.Textbox(label="Negative", value="bad hands, ugly, watermark")
                id_scale = gr.Slider(0.0, 1.0, value=0.8, label="Identity strength")
                upscale = gr.Slider(1.0, 2.0, value=1.5, label="Upscale factor")
                cfg = gr.Slider(config.CFG_MIN, config.CFG_MAX, value=1.5, label="CFG scale (HiresFix)")
                btn = gr.Button("Generate")
            with gr.Column():
                out1 = gr.Image(label="InstantID (1024px)")
                out2 = gr.Image(label="HiresFix")

        btn.click(_run, inputs=[face, prompt, neg, id_scale, upscale, cfg], outputs=[out1, out2])

    demo.launch()


if __name__ == "__main__":
    main()