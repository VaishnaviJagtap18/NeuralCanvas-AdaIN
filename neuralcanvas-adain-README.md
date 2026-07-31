# NeuralCanvas — AdaIN Style Transfer

A neural style transfer app that repaints any photo in the style of another image, in real time, using Adaptive Instance Normalization (AdaIN). Upload a content photo and a style reference, and get a stylized result back in one pass, no per-image optimization needed.

**Live app:** https://neuralcanvas-adain-2.onrender.com

## What it does

Most neural style transfer methods optimize a new image pixel by pixel for every content/style pair, which is slow. AdaIN skips that: it aligns the mean and standard deviation of the content image's feature maps to match the style image's, in a single forward pass through the network. You also get an `alpha` slider to blend between the original content and the fully stylized result.

## How it works

1. A pretrained **VGG encoder** extracts feature maps from both the content and style images.
2. **Adaptive Instance Normalization** (`utils/utils.py`) rescales the content features to match the channel-wise mean and variance of the style features.
3. A trained **decoder** network converts those adjusted features back into a full-resolution stylized image.
4. The `alpha` parameter interpolates between the original content features and the AdaIN-normalized features before decoding, so `alpha=1.0` gives full style transfer and lower values keep more of the original photo.

## Tech stack

- **Model**: PyTorch (VGG encoder + trained decoder)
- **Web app**: Flask + Flask-Bootstrap + Flask-WTF
- **Deployment**: Gunicorn on Render, with model weights downloaded at startup from an external URL (they're too large to keep in the repo)

## Project structure

```
flask_app.py              # Flask app: upload form, style transfer route, image serving
train.py                  # training script for the decoder
utils/
├── models.py              # VGGEncoder and Decoder architectures
└── utils.py               # AdaIN function, mean/std calculation, other helpers
templates/                 # HTML templates (Bootstrap-based upload UI)
content_data/               # sample content images
style_data/                 # sample style images
experiment/                  # training experiment outputs / checkpoints
```

## Running it locally

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Provide model weights. Either place them locally at `weights/vgg_normalised.pth` and `weights/decoder_50.pth`, or set download URLs as environment variables:
   ```
   VGG_WEIGHTS_URL=<url to vgg_normalised.pth>
   DECODER_WEIGHTS_URL=<url to decoder_50.pth>
   ```
3. Run the app:
   ```bash
   python flask_app.py
   ```
4. Open `http://localhost:5000`, upload a content image and a style image, adjust alpha, and generate.

## Training your own decoder

`train.py` trains the decoder against a frozen VGG encoder using content and style datasets:

```bash
python train.py --content_dir path/to/content_data --style_dir path/to/style_data --vgg path/to/vgg_normalised.pth --epochs 1
```

Key arguments: `--content_weight` and `--style_weight` control the loss balance, `--batch_size` and `--lr` control training, and `--resume` picks up from a saved checkpoint via `--decoder_path` / `--optimizer_path`.

## Reference

Based on Huang & Belongie's *Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization* (2017).
