# Digital Humans Project: Learning Motion Priors for LISST Multi-Person Capturing
This code is training a convolutional autoencoder to represent LISST motions as latent variables. These latent variables have a smoothness constraint, so that they can be used as a motion prior for further motion reconstruction. The implementation is based on the paper "Learning Motion Priors for 4D Human Body Capture in 3D Scenes": https://arxiv.org/pdf/2108.10399.pdf

## Usage
The current implementation can be easily run on Google Colab. It is basically Plug-and-Play.

* Download the dataset: https://polybox.ethz.ch/index.php/s/s753qhM7Axmfvdp
* --> Upload to your Google Drive into: /content/drive/MyDrive/LEMO/CMU-canon-MPx8-train.pkl
* Create empty folders:
    * /content/drive/MyDrive/LEMO/preprocess_stats
    * /content/drive/MyDrive/LEMO/runs_try
* Run Google Colab Notebook: LISST_motion_prior.ipynb

The resulting Encoder will be saved to your Drive folder in ".../runs_try"



Code forked from: https://github.com/sanweiliti/LEMO