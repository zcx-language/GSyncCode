<div align="center">

# GSyncCode: Geometry Synchronous Hidden Code for
One-step Photography Decoding

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

## Description

Invisible hyperlinks and hidden barcodes have recently emerged as a hot topic in offline-to-online messaging, where an invisible message or barcode is embedded in an image and can be decoded via camera shooting. Current schemes involve a two-step decoding process: starting with vertex localization of the embedded region to correct the perspective distortion introduced by shooting, followed by decoding the message from the corrected region. However, vertex localization can be complex and time-consuming, which affects the efficiency and accuracy of message decoding. To address this issue, this paper proposes a geometry synchronous decoding scheme called GSyncCode, allowing for one-step extraction of a Data Matrix code from the photograph. Instead of correction before decoding, GSyncCode directly decodes a geometry-transformed Data Matrix that is synchronized with the embedded region. A barcode scanner is then used to efficiently retrieve messages. We design a Haar-transform based encoder HaarUNet and a HaarLoss visual function to select the key component of the Data Matrix for embedding. They improve the visual quality of the embedded image by reducing redundant embedding signals. Extensive simulated and real-world experiments demonstrate the superiority of GSyncCode in both decoding efficiency and accuracy.

## How to run

Install dependencies

```bash
# clone project
git clone https://github.com/zcx-language/GSyncCode
cd GSyncCode

# [OPTIONAL] create conda environment
conda create -n myenv python=3.10
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```
The decoding results under the [Screen-camera](https://youtu.be/WbEb_JnJRaM) scenario and the [Printer-camera](https://youtu.be/PSlMney6AO4) scenario. 
