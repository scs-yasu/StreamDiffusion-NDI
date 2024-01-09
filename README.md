# StreamDiffusion-NDI
<a href="https://discord.com/invite/wNW8xkEjrf"><img src="https://discord.com/api/guilds/838923088997122100/widget.png?style=shield" alt="Discord Shield"/></a>

NDI & OSC extension for real-time Stable Diffusion interactive generation with [StreamDiffusion](https://pages.github.com/](https://github.com/cumulo-autumn/StreamDiffusion)https://github.com/cumulo-autumn/StreamDiffusion).

## Features:
* NDI (video) streaming over the network or localhost.
* OSC communication for prompt and FPS.

## Installation:
Supported version Python 3.10
1) Install [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion?tab=readme-ov-file#installation) with TensorRT.
2) Install SteamDiffusion-NDI in StreamDiffusion environment with ```pip install -r requirements.txt```

## Usage:
1) Accelerate model with an [example](https://github.com/cumulo-autumn/StreamDiffusion/blob/main/examples/img2img/single.py) script  (temporal solution). Change [acceleration](https://github.com/cumulo-autumn/StreamDiffusion/blob/63a240a771247968b0fed9877d3c0436d3110b86/examples/img2img/single.py#L24C7-L24C7) to ```tensorrt```.
2) Configure ```config.json```
3) Run in StreamDiffusion environment ```python sd_ndi.py```
4) Add NDI output to send images in Stable Diffusion and NDI input (SD-NDI) to receive processed images
5) Send string with OSC at ```/prompt``` address change the prompt during the inference
6) You can get inference FPS at ```/fps``` address on client side

## Config:
Look in ```config.json``` for an example configuration.

	"sd_model": "path to diffusers model",
	"t_index_list": [Number of inference steps],
	"engine": "path to the folder with the accelerated model",
	"min_batch_size": depends on your configuration,
	"max_batch_size": depends on your configuration,
 	"ndi_name": "NDI client name to recieve from",
	"osc_out_adress": client address for receiving FPS value,
	"osc_out_port": client port for receiving FPS value,
	"osc_in_adress": server address for receiving commands.,
	"osc_in_port": server port for receiving commands.


## This project is a fork of [StreamDiffusion-NDI](https://github.com/olegchomp/StreamDiffusion-NDI) with additional features and modifications. 

The main changes include:

1. The addition of a TouchDesigner folder containing TD files.
2. The `sd_ndi_td.py` script has been modified to interact with TouchDesigner and handle two types of prompts.
3. The NDI source can be refreshed and selected.
4. A batch file has been created to activate the conda environment and start the Python script.

## このプロジェクトは、追加の機能と修正を加えた[StreamDiffusion-NDI](https://github.com/olegchomp/StreamDiffusion-NDI)のフォークです。

主な変更点は以下の通りです：

1. TDファイルを含むTouchDesignerフォルダを追加しました。
2. `sd_ndi_td.py`スクリプトを修正して、TouchDesignerとの対話と2種類のプロンプトを処理できるようにしました。
3. NDIソースをリフレッシュし、選択できるようにしました。
4. conda環境をアクティベートし、Pythonスクリプトを開始するためのバッチファイルを作成しました。