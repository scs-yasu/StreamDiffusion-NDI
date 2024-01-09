import torch
from diffusers import AutoencoderTiny, StableDiffusionPipeline
from diffusers.utils import load_image

from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import postprocess_image
from streamdiffusion.acceleration.tensorrt import accelerate_with_tensorrt

import time
import numpy as np
import cv2 as cv
import NDIlib as ndi

from pythonosc import udp_client
from pythonosc import osc_server
from pythonosc.dispatcher import Dispatcher
from threading import Thread
from typing import List, Any, Tuple
import json
from PIL import Image

import sys
sys.path.append('C:\\StreamDiffusion')
from utils.wrapper import StreamDiffusionWrapper

def process_image(image_np: np.ndarray, range: Tuple[int, int] = (-1, 1)) -> Tuple[torch.Tensor, np.ndarray]:
    image = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
    r_min, r_max = range[0], range[1]
    image = image * (r_max - r_min) + r_min
    return image.unsqueeze(0), image_np


def np2tensor(image_np: np.ndarray) -> torch.Tensor:
    height, width, _ = image_np.shape
    imgs = []
    img, _ = process_image(image_np)
    imgs.append(img)
    imgs = torch.vstack(imgs)
    images = torch.nn.functional.interpolate(
        imgs, size=(height, width), mode="bilinear", align_corners=False
    )
    image_tensors = images.to(torch.float16)
    return image_tensors

def oscprompt1(address: str, *args: List[Any]) -> None:
    if address == "/prompt1":
        global osc_prompt1
        osc_prompt1 = args[0]

def oscprompt2(address: str, *args: List[Any]) -> None:
    if address == "/prompt2":
        global osc_prompt2
        osc_prompt2 = args[0]

def load_config(file_path):
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config

# Load config
config_data = load_config('config.json')
sd_model = config_data['sd_model']
t_index_list = config_data['t_index_list']
engine = config_data['engine']
min_batch_size = config_data['min_batch_size']
max_batch_size = config_data['max_batch_size']
osc_out_adress = config_data['osc_out_adress']
osc_out_port = config_data['osc_out_port']
osc_in_adress = config_data['osc_in_adress']
osc_in_port = config_data['osc_in_port']

osc_guidance_scale = 1.2
prompt = "default prompt text"
default_prompt = "Portrait of The Joker halloween costume, face painting, with , glare pose, detailed, intricate, full of colour, cinematic lighting, trending on artstation, 8k, hyperrealistic, focused, extreme details, unreal engine 5 cinematic, masterpiece"
default_negative_prompt = "black and white, blurry, low resolution, pixelated,  pixel art, low quality, low fidelity"


# You can load any models using diffuser's StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained(sd_model).to(
    device=torch.device("cuda"),
    dtype=torch.float16,
)

frame_buffer_size = 1

stream = StreamDiffusionWrapper(
            model_id_or_path=sd_model,
            use_tiny_vae=True,
            device=torch.device("cuda"),
            dtype=torch.float16,
            t_index_list=[25],
            frame_buffer_size=frame_buffer_size,
            width=512,
            height=512,
            use_lcm_lora=False,
            output_type="pil",
            warmup=10,
            vae_id=None,
            acceleration="tensorrt",
            mode="img2img",
            use_denoising_batch=True,
            cfg_type="none",
            use_safety_checker=False,
            # enable_similar_image_filter=True,
            # similar_image_filter_threshold=0.98,
        )

stream.prepare(
            prompt=default_prompt,
            negative_prompt=default_negative_prompt,
            num_inference_steps=50,
            guidance_scale=1.2,
        )


# NDI
ndi_find = ndi.find_create_v2()

sources = []
while True:
    print('Looking for sources ...')
    ndi.find_wait_for_sources(ndi_find, 1000)
    sources = ndi.find_get_current_sources(ndi_find)
    
    if len(sources) > 0:
        for i, source in enumerate(sources):
            print(f"{i+1}: {source.ndi_name}")
        
        index = int(input("Enter the index of the source to connect to, or 0 to refresh: ")) - 1
        
        if index == -1:
            continue
        elif 0 <= index < len(sources):
            target_source = sources[index]
            break
        else:
            print("Invalid index. Please try again.")
    else:
        print("No sources found. Retrying...")

ndi_recv = ndi.recv_create_v3()
ndi.recv_connect(ndi_recv, target_source)
print(f"Connected to NDI source: {target_source.ndi_name}")

print('NDI connected')

ndi.find_destroy(ndi_find)
cv.startWindowThread()
send_settings = ndi.SendCreate()
send_settings.ndi_name = 'SD-NDI'
ndi_send = ndi.send_create(send_settings)
video_frame = ndi.VideoFrameV2()

# OSC
server_address = osc_out_adress
server_port = osc_out_port
client = udp_client.SimpleUDPClient(server_address, server_port)

server_address = osc_in_adress
server_port = osc_in_port
osc_prompt1 = None
osc_prompt2 = None

dispatcher = Dispatcher()
dispatcher.map("/prompt1", oscprompt1)
dispatcher.map("/prompt2", oscprompt2)

server = osc_server.ThreadingOSCUDPServer(
      (server_address, server_port), dispatcher)

server_thread = Thread(target=server.serve_forever)
server_thread.start()

# Run the stream infinitely
try:
    while True:
        if osc_prompt1 is not None:

            prompt = str(osc_prompt1) + "," + str(osc_prompt2)
            stream.prepare(
                prompt = prompt + ",masterpiece best quality,NSFW",
                negative_prompt = "(worst quality:2),(low quality:2),(normal quality:2),(deformed:1.3),(monochrome:1.2),(grayscale:1.2)",
                guidance_scale = osc_guidance_scale
                )
            # Process the received message within the loop as needed
            print(f"Prompt1: {osc_prompt1}")
            print(f"Prompt2: {osc_prompt2}")
            # Reset the shared_message variable
            shared_message = None

        t, v, _, _ = ndi.recv_capture_v2(ndi_recv, 5000)

        if t == ndi.FRAME_TYPE_VIDEO:

            frame = np.copy(v.data)
            framergb = cv.cvtColor(frame, cv.COLOR_BGRA2BGR)
            pil_image = Image.fromarray(framergb)

            image_tensor = stream.preprocess_image(pil_image)
            output_image = stream(image=image_tensor, prompt=prompt)

            start_time = time.time()

            open_cv_image = np.array(output_image)
            img = cv.cvtColor(open_cv_image, cv.COLOR_RGB2RGBA)
            ndi.recv_free_video_v2(ndi_recv, v)

            video_frame.data = img
            video_frame.FourCC = ndi.FOURCC_VIDEO_TYPE_BGRX

            ndi.send_send_video_v2(ndi_send, video_frame)

            fps = 1 / (time.time() - start_time)

            client.send_message("/fps", fps)

except KeyboardInterrupt:
    # Handle KeyboardInterrupt (Ctrl+C)
    print("KeyboardInterrupt: Stopping the server")
finally:
    # Stop the server when the loop exits
    ndi.recv_destroy(ndi_recv)
    ndi.send_destroy(ndi_send)
    ndi.destroy()
    cv.destroyAllWindows()
    server.shutdown()
    server_thread.join()
