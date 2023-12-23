import sys
import torch

custom_task_cp_path = sys.argv[1]

custom_task_cp = torch.load(custom_task_cp_path, map_location="cpu")
custom_task_cp['cfg']['task']['_name'] = "audio_pretraining"

torch.save(custom_task_cp, custom_task_cp_path)
