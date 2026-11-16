"""Run a saved prototype on a stereo PCM16 WAV and write JSON window results."""
import argparse
import json
from pathlib import Path
import torch
from runtime import VAEEngine, predict_wav
from prediction_summary import summarize_prediction, print_prediction_summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--wav',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('prediction.json'))
    p.add_argument('--max-windows',type=int,default=32,help='Default: first 32 full windows; 0: all full windows')
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    a=p.parse_args();torch.set_num_threads(a.threads)
    device_options = {} if a.device == 'auto' else {'device': a.device}
    engine=VAEEngine.from_checkpoint(a.checkpoint, **device_options)
    print('Checkpoint loaded successfully.',flush=True)
    print(f'Device: {engine.device}',flush=True)
    result=predict_wav(engine,a.wav,a.max_windows,a.batch_size)
    result['checkpoint_file']=str(a.checkpoint.resolve())
    result['summary']=summarize_prediction(result)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print_prediction_summary(result['summary'])


if __name__=='__main__':main()
