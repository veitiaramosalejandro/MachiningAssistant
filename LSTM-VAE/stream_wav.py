"""Replay a WAV in small chunks through the live inference interface.

--realtime releases each chunk at the time its last sample would become
available from a source. JSONL predictions are flushed as windows complete.
"""
from __future__ import annotations

import argparse
import json
import time
import wave
from pathlib import Path

import numpy as np
import torch

from runtime import MODEL_FAMILY, SAMPLE_RATE, StreamingPredictor, VAEEngine
from prediction_summary import print_prediction_summary


def replay_stream(engine, wav_path, output_path, chunk_ms=20., realtime=False,
                  reference_json=None, on_result=None):
    if chunk_ms <= 0:
        raise ValueError('chunk_ms must be positive')
    chunk_frames=max(1,round(SAMPLE_RATE*chunk_ms/1000))
    output_path=Path(output_path)
    source=wave.open(str(wav_path),'rb')
    try:
        if (source.getframerate(),source.getnchannels(),source.getsampwidth(),source.getcomptype()) != (SAMPLE_RATE,2,2,'NONE'):
            raise ValueError('Expected 44100 Hz, stereo, uncompressed PCM16 WAV')
        reference=None
        if reference_json is not None:
            reference=json.loads(Path(reference_json).read_text(encoding='utf-8'))
            if reference['model_family']!=MODEL_FAMILY or reference['model_status']!=engine.model_status:
                raise ValueError('Reference model family or training status does not match')
            if Path(reference['wav_file']).resolve()!=Path(wav_path).resolve():
                raise ValueError('Reference WAV does not match')
        # Initialize CPU kernels before the timed source replay begins.
        engine.predict_window(np.zeros((engine.config.sequence_length,2),np.float32),include_reconstruction=False)
        streamer=StreamingPredictor(engine)
        received=0; emitted=0; chunks=0
        evaluated_frames=0; mse_total=0.; channel_mse_totals=[0.,0.]
        push_ms=[]; latency_ms=[]; source_lag_ms=[]
        max_score_delta=0.; max_channel_delta=0.; max_latent_delta=0.
        output_path.parent.mkdir(parents=True,exist_ok=True)
        started=time.perf_counter()
        with output_path.open('w',encoding='utf-8') as writer:
            while True:
                payload=source.readframes(chunk_frames)
                if not payload:break
                pcm=np.frombuffer(payload,dtype='<i2').reshape(-1,2)
                received+=len(pcm);chunks+=1
                due_s=received/SAMPLE_RATE
                if realtime:
                    wait=started+due_s-time.perf_counter()
                    if wait>0:time.sleep(wait)
                    source_lag_ms.append(max(0.,(time.perf_counter()-started-due_s)*1000))
                before=time.perf_counter()
                results=streamer.push(pcm,SAMPLE_RATE)
                finished=time.perf_counter()
                if results:push_ms.append((finished-before)*1000)
                for result in results:
                    if reference is not None:
                        if emitted>=len(reference['windows']):
                            raise ValueError('Stream emits more windows than reference')
                        ref=reference['windows'][emitted]
                        if (result['start_frame'],result['end_frame'])!=(ref['start_frame'],ref['end_frame']):
                            raise ValueError('Stream window boundaries differ from offline reference')
                        for key in ['reconstruction_mse_standardized','reconstruction_mse_by_channel_standardized','latent_mu']:
                            if not np.allclose(result[key],ref[key],rtol=1e-5,atol=1e-5):
                                raise ValueError(f'Stream/offline mismatch in {key}')
                        max_score_delta=max(max_score_delta,abs(result['reconstruction_mse_standardized']-ref['reconstruction_mse_standardized']))
                        max_channel_delta=max(max_channel_delta,float(np.max(np.abs(np.asarray(result['reconstruction_mse_by_channel_standardized'])-ref['reconstruction_mse_by_channel_standardized']))))
                        max_latent_delta=max(max_latent_delta,float(np.max(np.abs(np.asarray(result['latent_mu'])-ref['latent_mu']))))
                    row={**result,'model_family':MODEL_FAMILY,'sequence_number':emitted,
                         'source_available_through_s':due_s,
                         'emitted_after_replay_start_s':finished-started,
                         'latency_after_window_end_ms':(finished-started-result['end_s'])*1000 if realtime else None}
                    if realtime:latency_ms.append(row['latency_after_window_end_ms'])
                    writer.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
                    writer.flush()
                    if on_result is not None:
                        on_result(row)
                    evaluated_frames+=row['end_frame']-row['start_frame']
                    mse_total+=row['reconstruction_mse_standardized']
                    for channel in range(2):
                        channel_mse_totals[channel]+=row['reconstruction_mse_by_channel_standardized'][channel]
                    emitted+=1
        wall_s=time.perf_counter()-started
        if received!=source.getnframes():raise ValueError('Read frame count differs from WAV header')
        if received!=streamer.processed_frames+streamer.pending_frames:
            raise AssertionError('Frame accounting mismatch')
        if reference is not None:
            if emitted!=reference['processed_windows'] or streamer.pending_frames!=reference['trailing_partial_frames']:
                raise ValueError('Window or tail count differs from offline reference')
        return {
            'model_family':MODEL_FAMILY,'model_status':engine.model_status,'device':str(engine.device),
            'validated_for_fault_detection':False,'mode':'realtime_wav_replay' if realtime else 'fast_chunk_replay',
            'source_file':str(Path(wav_path).resolve()),'sample_rate_hz':SAMPLE_RATE,
            'output_file':str(output_path.resolve()),
            'chunk_frames':chunk_frames,'chunk_duration_ms':chunk_frames/SAMPLE_RATE*1000,
            'window_frames':engine.config.sequence_length,'window_duration_ms':engine.config.sequence_length/SAMPLE_RATE*1000,
            'received_chunks':chunks,'received_frames':received,'signal_duration_s':received/SAMPLE_RATE,
            'emitted_windows':emitted,'processed_frames':streamer.processed_frames,
            'pending_frames':streamer.pending_frames,'replay_wall_time_s':wall_s,
            'emit_push_median_ms':float(np.median(push_ms)) if push_ms else None,
            'emit_push_p95_ms':float(np.quantile(push_ms,.95)) if push_ms else None,
            'window_end_to_output_median_ms':float(np.median(latency_ms)) if latency_ms else None,
            'window_end_to_output_p95_ms':float(np.quantile(latency_ms,.95)) if latency_ms else None,
            'window_end_to_output_max_ms':float(max(latency_ms)) if latency_ms else None,
            'max_source_schedule_lag_ms':float(max(source_lag_ms)) if source_lag_ms else None,
            'final_source_schedule_lag_ms':float(source_lag_ms[-1]) if source_lag_ms else None,
            'offline_reference_verified':reference is not None,
            'max_offline_score_delta':max_score_delta if reference is not None else None,
            'max_offline_channel_score_delta':max_channel_delta if reference is not None else None,
            'max_offline_latent_delta':max_latent_delta if reference is not None else None,
            'summary':{
                'evaluated_duration_s':evaluated_frames/SAMPLE_RATE,
                'mean_reconstruction_mse_standardized':mse_total/emitted if emitted else None,
                'mean_reconstruction_mse_by_channel_standardized':[
                    total/emitted if emitted else None for total in channel_mse_totals
                ],
            },
            'timing_scope':f'Warm {engine.device} replay. Window-end latency includes chunk arrival granularity and scheduling, ends when Python results are ready; excludes JSON flush and external transport.',
            'input_contract':'Already synchronized PCM16 sound/vibration channels at 44100 Hz; no independent-device alignment.',
            'state_contract':'Nonoverlapping windows; LSTM hidden state restarts per window. Partial final samples remain buffered until more data arrives.',
        }
    finally:
        source.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--wav',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('artifacts/stream_predictions.jsonl'))
    p.add_argument('--report',type=Path,default=Path('artifacts/stream_report.json'))
    p.add_argument('--chunk-ms',type=float,default=20.)
    p.add_argument('--realtime',action='store_true')
    p.add_argument('--reference-json',type=Path)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    a=p.parse_args();torch.set_num_threads(a.threads)
    device_options = {} if a.device == 'auto' else {'device': a.device}
    engine=VAEEngine.from_checkpoint(a.checkpoint, **device_options)
    print('Checkpoint loaded successfully.',flush=True)
    print(f'Device: {engine.device}',flush=True)
    print('-'*64,flush=True)
    print('Streaming results (standardized reconstruction MSE):',flush=True)
    next_signal_second=0.

    def display_result(row):
        nonlocal next_signal_second
        if row['sequence_number']==0 or row['end_s']>=next_signal_second:
            left,right=row['reconstruction_mse_by_channel_standardized']
            print(f"Time: {row['start_s']:.3f}-{row['end_s']:.3f} s | "
                  f"MSE: {row['reconstruction_mse_standardized']:.6f} | "
                  f"L: {left:.6f} | R: {right:.6f}",flush=True)
            next_signal_second=float(int(row['end_s'])+1)

    report=replay_stream(engine,a.wav,a.output,a.chunk_ms,a.realtime,a.reference_json,
                         on_result=display_result)
    report['checkpoint_file']=str(a.checkpoint.resolve())
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print_prediction_summary(report['summary'])


if __name__=='__main__':main()
