"""Frozen MSM classifier probe on AT's 16 recordings with UNVERIFIED units.

This is an input-compatibility diagnostic, not a scored external evaluation.
The right WAV channel is fed as its original numeric PCM values; no conversion
to the acceleration unit expected by MSM is claimed or invented.
The original 16 IDs are read from the model's saved artifacts/data_manifest.json;
this historical diagnostic does not follow the editable training split CSV.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly


def historical_pilot_ids(model_dir):
    manifest_path = Path(model_dir) / 'artifacts/data_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    if not isinstance(manifest, list):
        raise ValueError(f'{manifest_path}: expected the original 16-recording manifest')
    ids = [row.get('recording_id') for row in manifest]
    if len(ids) != 16 or any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != 16:
        raise ValueError(f'{manifest_path}: expected 16 distinct original recording IDs')
    return sorted(ids)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--source',type=Path,default=Path(__file__).resolve().parent.parent/'Dataset'/'Data')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--threads',type=int,default=2)
    a=p.parse_args();torch.set_num_threads(a.threads)
    sys.path.insert(0,str(a.model_dir.resolve()))
    from classifier import StateClassifier
    from model import ModelConfig
    checkpoint=a.model_dir/'artifacts/msm_classification/best_classifier.pt'
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    config=ModelConfig(**state['model_config']);model=StateClassifier(config)
    model.load_state_dict(state['state_dict']);model.eval()
    mean=np.asarray(state['normalizer']['mean'],np.float32)
    scale=np.asarray(state['normalizer']['scale'],np.float32)
    fs=state['preprocessing']['target_sample_rate_hz'];length=config.sequence_length
    if fs!=48000 or config.input_channels!=2:raise ValueError('Probe is defined for the current 48 kHz two-channel MSM model')
    files=[a.source/recording/'audio_000000.wav' for recording in historical_pilot_ids(a.model_dir)]
    missing=[str(path) for path in files if not path.is_file()]
    if missing:raise FileNotFoundError('Missing original pilot recordings: '+', '.join(missing))
    a.output.mkdir(parents=True,exist_ok=True)
    class_names=state['class_names']
    with torch.inference_mode():
        constant_scores=model(torch.zeros(1,length,2),reconstruct=False)['logits'].softmax(-1)[0].numpy()
        numeric_zero=(np.zeros((1,length,2),np.float32)-mean)/scale
        zero_scores=model(torch.from_numpy(numeric_zero),reconstruct=False)['logits'].softmax(-1)[0].numpy()
    rows=[];started=time.perf_counter()
    with (a.output/'window_predictions.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=['recording_id','context_group','window_index','start_s','end_s',
            'predicted_label','score_class_0','score_class_1','score_class_2','input_mapping_verified'])
        writer.writeheader()
        for path in files:
            record=path.parent.name
            metadata=json.loads((path.parent/'metadata.json').read_text(encoding='utf-8-sig'))
            operation=metadata['Context']['Operation']
            group='cutting_recording_weak_context' if operation in ['Acabamento','Desbaste'] else 'background_or_noncutting'
            with wave.open(str(path),'rb') as w:
                source_fs=w.getframerate();source_frames=w.getnframes()
                if (source_fs,w.getnchannels(),w.getsampwidth(),w.getcomptype())!=(44100,2,2,'NONE'):
                    raise ValueError(f'Unexpected source format: {record}')
                pcm=np.frombuffer(w.readframes(source_frames),dtype='<i2').reshape(-1,2).astype(np.float32)
            resampled=resample_poly(pcm,up=160,down=147,axis=0,window=('kaiser',5.)).astype(np.float32)
            full=len(resampled)//length;tail=len(resampled)%length
            if full==0:raise ValueError('No full windows')
            values=resampled[:full*length].reshape(full,length,2)
            scores=[]
            for first in range(0,full,a.batch_size):
                raw=values[first:first+a.batch_size]
                # Deliberately named numeric probe: channel R is NOT calibrated Acc0.
                normalized=(raw-mean)/scale
                with torch.inference_mode():
                    logits=model(torch.from_numpy(np.ascontiguousarray(normalized)),reconstruct=False)['logits']
                    probability=logits.softmax(-1).numpy()
                if not np.isfinite(probability).all():raise RuntimeError('Non-finite model output')
                scores.append(probability)
                for j,prob in enumerate(probability):
                    i=first+j
                    writer.writerow({'recording_id':record,'context_group':group,'window_index':i,
                        'start_s':i*length/fs,'end_s':(i+1)*length/fs,'predicted_label':int(prob.argmax()),
                        'score_class_0':float(prob[0]),'score_class_1':float(prob[1]),'score_class_2':float(prob[2]),
                        'input_mapping_verified':False})
            score=np.concatenate(scores);pred=score.argmax(axis=1)
            # Distribution statistics include all resampled frames, including the partial tail.
            numeric_mean=resampled.mean(axis=0,dtype=np.float64)
            numeric_std=resampled.std(axis=0,dtype=np.float64)
            z=(resampled-mean)/scale
            quantiles=np.quantile(z,[.01,.5,.99],axis=0)
            counts=np.bincount(pred,minlength=3)
            row={'recording_id':record,'operation_pt':operation,'notes_pt':metadata['Context']['Notes'],
                'context_group':group,'source_sample_rate_hz':source_fs,'source_frames':source_frames,
                'source_duration_s':source_frames/source_fs,'resampled_frames':len(resampled),
                'processed_windows':full,'processed_duration_s':full*length/fs,'trailing_frames_48k':tail,
                'window_class_counts':counts.tolist(),'window_class_fractions':(counts/full).tolist(),
                'mean_class_scores':score.mean(axis=0,dtype=np.float64).tolist(),
                'mean_score_argmax':int(score.mean(axis=0).argmax()),'dominant_window_label':int(counts.argmax()),
                'numeric_channel_mean_pcm':numeric_mean.tolist(),'numeric_channel_std_pcm':numeric_std.tolist(),
                'frozen_z_mean':z.mean(axis=0,dtype=np.float64).tolist(),
                'frozen_z_std':z.std(axis=0,dtype=np.float64).tolist(),
                'frozen_z_rms':np.sqrt(np.mean(z.astype(np.float64)**2,axis=0)).tolist(),
                'frozen_z_quantiles_01_50_99_by_channel':quantiles.tolist(),
                'sound_std_relative_to_msm_training_sound_std':float(numeric_std[0]/scale[0]),
                'ground_truth_msm_class':None,'input_mapping_verified':False}
            rows.append(row)
            print(json.dumps({'model':state['model_family'],'recording':record,'windows':full,
                              'class_counts':counts.tolist(),'sound_std_ratio':row['sound_std_relative_to_msm_training_sound_std']},ensure_ascii=False),flush=True)
    totals={}
    for group in ['cutting_recording_weak_context','background_or_noncutting']:
        selected=[r for r in rows if r['context_group']==group]
        counts=np.sum([r['window_class_counts'] for r in selected],axis=0)
        totals[group]={'recordings':len(selected),'window_count':int(counts.sum()),
                       'window_class_counts':counts.tolist(),'window_class_fractions':(counts/counts.sum()).tolist()}
    report={'model_family':state['model_family'],'checkpoint':str(checkpoint),
        'pilot_recording_manifest':str((a.model_dir/'artifacts/data_manifest.json').resolve()),
        'evaluation_scope':'frozen_model_numeric_input_compatibility_probe',
        'network_retrained':False,'normalizer_refitted':False,
        'input_mapping_verified':False,'formal_external_classification_metrics_available':False,
        'source_units':['left PCM16 counts','right PCM16 counts, sensor/calibration unconfirmed'],
        'model_expected_units':state['preprocessing']['channels'],
        'calibration_conversion_applied':False,
        'right_channel_treatment':'Original numeric PCM values after resampling, used only for an explicitly unverified diagnostic; not asserted to be mm/s^2.',
        'label_scope':'Professor described normal work at recording level; no exact cutting boundaries or abnormal/defective examples. No per-window MSM class is fabricated.',
        'resampling':'Both existing WAV channels jointly resampled 44100 to 48000 Hz (160/147), retaining their relative sample alignment; no gain adaptation.',
        'normalizer':state['normalizer'],'class_names':class_names,
        'constant_training_mean_input_control':{'description':'A synthetic constant window equal to MSM training-channel means, normalized to zero; not a real recording.',
            'class_scores':constant_scores.tolist(),'predicted_label':int(constant_scores.argmax())},
        'constant_raw_numeric_zero_control':{'description':'Synthetic raw numeric zeros in both input columns, then the frozen MSM normalization; not a real recording or a health label.',
            'class_scores':zero_scores.tolist(),'predicted_label':int(zero_scores.argmax())},
        'recordings':rows,'group_totals':totals,'elapsed_s':time.perf_counter()-started}
    (a.output/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'complete':True,'model':state['model_family'],'recordings':len(rows),
                      'windows':sum(r['processed_windows'] for r in rows),'output':str(a.output)},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
