"""Build timestamp-aligned Sound0/Acc0 windows from the MSM imi_vm20i subset.

Raw data remains untouched. Sound0 keeps its native 48 kHz PCM scale. Acc0 is
anti-alias resampled from 51.2 to 48 kHz, then aligned using its timestamps.
"""
from __future__ import annotations

import argparse
import json
import math
import time
import wave
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly


TARGET_FS=48000
ACC_FS=51200


def save_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def list_intervals(root):
    records=sorted((p for p in root.glob('dataset*') if p.is_dir()),key=lambda p:int(p.name.removeprefix('dataset')))
    if len(records)!=37:raise ValueError(f'Expected 37 experiment directories, found {len(records)}')
    intervals=[]
    for record in records:
        labels=pd.read_csv(record/'label.csv',encoding='utf-8-sig')
        labels.columns=labels.columns.str.strip()
        for row_number,row in labels.iterrows():
            start,end,label=float(row['start']),float(row['end']),int(row['label'])
            if not np.isfinite([start,end]).all() or start<0 or end<=start or label not in [0,1,2]:
                raise ValueError(f'Invalid annotation: {record.name}, row {row_number}')
            intervals.append({'recording_id':record.name,'interval_id':f'{record.name}:{row_number}',
                              'label_start_s':start,'label_end_s':end,'label':label})
    return records,intervals


def recording_split(intervals):
    groups=sorted({r['recording_id'] for r in intervals},key=lambda k:int(k.removeprefix('dataset')))
    counts=np.array([[sum(r['recording_id']==g and r['label']==label for r in intervals)
                      for label in range(3)] for g in groups])
    total=counts.sum(axis=0);target=np.array([.6,.2,.2])
    rng=np.random.default_rng(20261105);best=None
    # Class 2 occurs in only four recordings. Balance annotation counts under
    # a whole-recording constraint, before looking at any signal/model score.
    for _ in range(5000):
        permutation=rng.permutation(len(groups))
        index_sets=[permutation[14:],permutation[7:14],permutation[:7]]
        split_counts=np.array([counts[ix].sum(axis=0) for ix in index_sets])
        if np.any(split_counts==0):continue
        cost=float(np.mean(np.abs(split_counts/total-target[:,None]))+
                   .2*np.mean(np.abs(split_counts.sum(axis=1)/total.sum()-target)))
        if best is None or cost<best[0]:best=(cost,index_sets)
    if best is None:raise ValueError('Cannot construct recording-disjoint splits containing all classes')
    return {s:sorted([groups[i] for i in ix],key=lambda k:int(k.removeprefix('dataset')))
            for s,ix in zip(['train','val','test'],best[1])}


def select_windows(intervals,sequence_length,windows_per_interval,edge_trim_s=.02):
    result=[];excluded=[]
    for interval in intervals:
        first=math.ceil((interval['label_start_s']+edge_trim_s)*TARGET_FS)
        last=math.floor((interval['label_end_s']-edge_trim_s)*TARGET_FS)
        count=(last-first)//sequence_length
        if count<1:
            excluded.append({**interval,'reason':'too_short_for_one_window_after_20ms_edge_trim'});continue
        chosen=np.linspace(0,count-1,min(count,windows_per_interval),dtype=np.int64)
        for i,offset in enumerate(chosen):
            start=first+int(offset)*sequence_length
            result.append({**interval,'window_in_interval':i,'start_frame_48k':start,
                           'start_s':start/TARGET_FS,'end_s':(start+sequence_length)/TARGET_FS})
    return result,excluded


def wait_one(record,suffix,wait):
    announced=False
    while True:
        matches=list(record.glob(f'*_{suffix}'))
        if len(matches)==1:return matches[0]
        if len(matches)>1:raise ValueError(f'Multiple {suffix} files in {record.name}')
        if not wait:raise FileNotFoundError(f'{record.name}: {suffix} has not downloaded')
        if not announced:
            print(f'Waiting for {record.name}/{suffix}',flush=True);announced=True
        time.sleep(10)


def bandlimited_align(samples,origin_s,target_start_s,frames,fs=TARGET_FS):
    """Shift a uniform sequence to the requested time grid without linear-interpolation attenuation."""
    position=(target_start_s-origin_s)*fs
    center=math.floor(position);fraction=position-center
    half=64
    offsets=np.arange(-half,half+1)
    weights=np.sinc(fraction-offsets)*np.kaiser(2*half+1,8.6)
    weights/=weights.sum()
    first=center-half
    if first<0 or first+frames>len(samples)-2*half:
        raise ValueError('Insufficient context for bandlimited timestamp alignment')
    aligned=np.correlate(samples,weights,mode='valid')
    return aligned[first:first+frames].astype(np.float32)


def aligned_windows(record,wav_path,acc_path,requests,sequence_length):
    order=np.argsort([r['start_s'] for r in requests],kind='stable')
    ordered=[requests[int(i)] for i in order]
    margin=.01
    starts=np.array([r['start_s']-margin for r in ordered])
    ends=np.array([r['end_s']+margin for r in ordered])
    pieces=[[] for _ in ordered];first_times=[None]*len(ordered)
    cursor=0;active=[];rows=0;last_time=None;first_time=None;initial_dt=None
    for chunk in pd.read_csv(acc_path,usecols=['time','acc0'],dtype={'time':np.float64,'acc0':np.float32},chunksize=400000):
        t=chunk['time'].to_numpy();v=chunk['acc0'].to_numpy()
        if len(t)==0:continue
        if not np.isfinite(t).all() or not np.isfinite(v).all():raise ValueError(f'Non-finite accelerometer data: {record.name}')
        delta=np.diff(t)
        if np.any(delta<=0) or (last_time is not None and t[0]<=last_time):raise ValueError(f'Nonmonotonic accelerometer time: {record.name}')
        if first_time is None:
            first_time=float(t[0]);initial_dt=float(np.median(delta[:2000]))
            if not np.isclose(initial_dt,1/ACC_FS,rtol=1e-4):raise ValueError(f'Unexpected accelerometer sampling rate: {record.name}')
        if np.any(np.abs(delta-1/ACC_FS)>1e-7):raise ValueError(f'Accelerometer sampling gap: {record.name}')
        if last_time is not None and abs(t[0]-last_time-1/ACC_FS)>1e-7:raise ValueError(f'Accelerometer chunk discontinuity: {record.name}')
        while cursor<len(ordered) and starts[cursor]<=t[-1]:
            active.append(cursor);cursor+=1
        still=[]
        for i in active:
            lo=int(np.searchsorted(t,starts[i],side='left'));hi=int(np.searchsorted(t,ends[i],side='right'))
            if hi>lo:
                if first_times[i] is None:first_times[i]=float(t[lo])
                pieces[i].append(v[lo:hi].copy())
            if ends[i]>t[-1]:still.append(i)
        active=still;rows+=len(t);last_time=float(t[-1])
    signals=[];retained=[];excluded=[]
    with wave.open(str(wav_path),'rb') as w:
        fs=w.getframerate();frames=w.getnframes()
        if (fs,w.getnchannels(),w.getsampwidth(),w.getcomptype())!=(TARGET_FS,1,2,'NONE'):
            raise ValueError(f'Unexpected Sound0 WAV format: {record.name}')
        for i,metadata in enumerate(ordered):
            native=np.concatenate(pieces[i]) if pieces[i] else np.empty(0,np.float32)
            start_frame=metadata['start_frame_48k']
            if len(native)==0 or start_frame+sequence_length>frames:
                excluded.append({**metadata,'reason':'sensor_coverage_missing'});continue
            converted=resample_poly(native,up=15,down=16,window=('kaiser',5.0)).astype(np.float32)
            converted_times=first_times[i]+np.arange(len(converted))/TARGET_FS
            target_times=(start_frame+np.arange(sequence_length))/TARGET_FS
            if target_times[0]<converted_times[0] or target_times[-1]>converted_times[-1]:
                excluded.append({**metadata,'reason':'insufficient_accelerometer_context'});continue
            vibration=bandlimited_align(converted,first_times[i],metadata['start_s'],sequence_length)
            w.setpos(start_frame)
            sound=np.frombuffer(w.readframes(sequence_length),dtype='<i2').astype(np.float32)
            if len(sound)!=sequence_length:raise ValueError('Incomplete audio read')
            signals.append(np.stack([sound,vibration],axis=-1));retained.append(metadata)
    inspected={'recording_id':record.name,'sound_file':str(wav_path),'acc_file':str(acc_path),
               'audio_sample_rate_hz':fs,'audio_frames':frames,'audio_duration_s':frames/fs,
               'acc_sample_rate_hz':1/initial_dt,'acc_rows':rows,'acc_start_s':first_time,'acc_end_s':last_time,
               'retained_windows':len(retained),'excluded_windows':len(excluded)}
    return np.stack(signals),retained,excluded,inspected


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--raw',type=Path,required=True,
                   help='Directory containing the 37 original MSM imi_vm20i experiments')
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent.parent/'Data'/'msm_prepared')
    p.add_argument('--sequence-length',type=int,default=2048)
    p.add_argument('--windows-per-interval',type=int,default=8)
    p.add_argument('--wait-for-files',action='store_true')
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);cache=a.output/'recordings';cache.mkdir(exist_ok=True)
    records,intervals=list_intervals(a.raw)
    splits=recording_split(intervals)
    windows,excluded=select_windows(intervals,a.sequence_length,a.windows_per_interval)
    save_json(a.output/'splits.json',splits)
    save_json(a.output/'intervals.json',intervals)
    config={'target_sample_rate_hz':TARGET_FS,'sequence_length':a.sequence_length,
        'window_duration_s':a.sequence_length/TARGET_FS,'windows_per_interval_limit':a.windows_per_interval,
        'channels':['Sound0 microphone (PCM16 counts)','Acc0 spindle acceleration (mm/s^2)'],
        'audio_source_rate_hz':48000,'acc_source_rate_hz':51200,
        'resampling':'Sound0 unchanged; Acc0 scipy.resample_poly 15/16 Kaiser beta=5, then 129-tap Kaiser(beta=8.6)-windowed sinc fractional-delay timestamp alignment',
        'filter_context_seconds_each_side':.01,'label_edge_trim_seconds_each_side':.02,
        'sampling':'uniform selection from nonoverlapping full windows entirely inside each labelled cutting interval',
        'split_method':'23/7/7 whole-recording split; minimum annotation-count imbalance across 5000 seeded permutations (20261105), requiring all classes in all splits; no signal or model outcomes used',
        'label_definition':{'0':'normal cutting','1':'abnormal cutting requiring intervention','2':'defective tool with otherwise normal cutting'},
        'normalization':'training-window channel mean/std fitted later by trainer',
        'source':'https://www.kaggle.com/datasets/manufuturetoday/multi-sensor-for-metal-milling-anomaly',
        'source_version':2,'data_license':'CC BY 4.0'}
    old_config=a.output/'preprocessing.json'
    if old_config.exists() and json.loads(old_config.read_text())!=config:
        raise ValueError('Existing cache uses a different preprocessing configuration; use a new output directory')
    save_json(old_config,config)
    all_signals=[];all_metadata=[];inspections=[]
    started=time.perf_counter()
    for k,record in enumerate(records,1):
        prefix=cache/record.name
        if prefix.with_suffix('.npy').exists() and prefix.with_suffix('.json').exists():
            info=json.loads(prefix.with_suffix('.json').read_text());signal=np.load(prefix.with_suffix('.npy'))
            retained=info['windows'];bad=info['excluded'];inspection=info['inspection']
        else:
            wav=wait_one(record,'s0.wav',a.wait_for_files);acc=wait_one(record,'acc.csv',a.wait_for_files)
            selected=[r for r in windows if r['recording_id']==record.name]
            signal,retained,bad,inspection=aligned_windows(record,wav,acc,selected,a.sequence_length)
            np.save(prefix.with_suffix('.npy'),signal)
            save_json(prefix.with_suffix('.json'),{'windows':retained,'excluded':bad,'inspection':inspection})
        all_signals.append(signal);all_metadata.extend(retained);excluded.extend(bad);inspections.append(inspection)
        progress={'complete_recordings':k,'total_recordings':37,'prepared_windows':len(all_metadata),'elapsed_s':time.perf_counter()-started}
        save_json(a.output/'progress.json',progress)
        print(json.dumps({**progress,'recording_id':record.name}),flush=True)
    x=np.concatenate(all_signals);y=np.array([r['label'] for r in all_metadata],np.int64)
    if not np.isfinite(x).all():raise ValueError('Prepared data has non-finite values')
    np.save(a.output/'windows.npy',x);np.save(a.output/'labels.npy',y)
    save_json(a.output/'windows.json',all_metadata)
    save_json(a.output/'excluded.json',excluded)
    save_json(a.output/'source_inspection.json',inspections)
    summary={'recording_count':37,'labelled_intervals':len(intervals),'retained_intervals':len({r['interval_id'] for r in all_metadata}),
        'windows':len(x),'window_class_counts':np.bincount(y,minlength=3).tolist(),'excluded_count':len(excluded),
        'shape':list(x.shape),'actual_audio_duration_seconds':sum(r['audio_duration_s'] for r in inspections),
        'split_recordings':{s:len(v) for s,v in splits.items()},
        'split_window_class_counts':{s:np.bincount([r['label'] for r in all_metadata if r['recording_id'] in ids],minlength=3).tolist() for s,ids in splits.items()},
        'elapsed_seconds':time.perf_counter()-started}
    save_json(a.output/'summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
