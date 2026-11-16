import sys
import wave
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from Dataset.Preprocessing.prepare_msm import aligned_windows,recording_split,select_windows,bandlimited_align
from scipy.signal import resample_poly


def test_two_sensor_sample_rates_and_time_offset_align(tmp_path):
    sound_fs=48000;acc_fs=51200
    sound_t=np.arange(sound_fs)/sound_fs
    acc_t=18.75e-6+np.arange(acc_fs)/acc_fs
    source=tmp_path/'s0.wav'
    sound=np.round(10000*np.sin(2*np.pi*200*sound_t)).astype('<i2')
    with wave.open(str(source),'wb') as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(sound_fs);w.writeframes(sound.tobytes())
    acc=tmp_path/'acc.csv'
    pd.DataFrame({'time':acc_t,'acc0':np.sin(2*np.pi*200*acc_t),
                  'acc1':np.cos(2*np.pi*200*acc_t)}).to_csv(acc,index=False)
    request={'recording_id':'dataset1','interval_id':'dataset1:0','label':1,
             'label_start_s':.2,'label_end_s':.4,'window_in_interval':0,
             'start_frame_48k':12000,'start_s':.25,'end_s':.25+2048/sound_fs}
    x,rows,bad,inspection=aligned_windows(tmp_path,source,acc,[request],2048)
    truth=np.sin(2*np.pi*200*(12000+np.arange(2048))/sound_fs)
    assert x.shape==(1,2048,2) and len(rows)==1 and not bad
    np.testing.assert_allclose(x[0,:,0]/10000,truth,atol=6e-5)
    np.testing.assert_allclose(x[0,:,1],truth,atol=.002)
    assert np.isclose(inspection['acc_start_s'],18.75e-6)


def test_windows_stay_inside_their_label_and_do_not_overlap():
    intervals=[{'recording_id':'dataset1','interval_id':'dataset1:0','label':0,
                'label_start_s':1.,'label_end_s':2.},
               {'recording_id':'dataset1','interval_id':'dataset1:1','label':1,
                'label_start_s':2.1,'label_end_s':2.11}]
    windows,excluded=select_windows(intervals,2048,8)
    assert len(windows)==8 and len(excluded)==1
    assert all(r['start_s']>=1.02 and r['end_s']<=1.98 for r in windows)
    assert all(a['end_s']<=b['start_s'] for a,b in zip(windows,windows[1:]))


def test_recording_split_has_no_group_leakage_and_all_classes():
    intervals=[{'recording_id':f'dataset{i}','interval_id':f'dataset{i}:0',
                'label':i%3} for i in range(1,38)]
    split=recording_split(intervals)
    sets=[set(split[s]) for s in ['train','val','test']]
    assert [len(s) for s in sets]==[23,7,7]
    assert len(set.union(*sets))==37
    assert all(not a&b for a,b in [(sets[0],sets[1]),(sets[0],sets[2]),(sets[1],sets[2])])
    for s in sets:assert {r['label'] for r in intervals if r['recording_id'] in s}=={0,1,2}


def test_timestamp_alignment_preserves_high_frequency_amplitude():
    for origin in [18.75e-6,18.75e-6+1/51200,18.75e-6+13/51200]:
        native_time=origin+np.arange(51200)/51200
        target_time=.25+np.arange(2048)/48000
        for frequency in [1000,10000,20000]:
            native=np.sin(2*np.pi*frequency*native_time)
            converted=resample_poly(native,15,16,window=('kaiser',5.0))
            output=bandlimited_align(converted,origin,.25,2048)
            truth=np.sin(2*np.pi*frequency*target_time)
            np.testing.assert_allclose(output,truth,atol=.005)
