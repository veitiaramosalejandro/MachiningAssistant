import json
import sys
import wave

import numpy as np
import pytest
import torch

from model import ModelConfig
from runtime import VAEEngine,predict_wav
from stream_wav import main, replay_stream
from prediction_summary import summarize_prediction


def test_chunked_wav_replay_matches_offline_and_writes_jsonl(tmp_path):
    torch.set_num_threads(2);torch.manual_seed(10)
    engine=VAEEngine(ModelConfig(sequence_length=64))
    pcm=np.random.default_rng(7).integers(-200,201,(64*3+11,2),dtype=np.int16)
    source=tmp_path/'input.wav'
    with wave.open(str(source),'wb') as w:
        w.setnchannels(2);w.setsampwidth(2);w.setframerate(44100);w.writeframes(pcm.astype('<i2').tobytes())
    reference=tmp_path/'reference.json'
    reference.write_text(json.dumps(predict_wav(engine,source,max_windows=0)))
    output=tmp_path/'output.jsonl'
    report=replay_stream(engine,source,output,chunk_ms=.5,reference_json=reference)
    rows=[json.loads(line) for line in output.read_text().splitlines()]
    assert report['received_frames']==203 and report['emitted_windows']==3 and report['pending_frames']==11
    assert [r['sequence_number'] for r in rows]==[0,1,2]
    assert [r['start_frame'] for r in rows]==[0,64,128]
    assert all(r['source_available_through_s']>=r['end_s'] for r in rows)
    assert report['offline_reference_verified'] and report['max_offline_score_delta']<1e-5
    assert report['window_end_to_output_median_ms'] is None
    expected=summarize_prediction(json.loads(reference.read_text()))
    assert report['summary']['evaluated_duration_s']==expected['evaluated_duration_s']
    np.testing.assert_allclose(report['summary']['mean_reconstruction_mse_standardized'],
                               expected['mean_reconstruction_mse_standardized'],rtol=1e-5,atol=1e-7)
    np.testing.assert_allclose(report['summary']['mean_reconstruction_mse_by_channel_standardized'],
                               expected['mean_reconstruction_mse_by_channel_standardized'],rtol=1e-5,atol=1e-7)


def test_callback_arrives_before_eof_and_after_jsonl_flush(tmp_path,monkeypatch):
    torch.set_num_threads(2);torch.manual_seed(12)
    engine=VAEEngine(ModelConfig(sequence_length=64))
    source=tmp_path/'input.wav'
    pcm=np.arange((64*3+11)*2,dtype=np.int16).reshape(-1,2)
    with wave.open(str(source),'wb') as w:
        w.setnchannels(2);w.setsampwidth(2);w.setframerate(44100);w.writeframes(pcm.astype('<i2').tobytes())
    original_read=wave.Wave_read.readframes
    state={'eof_read':False,'frames_read':0}

    def tracked_read(reader,count):
        payload=original_read(reader,count)
        if not payload:state['eof_read']=True
        state['frames_read']+=len(payload)//4
        return payload

    monkeypatch.setattr(wave.Wave_read,'readframes',tracked_read)
    output=tmp_path/'output.jsonl'
    callbacks=[]

    def on_result(row):
        assert not state['eof_read']
        written=[json.loads(line) for line in output.read_text().splitlines()]
        assert len(written)==row['sequence_number']+1
        assert written[-1]==row
        if not callbacks:assert state['frames_read']<len(pcm)
        callbacks.append(row['sequence_number'])

    report=replay_stream(engine,source,output,chunk_ms=.5,on_result=on_result)
    assert callbacks==[0,1,2] and state['eof_read']
    assert report['processed_frames']+report['pending_frames']==report['received_frames']==203
    assert report['pending_frames']==11


def test_short_input_has_empty_jsonl_and_unavailable_means(tmp_path):
    torch.set_num_threads(2)
    engine=VAEEngine(ModelConfig(sequence_length=64))
    source=tmp_path/'short.wav'
    with wave.open(str(source),'wb') as w:
        w.setnchannels(2);w.setsampwidth(2);w.setframerate(44100);w.writeframes(bytes(63*4))
    callbacks=[];output=tmp_path/'output.jsonl'
    report=replay_stream(engine,source,output,chunk_ms=.5,on_result=callbacks.append)
    assert callbacks==[] and output.read_text()==''
    assert report['emitted_windows']==0 and report['pending_frames']==63
    assert report['summary']=={
        'evaluated_duration_s':0.,
        'mean_reconstruction_mse_standardized':None,
        'mean_reconstruction_mse_by_channel_standardized':[None,None],
    }


def test_cli_loads_once_and_prints_live_results_then_summary(tmp_path,monkeypatch,capsys):
    torch.set_num_threads(2)
    engine=VAEEngine(ModelConfig(sequence_length=64))
    checkpoint=tmp_path/'model.pt';engine.save(checkpoint)
    source=tmp_path/'input.wav'
    with wave.open(str(source),'wb') as w:
        w.setnchannels(2);w.setsampwidth(2);w.setframerate(44100);w.writeframes(bytes(203*4))
    output=tmp_path/'output.jsonl';report_file=tmp_path/'report.json'
    original_loader=VAEEngine.from_checkpoint
    loads=[]

    def load_once(path):
        loads.append(path)
        return original_loader(path)

    monkeypatch.setattr(VAEEngine,'from_checkpoint',staticmethod(load_once))
    monkeypatch.setattr(sys,'argv',['stream_wav.py','--checkpoint',str(checkpoint),
                                  '--wav',str(source),'--output',str(output),
                                  '--report',str(report_file),'--chunk-ms','0.5'])
    main()
    console=capsys.readouterr().out
    report=json.loads(report_file.read_text())
    assert len(loads)==1
    assert console.count('Checkpoint loaded successfully.')==1
    assert 'Streaming results' in console and 'Time:' in console
    assert 'Prediction results' in console and 'Mean MSE:' in console
    assert 'Processed ' not in console and str(checkpoint) not in console
    assert report['checkpoint_file']==str(checkpoint.resolve())
    assert report['output_file']==str(output.resolve())


def test_stream_reader_rejects_incompatible_wave_format(tmp_path):
    torch.set_num_threads(2)
    engine=VAEEngine(ModelConfig(sequence_length=64))
    source=tmp_path/'mono.wav'
    with wave.open(str(source),'wb') as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(44100);w.writeframes(bytes(200))
    with pytest.raises(ValueError,match='stereo'):
        replay_stream(engine,source,tmp_path/'output.jsonl')
