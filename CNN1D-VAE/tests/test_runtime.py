import numpy as np
import pytest
import torch

from model import ModelConfig
from runtime import VAEEngine, StreamingPredictor


@pytest.fixture
def engine():
    torch.manual_seed(11)
    torch.set_num_threads(2)
    return VAEEngine(ModelConfig(sequence_length=64))


def test_checkpoint_keeps_preprocessing_and_output(tmp_path,engine):
    from data import ChannelNormalizer
    x=np.arange(256,dtype=np.float32).reshape(2,64,2)
    engine.normalizer=ChannelNormalizer.fit(x)
    original=engine.predict_windows(x)
    engine.save(tmp_path/'prototype.pt')
    restored=VAEEngine.from_checkpoint(tmp_path/'prototype.pt')
    np.testing.assert_allclose(restored.predict_windows(x)['reconstruction_pcm'],original['reconstruction_pcm'],rtol=0,atol=0)
    assert not original['validated_for_fault_detection']
    assert original['model_status']=='untrained'


def test_streaming_matches_offline_and_preserves_tail(engine):
    x=np.random.default_rng(5).normal(0,200,(64*3+11,2)).astype(np.float32)
    stream=StreamingPredictor(engine,batch_size=2)
    output=[]
    for chunk in [x[:7],x[7:68],x[68:123],x[123:]]:output.extend(stream.push(chunk))
    reference=engine.predict_windows(x[:192].reshape(3,64,2),include_reconstruction=False)
    np.testing.assert_allclose([w['reconstruction_mse_standardized'] for w in output],reference['reconstruction_mse_standardized'],rtol=1e-5,atol=1e-7)
    assert [(w['start_frame'],w['end_frame']) for w in output]==[(0,64),(64,128),(128,192)]
    assert stream.pending_frames==11
    np.testing.assert_array_equal(stream.buffer,x[192:])
    stream.reset();assert stream.pending_frames==0 and stream.processed_frames==0


def test_integration_rejects_wrong_rate_shape_and_nonfinite(engine):
    x=np.zeros((64,2),np.float32)
    with pytest.raises(ValueError,match='44100'):engine.predict_window(x,sample_rate=22050)
    with pytest.raises(ValueError):engine.predict_window(x[:,0])
    x[0,0]=np.nan
    with pytest.raises(ValueError,match='non-finite'):engine.predict_window(x)
