import os
import sys

import soundfile as sf
import numpy as np

pcm = sys.argv[1]
wav = os.path.splitext(pcm)[0] + '.wav'

sig = np.fromfile(pcm, dtype=np.int16)

sf.write(wav, sig, 16000)


