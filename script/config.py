# File: config.py

SR       = 16000
NFFT     = 1024
NMEL     = 40
HOP_SIZE = 160
WIN_SIZE = 640
WIN_TYPE = 'hann'

SECONDS  = 4
LENGTH   = SECONDS * SR

N_CLASS = 36

TRAIN_CMD_FOLDER = [
    '/home/cimlabber/colin_ws/voice/datasets/speech_command_levoice/dataset/speech_cmd_dataset'
]

TRAIN_BGN_FOLDER = [
    '/home/cimlabber/colin_ws/voice/datasets/speech_command_levoice/dataset/bgn'
]

CLASS_TABLE_FILE = '/home/cimlabber/colin_ws/voice/datasets/speech_command_levoice/speech_cmd_class_table.json'
