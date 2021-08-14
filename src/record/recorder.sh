#!/usr/bin/bash -e 

if [ $# -ne 1 ]; then
    echo 'Usage: bash ./recorder.sh <out-filename>'
    exit 1
fi

outfile=$1
exe='./record.exe'

source py3/kws/bin/activate

function record() {
    local outfile=$1 

    rm recording.pcm
    ${exe} 
    python3 towav.py recording.pcm 
    mv recording.wav ${outfile}
    rm recording.pcm
}

record ${outfile}
