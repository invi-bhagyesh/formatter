/opt/homebrew/Caskroom/miniforge/base/envs/trdg_env/bin/trdg -l en -c 10 -w 1 -tc "#282828" -b 1 
/opt/homebrew/Caskroom/miniforge/base/envs/trdg_env/bin/trdg -l en -c 30000 -w 1 -tc "#282828" -b 1 -rs -let

`python -m pip install --break-system-packages \
    "pillow<10.0" \
    "diffimg==0.2.3" \
    "arabic-reshaper==2.1.4" \
    "python-bidi==0.4.2" \
    "requests>=2.20.0" \
    "opencv-python>=4.2.0.32" \
    "tqdm>=4.23.0" \
    "wikipedia>=1.4.0"`
# cmds
-w : words

    trdg -l en -c 1000 -w 1


Background Type Control:
Use the -b parameter to ensure consistent backgrounds:
`trdg -l en -c 1000 -w 1 -b 1  # Plain white background`

Text color

`trdg -l en -c 1000 -w 1 -tc "#282828"  # Dark text ` 

`trdg -l en -c 1000 -w 1 -tc "#FFFFFF"  # White text`


numbers
trdg -l en -c 1000 -w 1 -rs -let -tc "#282828" -b 1