#!/bin/bash

python3 ./src/run.py --input input_list/list_CASIA-Iris-Thousand.txt --output test/result_thousand.txt
python3 ./src/run.py --input input_list/list_CASIA-Iris-Lamp.txt --output test/result_lamp.txt
python3 ./src/run.py --input input_list/list_Ganzin-J7EF-Gaze.txt --output test/result_gaze.txt