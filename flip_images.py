#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2

input_path = "data/track_2_data_r/IMG/"
output_path = "data/track_2_data_r_flipped/IMG/"

lines = []
with open('data/track_2_data_r/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
            
#lines = lines[1:]

for line in lines:
    print(line)
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = input_path + filename
    image = cv2.imread(current_path)
    
    flipVertical = cv2.flip(image, 1)
    
    cv2.imwrite(output_path + filename, flipVertical)