#!/usr/bin/env python3

from picamera2 import Picamera2, Preview
import time
import sys

print("NOTE: requires single argument: filename of saved image.")

picam2 = Picamera2()
camera_config = picam2.create_preview_configuration()
picam2.configure(camera_config)
picam2.color_effects = None
#picam2.start_preview(Preview.DRM)
picam2.start_preview(Preview.NULL)
picam2.start()
time.sleep(2)
picam2.capture_file(sys.argv[1] + ".jpg")
