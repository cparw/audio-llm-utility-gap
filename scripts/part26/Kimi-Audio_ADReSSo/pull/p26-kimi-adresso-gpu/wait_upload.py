# keeps one python process alive while the Mac streams the audio, so the watchdog does not count the pod as idle; exits at UPLOAD_DONE or after 45 min
import os, time
t0 = time.time()
while not os.path.exists("/workspace/UPLOAD_DONE") and time.time() - t0 < 2700: time.sleep(5)
