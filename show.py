import cv2
import numpy as np

a = np.load("for_viz.npz")
for c in a["arr_0"][0]:
    for b, title in zip(c, ['halite', 'ships', 'ships_content', 'bases', 'position']):
        if not title in ['halite', 'ships_content']:
            b = b+1/2.0
        cv2.imshow(title, cv2.resize(b, (0,0), fx=10, fy=10))
    cv2.waitKey(1000000)

