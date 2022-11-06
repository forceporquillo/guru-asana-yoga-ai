from urllib.error import HTTPError

from matplotlib import pyplot as plt


def show_image(img, figsize=(10, 10)):
    try:
        """Shows output PIL image."""
        plt.figure(figsize=figsize)
        plt.imshow(img)
        plt.show()
    except HTTPError:
        pass
