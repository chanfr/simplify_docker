import StringIO

import numpy as np
from PIL import Image


class ImageRetriever:

    @staticmethod
    def getImage(request, key):
        file = request.files[key]
        data = file.read()
        pil_image = Image.open(StringIO.StringIO(data))
        cv_image = np.array(pil_image)
        return cv_image
