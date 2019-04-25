import json

import cv2
import numpy as np
import requests
from sklearn import datasets
from sklearn.model_selection import train_test_split

from img_utils.image_sender import ImageSender


def getMNISImage():
    dataset = datasets.fetch_mldata("MNIST Original")
    data = dataset.data.reshape((dataset.data.shape[0], 28, 28))
    data = data[:, np.newaxis, :, :]
    (trainData, testData, trainLabels, testLabels) = train_test_split(
        data / 255.0, dataset.target.astype("int"), test_size=0.33)

    idx = np.random.choice(np.arange(0, len(testLabels)))
    image = testData[np.newaxis, idx][0][0]
    image = (image * 255).astype("uint8")
    return image


def process_lenet(host, port, n_samples):
    for i in range(0, n_samples):
        url = "{}:{}/process".format(host, port)
        image = getMNISImage()
        ret = ImageSender.prepare_to_send(image)
        files = {'media': ("temp.png", ret)}
        data = requests.post(url, files=files)

        try:
            responseData = json.loads(data.content)
            image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            cv2.putText(image, str(responseData["prediction"]), (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

            cv2.imshow("Result", image)
            cv2.waitKey(0)
        except:
            print "Error: ", data.content


if __name__ == "__main__":
    api_endpoint = "35.180.91.233"
    host = "http://{}".format(api_endpoint)
    port = "80"
    n_samples = 10
    process_lenet(host, port, n_samples)
