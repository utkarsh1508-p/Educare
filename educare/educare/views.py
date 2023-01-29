from django.shortcuts import render
import os
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from base64 import b64decode
import tensorflow.lite as tflite

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Interpreter for alphabets
interpreter1 = tflite.Interpreter(model_path=BASE_DIR+r'\media\Alphabets.tflite')
interpreter1.allocate_tensors()
input_details1 = interpreter1.get_input_details()
output_details1 = interpreter1.get_output_details()

# Interpreter for numbers
interpreter2 = tflite.Interpreter(model_path=BASE_DIR+r'\media\Digits.tflite')
interpreter2.allocate_tensors()
input_details2 = interpreter2.get_input_details()
output_details2 = interpreter2.get_output_details()


def home(request):
    return render(request, 'home.html')


def alphabet(request):
    if request.method == "POST":
        captured_image = request.POST['canvasData']
        selectedword = request.POST['selectedword']
        datatext = str(captured_image)
        completeName = BASE_DIR + f"/media/urldata.txt"
        with open(completeName, 'w') as file:
            file.write(datatext)

        try:
            im = Image.open(BytesIO(b64decode(datatext.split(',')[1])))
            im.save(BASE_DIR + "/media/image.jpg")
        except:
            pass

        try:
            image = cv2.imread(BASE_DIR + "/media/image.jpg")
            original = image.copy()

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (3, 3), 0)
            canny = cv2.Canny(blur, 120, 255, 1)

            cnts = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            min_area = 100
            images = []
            boundary = []
            classes = []
            for c in cnts:
                area = cv2.contourArea(c)
                if area > min_area:
                    x, y, w, h = cv2.boundingRect(c)
                    arr = [x, y, w, h]
                    boundary.append(arr)

            boundary = np.array(sorted(boundary, key=lambda x: x[0]))
            for i in boundary:
                x, y, w, h = i
                cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 2)
                char = original[y:y + h, x:x + w]
                images.append(char)

            for i in images:
                i = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
                i = cv2.resize(i, (20, 20))

                x = np.asarray(i)
                add_c = np.zeros((20, 4))
                add_r = np.zeros((4, 28))
                x = np.concatenate((add_c, x), axis=1)
                x = np.concatenate((x, add_c), axis=1)
                x = np.concatenate((x, add_r), axis=0)
                x = np.concatenate((add_r, x), axis=0)

                x = x.reshape((1, 28, 28, 1))
                x = x / 255
                x = np.float32(x)

                # Use the TensorFlow Lite Interpreter to set the input tensor and run inference
                interpreter1.set_tensor(input_details1[0]['index'], x)
                interpreter1.invoke()
                temp = interpreter1.get_tensor(output_details1[0]['index'])
                classes.append(np.argmax(temp))

            dict = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 9: 'I', 10: 'J', 11: 'K',
                    12: 'L', 13: 'M', 14: 'N', 15: 'O', 16: 'P', 17: 'Q', 18: 'R', 19: 'S', 20: 'T', 21: 'U',
                    22: 'V', 23: 'W', 24: 'X', 25: 'Y', 26: 'Z', 27: 'a', 28: 'b', 29: 'd', 30: 'e', 31: 'f',
                    32: 'g', 33: 'h', 34: 'n', 35: 'q', 36: 'r', 37: 't'}

            # key_list = list(dict.keys())
            val_list = list(dict.values())
            word = val_list.index(selectedword)
            res = ''
            for c in classes:
                res += str(dict[c + 1])

            # confidence score
            c_score = round(temp[0][word] * 100, 2)

            # delete temporary files
            os.remove(BASE_DIR + f"/media/urldata.txt")
            os.remove(BASE_DIR + f"/media/image.jpg")
            return render(request, 'alphabets.html', {"dataval": res, "accuracy": c_score})

        except Exception as e:
            print(e)
            return render(request, 'alphabets.html')

    return render(request, 'alphabets.html')


def number(request):
    if request.method == "POST":
        captured_image = request.POST['canvasData']
        selectedword = request.POST['selectedword']
        datatext = str(captured_image)
        completeName = BASE_DIR + f"/media/urldata.txt"
        with open(completeName, 'w') as file:
            file.write(datatext)

        try:
            im = Image.open(BytesIO(b64decode(datatext.split(',')[1])))
            im.save(BASE_DIR + "/media/image.jpg")
        except:
            pass

        try:
            image = cv2.imread(BASE_DIR + "/media/image.jpg")
            original = image.copy()

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (3, 3), 0)
            canny = cv2.Canny(blur, 120, 255, 1)

            cnts = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            min_area = 100
            images = []
            boundary = []
            classes = []
            for c in cnts:
                area = cv2.contourArea(c)
                if area > min_area:
                    x, y, w, h = cv2.boundingRect(c)
                    arr = [x, y, w, h]
                    boundary.append(arr)

            boundary = np.array(sorted(boundary, key=lambda x: x[0]))
            for i in boundary:
                x, y, w, h = i
                cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 2)
                char = original[y:y + h, x:x + w]
                images.append(char)

            for i in images:
                i = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
                i = cv2.resize(i, (20, 20))

                x = np.asarray(i)
                add_c = np.zeros((20, 4))
                add_r = np.zeros((4, 28))
                x = np.concatenate((add_c, x), axis=1)
                x = np.concatenate((x, add_c), axis=1)
                x = np.concatenate((x, add_r), axis=0)
                x = np.concatenate((add_r, x), axis=0)
                x = x.reshape((1, 28, 28, 1))
                x = x / 255
                x = np.float32(x)

                # Use the TensorFlow Lite Interpreter to set the input tensor and run inference
                interpreter2.set_tensor(input_details2[0]['index'], x)
                interpreter2.invoke()
                temp = interpreter2.get_tensor(output_details2[0]['index'])
                classes.append(np.argmax(temp))

            res = ''
            for c in classes:
                res += str(c)

            # confidence score
            c_score = round(temp[0][int(selectedword)] * 100, 2)

            # delete temporary files
            os.remove(BASE_DIR + f"/media/urldata.txt")
            os.remove(BASE_DIR + f"/media/image.jpg")
            return render(request, 'numbers.html', {"dataval": res, "accuracy": c_score})

        except:
            return render(request, 'numbers.html')
    return render(request, 'numbers.html')
