from flask import Flask, render_template, session, url_for, request as req
import numpy as np
import cv2
import glob
import os
import math
from streamlit import caching
import natsort

app = Flask(__name__)
global nomer_rand

@app.after_request
def add_header(response):
    response.headers['Pragma'] = 'no-cache'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Expires'] = '0'

    return response


def proses(img, imgCrop):
    try:
        files = glob.glob('static/crop/*.jpg')
        for f in files:
            os.remove(f)

        h, w = img.shape[::]
        H, W, A = imgCrop.shape[::]

        source_image = img.shape

        if(w >= 3000):
            h = int(35 / 100 * h)
            w = int(35 / 100 * w)

            H = int(35 / 100 * H)
            W = int(35 / 100 * W)

            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
            imgCrop = cv2.resize(
                imgCrop, (W, H), interpolation=cv2.INTER_LINEAR)
        elif(w >= 1450):
            h = int(50 / 100 * h)
            w = int(50 / 100 * w)

            H = int(50 / 100 * H)
            W = int(50 / 100 * W)

            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
            imgCrop = cv2.resize(
                imgCrop, (W, H), interpolation=cv2.INTER_LINEAR)

        # sigma = 61/100
        # v = np.median(img)
        # lower = int(max(0, (1.0 - sigma) * v))
        # upper = int(min(255, (1.0 + sigma) * v))
        # edges = cv2.Canny(img, lower, upper)

        # cv2.imshow('Deteksi Canny', edges)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        result_image = img.shape
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 10,
                                   param1=61, param2=79, minRadius=10, maxRadius=150)
        jum = 0
        file_crop = []
        labelStatus = []
        statusLabel = ""
        if circles is not None:
            # Get the (x, y, r) as integers
            circles = np.round(circles[0, :]).astype("int")

            if (len(circles) < 150):
                print(circles)

                for (x, y, r) in circles:
                    jum += 1

                    cv2.circle(img, (x, y), r, (0, 255, 0), 2)

                    # nilai dari deteksi(titik pusat), sedangkan untuk crop jadi titik x maka dikurang r
                    hor = x - r
                    # nilai dari deteksi(titik pusat), sedangkan untuk crop jadi titik y maka dikurang r
                    ver = y - r
                    # lebar dan tinggi dari titik awal = diameter (r*2)
                    h = w = r*2
                    # croping bentuk kotak
                    crop = imgCrop[ver:ver+h, hor:hor+w]

                    # create a mask
                    # [0] = width [1] = height
                    mask = np.full(
                        (crop.shape[0], crop.shape[1]), 0, dtype=np.uint8)
                    # create circle mask, center, radius, fill color, size of the border
                    cv2.circle(mask, (r, r), r, (255, 255, 255), -1)
                    # get only the inside pixels
                    fg = cv2.bitwise_or(crop, crop, mask=mask)

                    mask = cv2.bitwise_not(mask)
                    background = np.full(crop.shape, 0, dtype=np.uint8)
                    bk = cv2.bitwise_or(background, background, mask=mask)
                    final = cv2.bitwise_or(fg, bk)

                    path = os.path.dirname(os.path.abspath(
                        __file__))+str('\static\crop\crop')+str(jum)+'.jpg'

                    # print(path)
                    cv2.imwrite(path, final)
                    isi = 'static/crop/crop'+str(jum)+'.jpg'
                    file_crop.append(isi)
            else:
                statusLabel = "Logo Terdeteksi Melebihi 150"
        else:
            statusLabel = "Tidak Terdeteksi Logo UTY"

        pathLogo = os.path.dirname(os.path.abspath(
            __file__))+str('\static\logoAcuan\detek.png')
        template = cv2.imread(pathLogo)
        hasil_deteksi = []

        if len(file_crop) > 0:
            for i in file_crop:
                deteksi = cv2.imread(i)

                # resize
                template = cv2.resize(template, (150, 150),
                                      interpolation=cv2.INTER_LINEAR)
                deteksi = cv2.resize(deteksi, (150, 150),
                                     interpolation=cv2.INTER_LINEAR)

                # simpan nilai keabuan setiap pixel
                tempR, tempG, tempB,  = [], [], []
                detekR, detekG, detekB = [], [], []

                for x in range(0, 150, 1):
                    for y in range(0, 150, 1):
                        B, G, R = template[x, y]
                        b, g, r = deteksi[x, y]
                        tempR.append(R)
                        tempG.append(G)
                        tempB.append(B)

                        detekR.append(r)
                        detekG.append(g)
                        detekB.append(b)

                # cari mean citra template dan deteksi
                # channel RED
                meanTempR = np.mean(tempR)
                meanDetekR = np.mean(detekR)
                # PROSES TEMPLATE MATCHING CORRELATION
                pembilang = 0
                penyebuti = 0
                penyebutj = 0

                for i, j in zip(tempR, detekR):

                    # cari nilai setiap piksel (pembilang)
                    nilaiPixeli = i - meanTempR
                    nilaiPixelj = j - meanDetekR
                    ixj = nilaiPixeli * nilaiPixelj
                    # Gabungin pembilang
                    pembilang += ixj

                    # cari nilai setiap piksel i (penyebut)
                    nilaiPixeli2 = pow((i-meanTempR), 2)
                    penyebuti += nilaiPixeli2

                    # cari nilai setiap piksel j (penyebut)
                    nilaiPixelj2 = pow((j-meanTempR), 2)
                    penyebutj += nilaiPixelj2

                rRED = pembilang / math.sqrt(penyebuti*penyebutj)

                # Channel GREEN
                meanTempG = np.mean(tempG)
                meanDetekG = np.mean(detekG)
                # PROSES TEMPLATE MATCHING CORRELATION
                pembilang = 0
                penyebuti = 0
                penyebutj = 0

                for i, j in zip(tempG, detekG):

                    # cari nilai setiap piksel (pembilang)
                    nilaiPixeli = i - meanTempG
                    nilaiPixelj = j - meanDetekG
                    ixj = nilaiPixeli * nilaiPixelj
                    # Gabungin pembilang
                    pembilang += ixj

                    # cari nilai setiap piksel i (penyebut)
                    nilaiPixeli2 = pow((i-meanTempG), 2)
                    penyebuti += nilaiPixeli2

                    # cari nilai setiap piksel j (penyebut)
                    nilaiPixelj2 = pow((j-meanTempG), 2)
                    penyebutj += nilaiPixelj2

                rGREEN = pembilang / math.sqrt(penyebuti*penyebutj)

                # channel BLUE
                meanTempB = np.mean(tempB)
                meanDetekB = np.mean(detekB)
                # PROSES TEMPLATE MATCHING CORRELATION
                pembilang = 0
                penyebuti = 0
                penyebutj = 0

                for i, j in zip(tempB, detekB):

                    # cari nilai setiap piksel (pembilang)
                    nilaiPixeli = i - meanTempB
                    nilaiPixelj = j - meanDetekB
                    ixj = nilaiPixeli * nilaiPixelj
                    # Gabungin pembilang
                    pembilang += ixj

                    # cari nilai setiap piksel i (penyebut)
                    nilaiPixeli2 = pow((i-meanTempB), 2)
                    penyebuti += nilaiPixeli2

                    # cari nilai setiap piksel j (penyebut)
                    nilaiPixelj2 = pow((j-meanTempB), 2)
                    penyebutj += nilaiPixelj2

                rBLUE = pembilang / math.sqrt(penyebuti*penyebutj)

                r = round((rRED + rGREEN + rBLUE)/3, 4)

                hasil_deteksi.append(r)
        hasil_prediksi = []
        if len(file_crop) > 0:
            persenKesamaan = max(hasil_deteksi)
            indeks = hasil_deteksi.index(max(hasil_deteksi))
            status_indeks = str(indeks+1)+'.jpg'
            confidence = hasil_deteksi
            nilai_conf = round(persenKesamaan*100, 2)
            status = ""
            if persenKesamaan*100 >= 72.5:
                status = "Logo Pada Sertifikat Sesuai Dengan Ketentuan"
            else:
                status = "Logo Pada Sertifikat Tidak Sesuai Dengan Ketentuan"
        hasil_prediksi.extend(
            [confidence, nilai_conf, status, status_indeks, source_image, result_image])
        return hasil_prediksi
    except:
        hasil_prediksi.extend(
            [0, 0, "Tidak adanya logo yang dicek", "tidak ada lingkaran terdeteksi", source_image, result_image])
        return hasil_prediksi


app.config["IMAGE_UPLOADS"] = "static/scaling"


@app.route('/', methods=['GET', 'POST'])
def index():
    if req.method == "GET":
        return render_template('index.html', hasil=[])
    else:
        imgCrop = req.files["image-source"]

        nama_file = "citraInput.jpg"
        # session['namafile'] = nama_file
        imgCrop.save(os.path.join(
            app.config["IMAGE_UPLOADS"], nama_file))
        print(imgCrop)

        path = os.path.dirname(os.path.abspath(__file__)) + \
            str('\static\scaling\citraInput.jpg')

        imgCrop = cv2.imread(path)
        img = cv2.imread(path, 0)

        hasil = proses(img, imgCrop)

        data = os.path.dirname(os.path.abspath(__file__))+str('\static\crop')
        img = []
        for i in os.listdir(data):
            if os.path.isfile:
                img.append(i)
        caching.clear_cache()
        

        img = natsort.natsorted(img, reverse=False)
        return render_template('index.html', hasil=hasil, img=img)


if __name__ == "__main__":
    app.run(debug=True)
