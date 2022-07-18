import numpy as np
import cv2

#########################################
# Membaca semua file byte malware
# dan menyimpannya ke dalam Gray dan RGB
#########################################


# Fungsi untuk merubah angka heksadesimal menjadi desimal
def hextodec(x):
    v = []
    for t in x:
        try:
            int(t, 16)
            v.append(int(t, 16))
        except:
            False
    return np.asarray(v)


# Membaca file binary dan merubahnya ke dalam bentuk desimal
def theByteFile(fileName, readElement=-1):
    data_read = 0
    im = []
    with open(fileName, "rb") as f:
        line = f.readline()
        while line:
            if data_read >= readElement and readElement >= 0:
                break
            # Memecah data setiap spasi ke dalam bentuk list
            line = np.asarray(line.split())
            # Melewatkan elemen pertama dari list yang merujuk pada nomor baris
            h = hextodec(line[1:])
            if (len(h) > 0):
                im.append(h)
                data_read += len(h)
            line = f.readline()
    # Menyatukan semua elemen ke dalam array
    return np.asarray(im, dtype="object").flatten("F")


# Membaca file CSV ke dalam bentuk sequence
def theCSV(file, bracket=False):
    import csv
    array = []
    with open(file, newline='') as csvfile:
        if(bracket != False):
            read = csv.reader(csvfile, delimiter=bracket, quotechar='|')

        else:
            read = csv.DictReader(csvfile)

        for row in read:
            array.append(row)
    return array


# Melakukan pengaturan array satu dimensi yang diinputkan untuk dirubah ke dalam matriks yang diinginkan untuk selanjutnya siap dirubah menjadi gambar
def settingArrayImageSize(array, mode="GRAY"):
    if(mode.upper() == "GRAY"):
        # Untuk output grayscale
        # Data dimampatkan untuk mengetahui nilai akar kuadrat data
        c = int(np.ceil(np.sqrt(array.shape[0])))
        # Data hasil akar kuadrat di kuadratkan kembali untuk membentuk persegi
        k = int(np.square(c))
        # Mengisi kekosongan data yang hilang dengan 0
        r = np.append(array, np.zeros(k - array.shape[0]))
        r = r.reshape(c, c, 1)  # 1 channel untuk grayscale
    elif (mode.upper() == "COLORFUL" or mode.upper() == "RGB"):
        # Untuk output RGB first find the exact split to 3 and then find the nearest frame
        # Data dimampatkan dengan akar kuadrat kemudian dibagi tiga
        c = int(np.ceil(np.sqrt(array.shape[0] / 3)))
        # Data hasil kuadrat dikuadratkan kembali dan dikalikan 3 untuk membentuk persegi
        k = int(np.square(c)) * 3
        # Mengisi kekosongan data yang hilang dengan 0
        r = np.append(array, np.zeros(k - array.shape[0]))
        r = r.reshape(c, c, 3)  # 3 channel untuk rgb
    else:
        r = False
    return r


# Menyimpan matriks yang siap diubah menjadi gambar menggunakan CV2
def savePictureAsArray(array, img_name):
    try:
        data_read = array.shape[0]
        data_read = array.shape[1]
        data_read = array.shape[2]
        cv2.imwrite(img_name, array)
    except:
        print("Index NumPy harus memiliki 3 dimensi dan color channel supaya dapat diubah menjadi gambar")
