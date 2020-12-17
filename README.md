# TA_RezaAndriady

Publishmqtt.py = read data training per row dan dikirimkan menggunakan mqtt

RingBuffer.py = mengolah data training sehingga didapatkan persamaan proyeksi 2 dimensi menggunakan library KFDA
                mendapatkan classifier berbasis SVM berdasarkan data training yang telah diproyeksikan
                menerima data testing melalui mqtt dan dimasukkan kedalam ringbuffer berkapasitas 100
                mengolah data testing dan memprediksi label data testing

Classifier.ipynd = pembuatan ulang KFDA berdasarkan referensi [1], [2], [3]

Pada Classifier.ipynd, dapat dibandingkan perbedaan hasil pengolahan data menggunakan rumus (cell 2) dibandingkan menggunakan library Kfda (cell 3)
Apabila menggunakan seluruh data sebagai data training (2420 sampel), cell 2 menghasilkan scatter plot yang lebih tersebar dibandingkan cell 3 yang sangat terpusat
Tetapi pada cell 3, data berlabel 1 sangat berdekatan dengan data berlabel 7

Konsekuensinya, apabila menggunakan seluruh data sebagai data testing (data training = data testing), pada cell 3, SVM sama sekali tidak berhasil memisahkan data 1 dan data 7
Sehingga cell 3 memiliki akurasi 80%, sedangkan cell 2 memiliki akurasi 99%

Apabila data testing yang digunakan sama dengan data training, akurasi pada cell 2 bisa ditingkatkan lagi dengan penambahan jumlah eigenvector yang digunakan untuk
memproyeksikan data training. Hal ini akan mengakibatkan scatter plot menjadi lebih terpusat (tapi lebih terpisah dibandingkan pada cell 3). Tetapi apabila data testing
berbeda dengan data training, maka penambahan jumlah eigenvector justru kontra-produktif, karena scatter plot pada data testing akan lebih terpusat, sedangkan data training
justru lebih membesar, sehingga classifier (yang berbasis data training) lebih mungkin untuk melakukan misklasifikasi.

Pada RingBuffer.py terdapat isu yaitu hanya bisa menerima data masuk apabila subscribe ke-'#', belum diketahui pengaruh dari isu ini.
Sedang diusahakan untuk menggunakan deque (double-ended queue) alih-alih numpy_ringbuffer dikarenakan deque lebih cepat dalam meng-append dan meremove data.


