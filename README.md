# CapstoneProjectModule3_MachineLearning
Folder ini berisi Capstone Project Purwadhika BSD membuat Machine Learning

## CPM3_mzm.ipynb
## **BUSINESS UNDERSTANDING**

**Context**  
Saat ini industri telekomunikasi seluler telah berkembang pesat, sehingga persaingan antar provider (perusahaan operator telekomunikasi seluler) menjadi semakin ketat. Salah satu tantangan yang kini dihadapi provider adalah usaha
menurunkan jumlah pelanggan yang berhenti menggunakan layanan perusahaan dan beralih langganan ke perusahaan kompetitor.

Suatu provider ingin mengetahui pelanggan yang bagaimana yang akan pindah (churn) dari provider tersebut sehingga jumlah pelanggan yang beralih (churn) dapat dikurangi. Seorang Data Scientist diminta untuk membuat model prediksi yang tepat untuk menentukan pelanggan akan berhenti berlangganan (churn) atau tidak dengan menggunakan machine learning. 

Target :
0 : Tidak berhenti berlangganan
1 : Berhenti berlangganan (churn)

**Problem Statement :**

Tingginya persentase customer churn menjadi salah satu indikator tingkat kegagalan suatu perusahaan telekomunikasi, maka perlu dilakukan upaya-upaya untuk mengurangi persentase customer churn tersebut. Perusahaan umumnya lebih memilih untuk mempertahankan pelanggan, karena dibutuhkan biaya yang lebih sedikit untuk mempertahankan pelanggan {customer retention cost) daripada menambah pelanggan yang baru (customer acquisition cost). Berdasarkan informasi dari [internet](https://www.outboundengine.com/blog/customer-retention-marketing-vs-customer-acquisition-marketing/), memperoleh pelanggan baru dapat menghabiskan biaya lima kali lebih banyak daripada mempertahankan pelanggan yang sudah ada. Adapun rata-rata biaya customer acquisition cost untuk industri telekomunikasi adalah sekitar $315 per new customer ([sumber](https://www.revechat.com/blog/customer-acquisition-cost/)).

Perusahaan telekomunikasi dapat memberikan insentif retensi seperti memberikan potongan harga, memberikan paket layanan yang menarik, memberikan
prioritas pelayanan dan lain-lain dalam upaya untuk mempertahankan pelanggan. Namun, kebijakan pemberian insentif retensi belum sepenuhnya dilakukan secara efektif. Karena jika insentif retensi tersebut diberikan secara merata kepada seluruh pelanggan, maka pengeluaran biaya tersebut menjadi tidak efektif dan mengurangi potensi keuntungan apabila pelanggan tersebut memang loyal dan tidak ingin berhenti berlangganan.

**Goals :**

Maka berdasarkan permasalahan-permasalahan di atas, perusahaan ingin memiliki kemampuan untuk memprediksi kemungkinan seorang pelanggan akan berhenti berlangganan atau tidak, sehingga dapat memfokuskan upaya-upaya retensi pada pelanggan yang terindikasi untuk churn.

Dan juga, perusahaan ingin mengetahui faktor-faktor apa saja yang cenderung mempengaruhi pelanggan bertahan, sehingga mereka dapat membuat program-program yang lebih tepat sasaran dalam mengurangi jumlah pelanggan yang churn.

**Analytic Approach :**

Jadi yang akan kita lakukan adalah menganalisis data untuk menemukan pola yang membedakan pelanggan yang akan berhenti berlangganan (churn) atau tidak.

Kemudian kita akan membangun model klasifikasi yang akan membantu perusahaan untuk dapat memprediksi probabilitas seorang pelanggan akan berhenti berlangganan (churn) atau tidak.

**Metric Evaluation**

Karena fokus utama kita adalah pelanggan yang akan berhenti berlangganan, maka target yang kita tetapkan adalah sebagai berikut:

Target :
- 0 : Tidak berhenti berlangganan
- 1 : Berhenti berlangganan (churn)

Type 1 error : False Positive (pelanggan yang aktualnya tidak churn tetapi diprediksi churn)
Konsekuensi: tidak efektifnya pemberian insentif retensi

Type 2 error : False Negative (pelanggan yang aktualnya churn tetapi diprediksi tidak akan churn)
Konsekuensi: kehilangan pelanggan

Untuk memberikan gambaran konsekuensi secara kuantitatif, maka kita akan coba hitung dampak biaya berdasarkan asumsi berikut :
- Customer Lifetime Period untuk pelanggan yang churn sekitar 17.7 bulan 
- Customer Acquisition Cost (CAC) = $315 per customer ([sumber](https://www.revechat.com/blog/customer-acquisition-cost/)) / 17.7 bulan = $17.79 per bulan per customer
- Customer Retention Cost (CRC)= 1/5 ([sumber](https://www.outboundengine.com/blog/customer-retention-marketing-vs-customer-acquisition-marketing/)) * CAC = 1/5 * $17.79 = $3.56 per bulan per customer
- Average Customer MonthlyCharge = $64.88 per bulan per customer

Berdasarkan asumsi di atas maka kita dapat coba kuantifikasi konsekuensinya sebagai berikut :
- tidak efektifnya pemberian insentif retensi --> maka kita menyia-nyiakan biaya CRC sebesar $3.56 per bulan per pelanggan
- kehilangan pelanggan --> maka kita kehilangan pendapatan dan juga perlu mengeluarkan kembali biaya CAC sehingga secara total kita kehilangan $17.79 + $64.88 = $82.67 per bulan per pelanggan

Berdasarkan konsekuensinya, maka sebisa mungkin yang akan kita lakukan adalah membuat model yang dapat mengurangi customer churn dari perusahaan tersebut, khususnya jumlah False Negative (pelanggan yang aktualnya churn tetapi diprediksi tidak akan churn), tetapi juga dapat meminimalisir pemberian insentif yang tidak tepat. Jadi nanti metric utama yang akan kita gunakan adalah f2_score, karena recall kita anggap dua kali lebih penting daripada precision.

## **DATA UNDERSTANDING**

Dataset ini berisi informasi tentang :

- Informasi demografi pelanggan yaitu `Dependents`.
- Service yang digunakan oleh pelanggan : `Online Security`, `Online Backup`, `Internet Service`, `Device Protection`, `Tech Support`
- Informasi akun pelanggan : `tenure`, `Contract`, `PaperlessBilling`, dan `MonthlyCharges`
- Pelanggan yang berhenti berlangganan – kolomnya disebut `Churn`

Deskripsi terkait kolom-kolom pada dataset Telco Customer Churn tersebut, dapat dilihat pada tabel berikut :  

| Attribute | Data Type | Description |
| --- | --- | --- |
| Dependents | Text | Indicates if the customer lives with any dependents: Yes, No. Dependents could be children, parents, grandparents, etc. |
| tenure | Integer | Indicates the total amount of months that the customer has been with the company.  |
| OnlineSecurity | Text | Indicates if the customer subscribes to an additional online security service provided by the company. |
| OnlineBackup | Text | Indicates if the customer subscribes to an additional online backup service provided by the company. |
| InternetService | Text | Indicates if the customer subscribes to Internet service with the company. |
| DeviceProtection | Text | Indicates if the customer subscribes to an additional device protection plan for their Internet equipment provided by the company. |
| TechSupport | Text | Indicates if the customer subscribes to an additional technical support plan from the company with reduced wait times. |
| Contract | Text | Indicates the customer’s current contract type. |
| PaperlessBilling | Text | Indicates if the customer has chosen paperless billing. |
| MonthlyCharges | Float | Indicates the customer’s current total monthly charge for all their services from the company. |
| Churn | Text | Yes = the customer left the company this quarter. No = the customer remained with the company. |

Note : 
- Setiap baris data merepresentasikan informasi seorang pelanggan
- Sebagian besar feature bersifat kategorikal :
    - Kategorikal : `Dependents, Online Security, Online Backup, Internet Service, Device Protection, Tech Support, Contract, PaperlessBilling`
    - Numerikal : `tenure, MonthlyCharges`
- Target adalah kolom `Churn` (No = 3614 data, Yes = 1316 data)
- Dataset tidak seimbang (mild imbalance) karena proporsi kelas minoritas sebesar 26.69% berada dalam rentang 20-40% dari dataset ([sumber](https://developers.google.com/machine-learning/data-prep/construct/sampling-splitting/imbalanced-data))
- Tidak terdapat missing values pada dataset

## **DATA PREPARATION**
### *CHANGE VALUE*

Kita ubah terlebih dahulu nilai pada kolom `Churn` dari 'Yes' dan 'No' menjadi 1 dan 0. Karena fokus utama kita adalah yang churn, maka :

#### `Churn`
Target :
0 : Tidak berhenti berlangganan (`Churn` == 'No')
1 : Berhenti berlangganan (`Churn` == 'Yes')

#### `OnlineSecurity, OnlineBackup, DeviceProtection` dan `TechSupport`
Jika kita perhatikan pada feature `OnlineSecurity, OnlineBackup, DeviceProtection` dan `TechSupport`, maka kita dapat melihat terdapat nilai unik 'No internet service' selain nilai 'Yes' dan 'No'.\
Padahal secara pengertian, data dengan nilai 'No internet service' pada kolom-kolom tersebut memiliki arti yang sama dengan nilai 'No' karena ketika pelanggan tidak menggunakan `InternetService`, maka pelanggan tersebut juga tidak menggunakan layanan `OnlineSecurity, OnlineBackup, DeviceProtection` dan `TechSupport` seperti terlihat pada tabel di bawah (tidak ada yang 'Yes').

### *ENCODING*
Sekarang mari kita melakukan fitur encoding untuk fitur-fitur categorical yang kita miliki.
Yang akan kita lakukan adalah :


1. Merubah fitur/kolom `Dependents` menggunakan One Hot Encoding, karena fitur ini tidak memiliki urutan/tidak ordinal, dan juga jumlah unique datanya hanya sedikit.
2. Merubah fitur/kolom `OnlineSecurity` menggunakan One Hot Encoding, karena fitur ini tidak memiliki urutan/tidak ordinal, dan juga jumlah unique datanya hanya sedikit.
3. Merubah fitur/kolom `OnlineBackup` menggunakan One Hot Encoding, karena fitur ini tidak memiliki urutan/tidak ordinal, dan juga jumlah unique datanya hanya sedikit.
4. Merubah fitur/kolom `InternetService` menggunakan One Hot Encoding, karena fitur ini memiliki churn rate yang berbeda-beda. Untuk nilainya akan kita urutkan berdasarkan churn rate tertinggi dimana 'Fiber optic' akan kita ubah menjadi 3, 'DSL' kita ubah menjadi 2, dan sisanya 'No' diubah menjadi 1.
5. Merubah fitur/kolom `DeviceProtection` menggunakan One Hot Encoding, karena fitur ini tidak memiliki urutan/tidak ordinal, dan juga jumlah unique datanya hanya sedikit.
6. Merubah fitur/kolom `TechSupport` menggunakan One Hot Encoding, karena fitur ini tidak memiliki urutan/tidak ordinal, dan juga jumlah unique datanya hanya sedikit.
7. Merubah fitur/kolom `Contract` menjadi integer 1-3 dengan Ordinal Encoding, karena fitur ini adalah lama kontrak dalam satuan bulan dan tahun. Untuk nilainya akan kita urutkan berdasarkan churn rate tertinggi dimana 'Month-to-month' akan kita ubah menjadi 3, 'One year' kita ubah menjadi 2, dan sisanya 'Two year' diubah menjadi 1.
8. Merubah fitur/kolom `PaperlessBilling` menggunakan One Hot Encoding, karena fitur ini tidak memiliki urutan/tidak ordinal, dan juga jumlah unique datanya hanya sedikit.

### *SCALING*
Mengingat dalam pembuatan model machine learning nantinya kita juga akan mencoba menggunakan algoritma Logistic Regression dan KNN maka kita akan menerapkan scaling.

Harapannya dengan memiliki fitur pada skala yang sama maka kinerja algoritma machine learning akan meningkat karena setiap fitur dapat berkontibusi sama pada target. Jika tidak dilakukan scaling maka variabel skala besar akan mendominasi fitur skala kecil khususnya untuk algoritma yang memperhitungkan jarak seperti KNN.

Kemudian karena data pada kolom numerikal `tenure` dan `MonthlyCharges` juga tidak terdapat outlier (berdasarkan boxplot di pada EDA), maka kita bisa menggunakan MinMaxScaler().

### *IMBALANCE DATA*
Kita dapat melihat bahwa dataset tidak seimbang (mild imbalance) karena proporsi kelas minoritas sebesar 26.69% berada dalam rentang 20-40% dari dataset ([sumber](https://developers.google.com/machine-learning/data-prep/construct/sampling-splitting/imbalanced-data)).\
Untuk mensiasati hal tersebut maka kita akan menerapkan metode resampling agar data kita memiliki distribusi kelas yang lebih seimbang.

Adapun metode yang akan dipakai adalah Synthetic Minority Over-sampling Technique for Nominal and Continuous (SMOTENC). Hal ini dikarenakan : 
- untuk menghindari terbuangnya data pada kelas mayoritas jika menggunakan undersampling sehingga dapat menghilangkan
informasi penting pada data tersebut. 
- untuk menghindari overfitting karena penduplikasian data yang telah ada sebelumnya sehingga pengklasifikasi terkena informasi yang sama jika menggunakan Random Oversampling.
- pada data kita terdapat fitur yang numerikal (continuous) and kategorikal (nominal).

## **MODELING & EVALUATION**
- Terlampir pada file

## **CONCLUSION & RECOMMENDATION**
#### *CONCLUSION*

- Metric utama yang akan kita gunakan adalah f2_score, karena recall kita anggap dua kali lebih penting daripada precision.
    <br>
- Berdasarkan hyperparameter tuning, parameter terbaik yang dapat digunakan untuk benchmark model Decision Tree adalah :
    - max_depth = 2 
    - min_samples_split=182
    - min_samples_leaf=21
    <br>
- Berdasarkan pemodelan Decision Tree, fitur/kolom `Contract` adalah yang paling penting dan berpengaruh terhadap target (Churn), kemudian diikuti dengan `tenure` dan `MonthlyCharges`.
    <br>
- Interpretasi pada plot tree dari Decision Tree adalah sebagai berikut :
    - Jika ada pelanggan dengan `Contract` Two year atau One Year dan membayar `Monthly Charges` <= 102.11, maka pelanggan tersebut akan bertahan (Not Churn)
    - Jika ada pelanggan dengan `Contract` Two year atau One Year dan membayar `Monthly Charges` > 102.11, maka pelanggan tersebut akan bertahan (Not Churn)
    - Jika ada pelanggan dengan `Contract` 'Month-to-Month' dan memiliki `tenure` <= 10.5 bulan, maka pelanggan tersebut akan berhenti berlangganan (Churn)
    - Jika ada pelanggan dengan `Contract` 'Month-to-Month' dan memiliki `tenure` > 10.5 bulan, maka pelanggan tersebut akan berhenti berlangganan (Churn)
    <br>
- Berdasarkan hyperparameter tuning, parameter terbaik yang dapat digunakan untuk benchmark model LightGBM adalah :
    - scale_pos_weight = 5
    - learning_rate = 0.01
    - max_depth = 6
    <br>
- Interpretasi SHAP untuk model LightGBM :
    - Pelanggan dengan `Contract` Month-to-month cenderung memiliki kemungkinan Churn yang lebih tinggi. Sedangkan semakin panjang `Contract` One-year dan Two-year, maka semakin besar kemungkinan untuk Not Churn.
    - Semakin tinggi `MonthlyCharges`, semakin besar kemungkinan pelanggan untuk Churn.
    - Semakin pendek `tenure`, semakin besar kemungkinan pelanggan untuk Churn
    - Pelanggan dengan `InternetService` Fiber optic, cenderung memiliki kemungkinan Churn yang lebih tinggi dibandingkan dengan pelanggan yang menggunakan `InternetService` DSL atau No. Bahkan pelanggan tanpa `InternetService` cenderung untuk Not Churn.
    - Pelanggan tanpa `OnlineSecurity`, cenderung memiliki kemungkinan Churn yang lebih tinggi dibandingkan yang menggunakan `OnlineSecurity`.
    <br>
- Berdasarkan contoh perhitungan biaya :
    - Potensi kerugian yang mungkin didapat tanpa adanya penerapan machine learning diperkirakan sebesar : $16572.02 per bulan untuk 986 pelanggan
    - Potensi kerugian yang mungkin didapat dengan menerapkan model Decision Tree yang telah dibuat diperkirakan sebesar : $16106.04 per bulan untuk 986 pelanggan
    - Potensi kerugian yang mungkin didapat dengan menerapkan model LightGBM yang telah dibuat diperkirakan sebesar : $15799.49 per bulan untuk 986 pelanggan.
    <br>
- Berdasarkan contoh hitungan tersebut, terlihat bahwa dengan menggunakan model kita, maka perusahaan dapat menghemat sebesar :
    - Dengan Model Decision Tree : $ 465.98 per bulan untuk 986 pelanggan.
    - Dengan Model LightGBM : $ 772.53 per bulan untuk 986 pelanggan.
    - Mengingat jumlah pelanggan untuk provider telekomunikasi bisa mencapai jumlah jutaan pelanggan, tentunya potensi penghematan yang didapat bisa lebih besar lagi apabila karakteristik pelanggan masih masuk dalam rentang data yang digunakan dalam pemodelan.
    
**Model Limitation**

Kita harus berhati-hati ketika melakukan interpretasi di luar interval amatan independen variabel.

Model ini hanya berlaku pada rentang data yang digunakan pada pemodelan ini yaitu :
* `tenure` antara 0 sampai dengan 72 bulan 
* `MonthlyCharges` antara 18.8 sampai dengan 118.65
* `Contract` dalam jangka Month-to-month, One year, dan Two Year
* `InternetService` berupa 'DSL', 'Fiber Optic' dan 'No'
* `Dependent, Paperless Billing` dengan nilai 'Yes' atau 'No'
* `OnlineSecurity, OnlineBackup, DeviceProtection, dan TechSupport` berisi pilihan 'Yes', 'No' atau 'No internet service'.

Pada kasus ini, analisis dan hasil prediksi dari model yang telah dibuat tidak valid untuk :
* `tenure` lebih besar dari 72 bulan 
* `MonthlyCharges` kurang dari 18.8 atau lebih besar dari 118.65
* Jenis `Contract` selain Month-to-month, One year, dan Two Year
* `InternetService` selain 'DSL', 'Fiber Optic' dan 'No'
* `Dependent, Paperless Billing` dengan nilai selain 'Yes' atau 'No'
* `OnlineSecurity, OnlineBackup, DeviceProtection, dan TechSupport` berisi pilihan selain 'Yes', 'No' atau 'No internet service'.

#### *RECOMMENDATION*

Beberapa langkah aksi yang dapat dilakukan perusahaan untuk meminimalisir jumlah pelanggan yang akan berhenti berlangganan (churn) di antaranya :
- Memberikan insentif atau reward yang menarik bagi pelanggan untuk beralih dari `Contract` Month-to-month yang bersifat jangka pendek menjadi `Contract` One year atau Two year yang lebih bersifat jangka panjang.
- Membuat Customer Loyalty Program yang mendorong pelanggan agar tetap bertahan dan memiliki waktu `tenure` yang panjang. Bentuk program bisa berupa pemberian reward yang besarannya disesuaikan dengan masa `tenure`. Semakin panjang `tenure`, semakin besar reward yang bisa didapat, sehingga mendorong pelanggan untuk memiliki `tenure` yang lebih panjang.
- Memberikan diskon/potongan harga `MonthlyCharges` bagi pegawai yang terindikasi/diprediksi akan churn, khususnya untuk pelanggan yang memiliki `MonthlyCharges`yang tinggi.
- Menyediakan layangan `InternetService` Fiber optic dengan harga yang lebih murah. Kita bisa lihat pada EDA bahwa `MonthlyCharges` rata-rata untuk Fiber optic sebesar $91.4 jauh lebih tinggi dibandingkan DSL ($58.1). Harga yang tinggi dari layanan tersebut bisa saja menjadi pemicu pelanggan dengan `InternetService` Fiber optic untuk cenderung Churn.
- Secara berkala melakukan survey kepuasan pelanggan untuk mengetahui kualitas layanan yang telah diberikan dan memperbaiki jika ada reviu yang negatif.

Hal-hal yang bisa dilakukan untuk mengembangkan project dan modelnya lebih baik lagi diantaranya:
- Menambahkan fitur-fitur atau kolom baru yang berisi tingkat kepuasan pelanggan untuk masing-masing layanan, sehingga dapat diketahui lebih lanjut apakah churn disebabkan oleh kualitas layanan yang buruk atau tidak.
- Menambahkan fitur-fitur atau kolom baru yang berisi durasi atau biaya penggunaan produk-produk yang ada seperti panggilan suara, SMS, dan internet. Sehingga perusahaan dapat melakukan segmentasi pelanggan untuk menentukan jenis produk yang paling sesuai untuk ditawarkan.
- Melakukan penambahan data khususnya untuk kelas minoritas (Churn) agar dapat membantu meningkatkan performa model.
- Mencoba algorithm ML dan hyperparameter tuning yang berbeda (misal algoritma Logistic Regression, CatBoost, etc) serta menggunakan teknik oversampling yang berbeda selain SMOTENC. 
- Menganalisa data-data yang model yang masih salah tebak (False Negative dan False Positive) untuk mengetahui alasan dan karakteristiknya.
