# CapstoneProjectModule3_MachineLearning
Folder ini berisi Capstone Project Purwadhika BSD membuat Machine Learning

## CPM3_mzm.ipynb
### BUSINESS UNDERSTANDING

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
