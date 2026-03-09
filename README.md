Hastalıklı domates yaprağı görüntüsünden hastalığı tespit edip kullanıcıya o hastalıkla ilgli en güncel ve doğru çözüm önerileri veren bir web sitesi yapım aşaması. 
attık tohumu bakalım büyüyecek mi

10 Mart 26
-
Mobilnet modelini fine tuning etmiştim. PlantVillage veri setindeki domates verileriyle Mobilneti eğittim. 20 farklı değer kombinasyonu denedim en    yüksek doğruluk değerini veren deneyimin sonuçlarını best_model(1).pth dosyasında tuttum. Basit bir arayüz ile modelimin ilk tahminlerini elde ettim.
İlk başta bir sorun oluştu felaket derecesinde yanlış tahmin yaptı. Bunun sebebinin en başından yanlış parametrelerle eğittiğim modelimin olduğunu düşündüm. Ama aslında tüm mesele etiket kaymasından kaynaklanıyormuş. Pytorch klasörleri genellikle alfabetik sıralıyormuş ben direkt olarak modeli eğitirken kullandığım sırayla eklemiştim.

Peki şu an sorunlarımız neler?
- 
1- Modelimi daha çok denemeliyim bugün az denedik maalesef geç oldu biraz

2- Denediğim kadarıyla eğitim verimden kaynaklanan bir sorun var: tahminlerim gerçek hayattan kopuk (resmin arkasının beyaz değilde gürültü oluşturacak şekilde olması), sadece saf yaprak fotosu verilince iyi yapıyor. 

3- Hastalıklı olanları genel olarak yanlış tahmin ediyor bunun ana sebeplerinden biri yukarıda duruyor iki numara büyük ihtimalle o. Domates tarlası mı bulmam lazım şimdiiiikasdjfi 
  
