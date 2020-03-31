# Bölüm-01 Mahrem Derin Öğrenmenin Temel Araçları (draft version!)

PySyft'in gizliliğin korunması, dağıtık derin öğrenme ile ilgili eğitimine hoş geldiniz. Bu seri gizli veri ya da modeller üzerinde güvenli derin öğrenme yapabilmek için ihtiyaç duyacağınız yeni araç ve teknikleri adım adım öğrenmenizi sağlayacak bir kılavuzdur. 

**Kapsam:** Sadece veriyi nasıl dağıtık hale getirebiliriz ya da şifreleyebiliriz hususlarını konuşmayacağız; ayrıca verinin depolandığı ve sorgulandığı veri tabanları ve verilerden bilgi elde etmek için kullanılan modeller de dahil olmak üzere, Pysft'in veri etrafındaki tüm ekosistemi nasıl dağıtık hale getirmeye  yardımcı olduğunu da konuşacağız.

Yazarlar :
- Andrew Trask-Twitter: @iamtrask

Çeviri :

- Zumrut Muftuoglu : @zmuftuoglu

## Ana Hat :
- Bölüm 01: Mahrem Derin Öğrenmenin Temel Araçları

## Bu eğitimi niçin almalıyım?

**1) Rekabetçi Kariyer Avantajı:** Son 20 yıldır,dijital devrim ile analog süreçler sayısallaştıkça ,veri giderek daha da erişilebilir hal aldı. Bununla birlikte GDPR (ülkemizde KVKK) gibi yeni düzenlemelerle, işletmeler kişisel bilgileri nasıl kullandıkları ve daha da önemlisi nasıl analiz ettikleri konusunda daha az özgürlüğe sahip olma baskısı altındadır.**Dip not:** Veri Bilimciler; "eski okul" araçlarıyla çok fazla veriye erişemeyecekler, fakat mahrem derin öğrenme araçlarını öğrenerek,bu dönemecin önünde yerinizi alabilir ve kariyerinizde rekabet avantajı elde edebilirsiniz.   

**2) Girişimci Fırsatlar :** Toplumda Derin Öğrenme'nin çözebileceği bir dizi sorun vardır, ancak en önemlilerinin çoğu araştırılmamıştır çünkü insanlar hakkında inanılmaz derecede hassas bilgilere erişim gerektirecektir (zihinsel veya ilişki sorunları olan insanlara yardımcı olmak için Derin Öğrenme'yi kullanmayı düşünün !). Böylece, Özel Derin Öğrenme öğrenmek sizin için daha önce bu araç setleri olmadan başkaları tarafından kullanılabilir olmayan bir dizi yeni başlangıç fırsatlarının kilidini açar.

**3)Sosya Fayda :**  Derin Öğrenme, gerçek dünyada çok çeşitli sorunları çözmek için kullanılabilir,ancak kişisel veriler üzerinde derin öğrenme, insanlar için, insanlar hakkında derin öğrenmedir. Sahip olmadığınız veriler üzerinde Derin Öğrenme'nin nasıl yapılacağını öğrenmek, bir kariyer veya girişimci fırsattan daha fazlasını temsil eder, insanların yaşamlarındaki en kişisel ve önemli sorunların bazılarının çözülmesine yardımcı olma ve bunu ölçekli olarak yapma fırsatıdır.

## Nasıl ekstra kredi kazanabilirim?
- Github'ta Pysyft reposunu yıldızlamayı unutmayın! -  https://github.com/OpenMined/PySyft
- Bir Youtube eğitim videosu hazırlamaya ne dersiniz?

...hadi başlayalım!

## Ön Şartlar : 
- Pytorch' u tanımak-eğer tanımıyorsanız linki tıklayabilirsiniz https://www.fast.ai/
- Pysyft çerçevesini anlatan makaleyi okuyun! https://arxiv.org/pdf/1811.04017.pdf Bu makale Pysyft'in nasıl inşa edildiği hakkında hakkında detaylı bir arka plan vererek, işlerin daha anlamlı olmasına yardımcı olacaktır.

## Kurulum :

Başlamak için, doğru kurulumlar yaptığınıza emin olmalısınız.Bunun için Pysyft 'readme' dosyasına giderek talimatları takip edebilirsiniz. Ya da özetle aşağıdaki adımları takip edebilirsiniz :

- Python 3.6 veya daha yukarı bir sürümü kurunuz
- Pytorch 1.3 kurun
- Pysyft' i kopyalayın (git clone https://github.com/OpenMined/PySyft.git)
- Aşağıdaki komutları çalıştırın:
    cd PySyft
    pip install -r pip-dep/requirements.txt
    pip install -r pip-dep/requirements_udacity.txt
    python setup.py install udacity
    python setup.py test

Herhangi bir komut çalışmaz ise(veya testlerbaşarısız olursa)- kurulum yardımı için ilk olarak README dosyasını kontrol ediniz ve sonra Gitgub' ta görev yönetimi (issues) açın veya slack grubumuzda #beginner kanalımızda paylaşın! slack.openmined.org
