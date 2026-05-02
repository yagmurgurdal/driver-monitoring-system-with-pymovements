import pandas as pd
import re
import os
import glob

# Türkçe ay kısaltmaları ve sayısal karşılıkları
ay_sozlugu = {
    'Oca': '1', 'Şub': '2', 'Mar': '3', 'Nis': '4', 'May': '5', 'Haz': '6',
    'Tem': '7', 'Ağu': '8', 'Eyl': '9', 'Eki': '10', 'Kas': '11', 'Ara': '12'
}

def tarihi_sayiya_cevir(hucre_degeri):
    if pd.isna(hucre_degeri) or isinstance(hucre_degeri, (int, float)):
        return hucre_degeri
    
    hucre_metni = str(hucre_degeri).strip()
    eslesme = re.match(r"([A-ZŞİÖÇĞÜa-zşıöçğü]{3})[\.\-\s](\d{2})", hucre_metni)
    
    if eslesme:
        ay_kisaltmasi = eslesme.group(1).capitalize()
        yil_kismi = eslesme.group(2)
        
        if ay_kisaltmasi in ay_sozlugu:
            ay_sayisi = ay_sozlugu[ay_kisaltmasi]
            return float(f"{ay_sayisi}.{yil_kismi}")
            
    if isinstance(hucre_degeri, pd.Timestamp):
        return float(f"{hucre_degeri.month}.{hucre_degeri.strftime('%y')}")

    try:
        return float(hucre_metni.replace(',', '.'))
    except ValueError:
        return hucre_degeri

def klasordeki_dosyalarin_uzerine_yaz(klasor_yolu):
    """Klasördeki bozuk dosyaları düzeltir ve aynı dosyanın üzerine yazar."""
    dosya_yollari = glob.glob(os.path.join(klasor_yolu, "*.csv")) 
    
    for dosya in dosya_yollari:
        # Dosyayı oku
        df = pd.read_csv(dosya)
        
        # Pandas sürümlerine göre df.map() veya df.applymap() kullanılabilir
        try:
            df_duzeltilmis = df.map(tarihi_sayiya_cevir)
        except AttributeError:
            df_duzeltilmis = df.applymap(tarihi_sayiya_cevir)
        
        # Aynı dosya yolunu kullanarak orijinalin üzerine kaydet
        df_duzeltilmis.to_csv(dosya, index=False)
        print(f"Düzeltildi ve üzerine yazıldı: {os.path.basename(dosya)}")

# Kullanım Örneği:
# klasordeki_dosyalarin_uzerine_yaz("csv_dosyalarinin_bulundugu_klasor_yolu")