import cv2
import numpy as np
from ultralytics import YOLO
import os
import sys

class GlobalBlindOptimizer:
    def __init__(self, model_path='yolov8n.pt'):
        print("--- Inicjalizacja Modelu YOLO ---")
        self.model = YOLO(model_path)
        # Próg IoU (0.85 oznacza dużą precyzję, można zmniejszyć do 0.75 dla lepszej kompresji)
        self.iou_threshold = 0.85 

    def get_detections(self, image_source):
        """Uruchamia detekcję i zwraca sformatowaną listę obiektów."""
        # verbose=False wycisza logi w konsoli
        results = self.model(image_source, verbose=False)[0]
        detections = []
        for box in results.boxes:
            detections.append({
                'class': int(box.cls[0]),
                'conf': float(box.conf[0]),
                'box': box.xyxy[0].tolist()
            })
        return results, detections

    def show_resizable_window(self, title, results_object):
        """Wyświetla skalowalne okno z wynikami."""
        annotated_frame = results_object.plot()
        
        # Flaga cv2.WINDOW_NORMAL pozwala na zmianę rozmiaru okna myszką
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        
        # Ustawiamy domyślny, wygodny rozmiar (np. 800x600), ale user może go zmienić
        cv2.resizeWindow(title, 800, 600)
        
        cv2.imshow(title, annotated_frame)
        print(f"-> Otwarto okno: '{title}'. Naciśnij dowolny klawisz w oknie aby kontynuować...")
        cv2.waitKey(0)
        # Nie niszczymy okna od razu, żeby można było porównać oba na końcu

    def calculate_iou(self, boxA, boxB):
        """Matematyczne obliczenie nakładania się ramek (IoU)."""
        xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
        xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

    def verify_match(self, original_dets, new_dets):
        """
        Sprawdza czy wynik jest 'taki sam'.
        Warunki: ta sama liczba obiektów, te same klasy, pokrycie ramek > IoU.
        """
        if len(original_dets) != len(new_dets):
            return False
        
        matched_indices = set()
        for orig in original_dets:
            found_match = False
            for i, new in enumerate(new_dets):
                if i in matched_indices: continue
                
                # 1. Zgodność klasy (np. czy to nadal 'samochód')
                if orig['class'] != new['class']: continue
                
                # 2. Zgodność geometryczna (IoU)
                iou = self.calculate_iou(orig['box'], new['box'])
                if iou > self.iou_threshold:
                    found_match = True
                    matched_indices.add(i)
                    break
            
            if not found_match:
                return False
                
        return True

    def optimize_image(self, input_path, output_path):
        # 1. Wczytanie oryginału (Bitmapa)
        img_orig = cv2.imread(input_path)
        if img_orig is None: raise ValueError("Brak pliku input.bmp")
        
        # 2. Klasyfikacja REFERENCYJNA (nie wpływa na proces kompresji, służy tylko do weryfikacji)
        res_orig, gt_dets = self.get_detections(img_orig)
        print(f"Oryginał: Znaleziono {len(gt_dets)} obiektów.")
        
        # Wyświetlenie okna początkowego
        self.show_resizable_window("1. Obraz Oryginalny (Referencja)", res_orig)

        if not gt_dets:
            print("UWAGA: Brak obiektów na obrazie. Zapisuję z minimalną jakością.")
            # Jeśli pusto, robimy resize do 32px i jakość 1
            small = cv2.resize(img_orig, (32, 32))
            cv2.imwrite(output_path, small, [cv2.IMWRITE_WEBP_QUALITY, 1])
            return

        # 3. Pętla Optymalizacyjna (Global Search)
        # Szukamy najmniejszego pliku iterując po rozdzielczościach i jakości
        
        best_buffer = None
        min_size_bytes = float('inf')
        
        # Skale: zmniejszamy obraz globalnie. To daje największy zysk.
        # Od 100% (1.0) do 10% (0.1)
        scales = [1.0, 0.75, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2]
        
        print("\n--- Rozpoczynam poszukiwanie optymalnych parametrów kompresji ---")
        
        for scale in scales:
            # Obliczamy nowe wymiary
            h, w = img_orig.shape[:2]
            new_w, new_h = int(w * scale), int(h * scale)
            
            # Zabezpieczenie przed zbyt małym obrazem (YOLOv8 potrzebuje min ok. 32px)
            if new_w < 64 or new_h < 64: continue

            # Przeskalowanie obrazu (niezależne od detekcji)
            img_resized = cv2.resize(img_orig, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Binary Search dla jakości WebP (od 1 do 100)
            low, high = 1, 100
            current_best_q_for_scale = -1
            current_buffer = None

            while low <= high:
                mid_q = (low + high) // 2
                
                # Kompresja w pamięci RAM
                success, buffer = cv2.imencode('.webp', img_resized, [cv2.IMWRITE_WEBP_QUALITY, mid_q])
                if not success: break

                # Dekompresja
                img_reconstructed = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
                
                # Ponowna klasyfikacja (Weryfikacja)
                _, new_dets = self.get_detections(img_reconstructed)

                # Sprawdzenie warunku: Czy wynik jest identyczny z referencją?
                if self.verify_match(gt_dets, new_dets):
                    # Jest OK -> próbujemy pogorszyć jakość jeszcze bardziej (zmniejszyć Q)
                    current_best_q_for_scale = mid_q
                    current_buffer = buffer
                    high = mid_q - 1 # Szukamy w niższym zakresie
                else:
                    # Nie jest OK -> musimy polepszyć jakość
                    low = mid_q + 1

            # Po zakończeniu binary search dla danej skali:
            if current_buffer is not None:
                size = len(current_buffer)
                print(f"Skala {scale:.2f} | Min. Jakość {current_best_q_for_scale} -> Rozmiar: {size/1024:.2f} KB [OK]")
                
                if size < min_size_bytes:
                    min_size_bytes = size
                    best_buffer = current_buffer
            else:
                print(f"Skala {scale:.2f} -> Nie udało się zachować klasyfikacji nawet przy Q=100.")
                # Jeśli przy tej skali się nie udało, to przy mniejszych też się nie uda (zazwyczaj)
                # Możemy przerwać, żeby nie tracić czasu, lub kontynuować jeśli algorytm ma być 'exhaustive'
                # Tutaj: break dla optymalizacji czasu
                break

        # 4. Zapisz najlepszy wynik i wyświetl
        if best_buffer is not None:
            # Zapis na dysk
            with open(output_path, "wb") as f:
                f.write(best_buffer)

            # Statystyki
            orig_size = os.path.getsize(input_path)
            ratio = orig_size / min_size_bytes
            print(f"\n--- WYNIK KOŃCOWY ---")
            print(f"Oryginał: {orig_size/1024:.2f} KB")
            print(f"Po kompresji: {min_size_bytes/1024:.2f} KB")
            print(f"Stopień kompresji: {ratio:.1f}x")

            # Wizualizacja końcowa
            final_img = cv2.imdecode(best_buffer, cv2.IMREAD_COLOR)
            res_final, _ = self.get_detections(final_img)
            self.show_resizable_window(f"2. Wynik Kompresji ({min_size_bytes/1024:.1f} KB)", res_final)

        else:
            print("Nie znaleziono kompresji spełniającej warunki. Zapisuję oryginał jako JPEG HQ.")
            cv2.imwrite(output_path, img_orig, [cv2.IMWRITE_WEBP_QUALITY, 90])

        print("Zamykanie okien...")
        cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("No arguments provided! Please provide the image.")
        sys.exit(1)
    
    image = sys.argv[1]
    optimizer = GlobalBlindOptimizer()

    if os.path.exists(f"images/original/{image}.bmp"):
        optimizer.optimize_image(f"images/original/{image}.bmp", f"images/compressed/{image}_compressed.webp")
    else:
        print(f"Brak pliku {image}.bmp w katalogu ze zdjęciami images/original/.")