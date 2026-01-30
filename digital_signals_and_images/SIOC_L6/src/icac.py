import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os
import sys

class UltraOptimizer:
    def __init__(self, model_path='yolov8n.pt'):
        print("Ładowanie modelu YOLO...")
        self.model = YOLO(model_path)
        self.iou_threshold = 0.85 

    def get_results(self, image_source):
        """Uruchamia YOLO i zwraca obiekt wyników (Results)."""
        return self.model(image_source, verbose=False)[0]

    def parse_detections(self, results):
        """Wyciąga listę detekcji z obiektu Results."""
        detections = []
        for box in results.boxes:
            detections.append({
                'class': int(box.cls[0]),
                'conf': float(box.conf[0]),
                'box': box.xyxy[0].tolist()
            })
        return detections

    def show_window(self, title, results):
        """Wyświetla okno z obrazem i naniesionymi ramkami YOLO."""
        # Metoda .plot() tworzy tablicę numpy z narysowanymi ramkami
        annotated_frame = results.plot()
        cv2.imshow(title, annotated_frame)
        print(f"Wyświetlono okno: {title}. Naciśnij dowolny klawisz w oknie, aby kontynuować...")
        cv2.waitKey(0) # Czeka na reakcję użytkownika

    def calculate_iou(self, boxA, boxB):
        xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
        xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

    def verify_consistency(self, original_dets, new_dets):
        if len(original_dets) != len(new_dets): return False
        matches = 0
        used_indices = set()
        for orig in original_dets:
            found = False
            for idx, comp in enumerate(new_dets):
                if idx in used_indices: continue
                if orig['class'] != comp['class']: continue
                if self.calculate_iou(orig['box'], comp['box']) > self.iou_threshold:
                    found = True
                    used_indices.add(idx)
                    break
            if found: matches += 1
        return matches == len(original_dets)

    def apply_smart_blur(self, img_array, detections):
        mask = np.zeros(img_array.shape[:2], dtype=np.uint8)
        for det in detections:
            x1, y1, x2, y2 = map(int, det['box'])
            pad = 15
            cv2.rectangle(mask, (max(0, x1-pad), max(0, y1-pad)), (x2+pad, y2+pad), 255, -1)
        blurred_img = cv2.GaussianBlur(img_array, (51, 51), 0)
        mask_3ch = cv2.merge([mask, mask, mask])
        return np.where(mask_3ch > 0, img_array, blurred_img)

    def optimize(self, input_path, output_path):
        original_img = cv2.imread(input_path)
        if original_img is None: raise ValueError("Nie można otworzyć pliku")
        
        # 1. Klasyfikacja początkowa i wyświetlenie okna
        res_orig = self.get_results(original_img)
        gt_dets = self.parse_detections(res_orig)
        self.show_window("Klasyfikacja PRZED kompresja (BMP)", res_orig)
        
        if not gt_dets:
            print("Brak obiektów do ochrony.")
            return

        # 2. Pre-processing i Optymalizacja
        processed_img = self.apply_smart_blur(original_img, gt_dets)
        best_buffer = None
        min_size = float('inf')
        
        # Uproszczona pętla dla demonstracji (Skala i Jakość)
        for scale in [1.0, 0.5, 0.25]:
            low, high = 1, 80
            while low <= high:
                mid_q = (low + high) // 2
                h, w = processed_img.shape[:2]
                resized = cv2.resize(processed_img, (int(w*scale), int(h*scale)))
                _, buffer = cv2.imencode('.webp', resized, [cv2.IMWRITE_WEBP_QUALITY, mid_q])
                
                reconstructed = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
                new_res = self.get_results(reconstructed)
                new_dets = self.parse_detections(new_res)
                
                if self.verify_consistency(gt_dets, new_dets):
                    best_buffer = buffer
                    min_size = len(buffer)
                    high = mid_q - 1
                else:
                    low = mid_q + 1

        # 3. Klasyfikacja końcowa i wyświetlenie okna
        if best_buffer is not None:
            with open(output_path, "wb") as f:
                f.write(best_buffer)
            
            final_img = cv2.imdecode(best_buffer, cv2.IMREAD_COLOR)
            res_final = self.get_results(final_img)
            
            # Statystyki
            orig_kb = os.path.getsize(input_path) / 1024
            final_kb = len(best_buffer) / 1024
            print(f"\nSukces! Rozmiar spadl z {orig_kb:.1f}KB do {final_kb:.1f}KB")
            
            self.show_window(f"Klasyfikacja PO kompresji (WebP) - {final_kb:.1f}KB", res_final)
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Użycie: python icac.py <nazwa_pliku_bez_rozszerzenia>")
        sys.exit(1)
    
    image = sys.argv[1]
    optimizer = UltraOptimizer()

    if os.path.exists(f"images/original/{image}.bmp"):
        optimizer.optimize(f"images/original/{image}.bmp", f"images/compressed/{image}_compressed.webp")
    else:
        print(f"Brak pliku {image}.bmp w katalogu ze zdjęciami images/original/.")