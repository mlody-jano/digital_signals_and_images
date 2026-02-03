import numpy as np
import cv2
import pywt
import sys
import os
from collections import Counter
import heapq
from ultralytics import YOLO

class WaveletOptimizer:
    def __init__(self, model_name='yolov8n.pt'):
        self.model = YOLO(model_name)
        self.wavelet = 'db1'

    # --- 1. KODOWANIE RLE ---
    def rle_encode(self, data):
        if len(data) == 0: return []
        encoded = []
        prev = data[0]
        count = 1
        for i in range(1, len(data)):
            if data[i] == prev:
                count += 1
            else:
                encoded.append((prev, count))
                prev = data[i]
                count = 1
        encoded.append((prev, count))
        return encoded

    # --- 2. KODOWANIE HUFFMANA (Obliczanie bitów) ---
    def get_huffman_bit_count(self, data):
        if len(data) == 0: return 0
        freq = Counter(data)
        heap = [[weight, [symbol, ""]] for symbol, weight in freq.items()]
        heapq.heapify(heap)
        
        while len(heap) > 1:
            lo = heapq.heappop(heap)
            hi = heapq.heappop(heap)
            for pair in lo[1:]: pair[1] = '0' + pair[1]
            for pair in hi[1:]: pair[1] = '1' + pair[1]
            heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
        
        # Tworzymy słownik kody -> długość bitowa
        huff_dict = {p[0]: len(p[1]) for p in heapq.heappop(heap)[1:]}
        return sum(freq[symbol] * huff_dict[symbol] for symbol in freq)

    # --- PRZETWARZANIE OBRAZU ---
    def apply_dwt_compression(self, img, threshold=70):
        channels = cv2.split(img)
        processed = []
        for ch in channels:
            coeffs = pywt.wavedec2(ch, self.wavelet, level=2)
            new_coeffs = [coeffs[0]]
            for i in range(1, len(coeffs)):
                new_coeffs.append(tuple(pywt.threshold(c, threshold, mode='hard') for c in coeffs[i]))
            processed.append(pywt.waverec2(new_coeffs, self.wavelet))
        return cv2.merge(processed).astype(np.uint8)

    def run_pipeline(self, image_path, scale=0.2, dwt_thresh=80):
        img = cv2.imread(image_path)
        res_pre = self.model(img, verbose=False)
        
        # Preprocessing (Anscombe + DWT)
        img_ans = 2 * np.sqrt(img.astype(np.float32) + 3/8)
        img_dwt = self.apply_dwt_compression(img_ans, threshold=dwt_thresh)
        img_denoised = (np.square(img_dwt / 2) - 3/8).clip(0, 255).astype(np.uint8)
        
        # Interpolacja (Skalowanie)
        small_img = cv2.resize(img_denoised, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        
        # --- KLUCZOWY MOMENT: WŁASNE KODOWANIE ---
        # Spłaszczamy obraz do wektora i kwantujemy (np. do 16 poziomów)
        flat_data = (small_img // 16).flatten()
        
        # 1. RLE
        rle_data = self.rle_encode(flat_data)
        # 2. Huffman (na danych z RLE)
        # Traktujemy pary (wartość, licznik) jako symbole dla Huffmana
        total_bits = self.get_huffman_bit_count([f"{v}-{c}" for v, c in rle_data])
        custom_size_bytes = total_bits / 8

        # Rekonstrukcja dla YOLO
        recon_img = cv2.resize(small_img, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
        res_post = self.model(recon_img, verbose=False)
        
        return res_pre, res_post, recon_img, custom_size_bytes

if __name__ == "__main__":
    if len(sys.argv) < 2: 
        print("No input image. Please provide an image path.")
        sys.exit(1)
    
    input_path = sys.argv[1]
    opt = WaveletOptimizer()
    
    # Oryginalny rozmiar (RAW)
    orig_raw_size = os.path.getsize(input_path) 
    
    res_before, res_after, final_img, compressed_size = opt.run_pipeline(input_path)
    
    # Zapisywanie zrekonstruowanego obrazu
    file_name, file_ext = os.path.splitext(input_path)
    cv2.imwrite(f"{file_name}_reconstructed{file_ext}", final_img)

    print("-" * 40)
    print(f"KOMPRESJA RLE + HUFFMAN (Custom):")
    print(f"Oryginalny plik: {orig_raw_size / 1024:.2f} KB")
    print(f"Rozmiar bitstreamu: {compressed_size / 1024:.2f} KB")
    print(f"Stopień kompresji: {orig_raw_size / compressed_size:.2f}:1")
    print("-" * 40)
    
    # Wizualizacja YOLO
    res_before[0].show()
    res_after[0].show()