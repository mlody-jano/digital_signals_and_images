import numpy as np
import cv2
import pywt
import sys
import os
import pickle
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
                encoded.append((int(prev), int(count)))
                prev = data[i]
                count = 1
        encoded.append((int(prev), int(count)))
        return encoded

    def rle_decode(self, encoded):
        """Dekodowanie RLE"""
        decoded = []
        for value, count in encoded:
            decoded.extend([value] * count)
        return np.array(decoded, dtype=np.uint8)

    # --- 2. KODOWANIE HUFFMANA ---
    def build_huffman_tree(self, data):
        """Buduje drzewo Huffmana i zwraca słownik kodów"""
        if len(data) == 0: return {}
        freq = Counter(data)
        
        if len(freq) == 1:
            # Specjalny przypadek: tylko jeden symbol
            return {list(freq.keys())[0]: '0'}
        
        heap = [[weight, [symbol, ""]] for symbol, weight in freq.items()]
        heapq.heapify(heap)
        
        while len(heap) > 1:
            lo = heapq.heappop(heap)
            hi = heapq.heappop(heap)
            for pair in lo[1:]: pair[1] = '0' + pair[1]
            for pair in hi[1:]: pair[1] = '1' + pair[1]
            heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
        
        huff_dict = {p[0]: p[1] for p in heapq.heappop(heap)[1:]}
        return huff_dict

    def huffman_encode(self, data, huff_dict):
        """Koduje dane używając słownika Huffmana"""
        bitstring = ''.join(huff_dict[symbol] for symbol in data)
        return bitstring

    def huffman_decode(self, bitstring, huff_dict):
        """Dekoduje bitstring używając odwróconego słownika"""
        reverse_dict = {v: k for k, v in huff_dict.items()}
        decoded = []
        current = ""
        for bit in bitstring:
            current += bit
            if current in reverse_dict:
                decoded.append(reverse_dict[current])
                current = ""
        return decoded

    def bitstring_to_bytes(self, bitstring):
        """Konwertuje string bitów na bajty"""
        # Padding do wielokrotności 8
        padding = (8 - len(bitstring) % 8) % 8
        bitstring = bitstring + '0' * padding
        
        byte_array = bytearray()
        for i in range(0, len(bitstring), 8):
            byte = bitstring[i:i+8]
            byte_array.append(int(byte, 2))
        
        return bytes(byte_array), padding

    def bytes_to_bitstring(self, byte_data, padding):
        """Konwertuje bajty z powrotem na string bitów"""
        bitstring = ''.join(format(byte, '08b') for byte in byte_data)
        # Usuń padding
        if padding > 0:
            bitstring = bitstring[:-padding]
        return bitstring

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

    def compress_image(self, image_path, scale=0.2, dwt_thresh=80, quantization=16):
        """Kompresuje obraz i zwraca skompresowane dane"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Nie można wczytać obrazu: {image_path}")
        
        original_shape = img.shape
        
        # Preprocessing (Anscombe + DWT)
        img_ans = 2 * np.sqrt(img.astype(np.float32) + 3/8)
        img_dwt = self.apply_dwt_compression(img_ans, threshold=dwt_thresh)
        img_denoised = (np.square(img_dwt / 2) - 3/8).clip(0, 255).astype(np.uint8)
        
        # Skalowanie
        small_img = cv2.resize(img_denoised, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        small_shape = small_img.shape
        
        # Kwantyzacja i spłaszczenie
        flat_data = (small_img // quantization).flatten()
        
        # RLE
        rle_data = self.rle_encode(flat_data)
        
        # Huffman - tworzymy symbole z par RLE
        rle_symbols = [f"{v}-{c}" for v, c in rle_data]
        huff_dict = self.build_huffman_tree(rle_symbols)
        bitstring = self.huffman_encode(rle_symbols, huff_dict)
        
        # Konwersja na bajty
        byte_data, padding = self.bitstring_to_bytes(bitstring)
        
        # Pakiet skompresowanych danych
        compressed_data = {
            'byte_data': byte_data,
            'padding': padding,
            'huff_dict': huff_dict,
            'original_shape': original_shape,
            'small_shape': small_shape,
            'quantization': quantization,
            'wavelet': self.wavelet,
            'dwt_thresh': dwt_thresh,
            'scale': scale
        }
        
        return compressed_data, img

    def decompress_image(self, compressed_data):
        """Dekompresuje obraz ze skompresowanych danych"""
        # Rozpakowanie danych
        byte_data = compressed_data['byte_data']
        padding = compressed_data['padding']
        huff_dict = compressed_data['huff_dict']
        original_shape = compressed_data['original_shape']
        small_shape = compressed_data['small_shape']
        quantization = compressed_data['quantization']
        
        # Dekodowanie Huffmana
        bitstring = self.bytes_to_bitstring(byte_data, padding)
        rle_symbols = self.huffman_decode(bitstring, huff_dict)
        
        # Odtworzenie par RLE
        rle_data = []
        for symbol in rle_symbols:
            parts = symbol.split('-')
            value = int(parts[0])
            count = int(parts[1])
            rle_data.append((value, count))
        
        # Dekodowanie RLE
        flat_data = self.rle_decode(rle_data)
        
        # Odkwantyzacja
        flat_data = flat_data * quantization
        
        # Reshape do małego obrazu
        small_img = flat_data.reshape(small_shape)
        
        # Skalowanie do oryginalnego rozmiaru
        reconstructed = cv2.resize(small_img, (original_shape[1], original_shape[0]), 
                                   interpolation=cv2.INTER_CUBIC)
        
        return reconstructed

    def save_compressed(self, compressed_data, output_path):
        """Zapisuje skompresowane dane do pliku"""
        with open(output_path, 'wb') as f:
            pickle.dump(compressed_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_compressed(self, input_path):
        """Wczytuje skompresowane dane z pliku"""
        with open(input_path, 'rb') as f:
            return pickle.load(f)

    def run_full_pipeline(self, image_path, output_dir='./images/compressed/', 
                         scale=0.2, dwt_thresh=80):
        """Pełny pipeline: kompresja, zapis, YOLO"""
        # Tworzenie katalogu wyjściowego
        os.makedirs(output_dir, exist_ok=True)
        
        # Oryginalny rozmiar
        orig_size = os.path.getsize(image_path)
        
        # YOLO przed kompresją
        img_original = cv2.imread(image_path)
        res_before = self.model(img_original, verbose=False)
        
        # Kompresja
        compressed_data, img = self.compress_image(image_path, scale, dwt_thresh)
        
        # Ścieżka zapisu
        base_name = os.path.basename(image_path)
        file_name, _ = os.path.splitext(base_name)
        compressed_path = os.path.join(output_dir, f"{file_name}_compressed.pkl")
        
        # Zapis skompresowanych danych
        self.save_compressed(compressed_data, compressed_path)
        compressed_size = os.path.getsize(compressed_path)
        
        # Dekompresja i rekonstrukcja
        reconstructed = self.decompress_image(compressed_data)
        
        # YOLO po dekompresji
        res_after = self.model(reconstructed, verbose=False)
        
        # Zapis zrekonstruowanego obrazu (dla porównania wizualnego)
        reconstructed_path = os.path.join(output_dir, f"{file_name}_reconstructed.png")
        cv2.imwrite(reconstructed_path, reconstructed)
        
        return {
            'original_size': orig_size,
            'compressed_size': compressed_size,
            'compressed_path': compressed_path,
            'reconstructed_path': reconstructed_path,
            'compression_ratio': orig_size / compressed_size,
            'res_before': res_before,
            'res_after': res_after,
            'reconstructed_img': reconstructed
        }


if __name__ == "__main__":
    if len(sys.argv) < 2: 
        print("No image file given. Please provide image path!")
        sys.exit(1)
    
    input_path = sys.argv[1]
    
    if not os.path.exists(input_path):
        print(f"File not exists: {input_path}")
        sys.exit(1)
    
    opt = WaveletOptimizer()
    
    print("=" * 60)
    print("IMAGE COMPRESSION")
    print("=" * 60)
    
    results = opt.run_full_pipeline(input_path, scale=0.2, dwt_thresh=80)
    
    print(f"\nCOMPRESSION RESULTS:")
    print(f"{'─' * 60}")
    print(f"Original:      {results['original_size'] / 1024:.2f} KB")
    print(f"Compressed:   {results['compressed_size'] / 1024:.2f} KB")
    print(f"Compression level:    {results['compression_ratio']:.2f}:1")
    print(f"{'─' * 60}")
    print(f"\nSAVED FILES:")
    print(f"{'─' * 60}\n")
    print(f"  • Compressed (raw data):                 {results['compressed_path']}")
    print(f"  • Reconstructed (from compressed data):  {results['reconstructed_path']}")
    print(f"{'─' * 60}\n")
    
    # Wizualizacja YOLO
    print("Showing detection results...")
    results['res_before'][0].show()
    results['res_after'][0].show()
    
    print("\nCompression finished!")