import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Step 1: Read target image
# =========================
TARGET_DIR = "Quiz1_Exercise/images/target"
SOURCE_DIR = "Quiz1_Exercise/images/source"

target_name = os.listdir(TARGET_DIR)[1]
target_path = TARGET_DIR + "/" + target_name

print(target_path)
target = cv2.imread(target_path)
target_rgb = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

# =========================
# Step 2: Read all source images
# =========================
source_data = []
source_names = os.listdir(SOURCE_DIR)

for file_name in source_names:
    file_path = SOURCE_DIR + "/" + file_name
    img = cv2.imread(file_path)
    source_data.append(img)

print("Target image :", target_name)
print("Jumlah source image :", len(source_data))

# =========================
# Step 3: Preprocessing target
# grayscale + Gaussian Blur
# =========================
gray_target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
gray_target = cv2.GaussianBlur(gray_target, (3, 3), 0)

# =========================
# Step 4: Prepare methods
# =========================
methods = ["SIFT", "AKAZE", "ORB"]
ratio_list = [0.45, 0.5, 0.55]

best_matches = 0
best_matches_data = None
best_method = None
best_ratio = None

# =========================
# Step 5: Try all methods
# =========================
for method in methods:
    print("\nMethod :", method)

    # pilih detector
    if method == "SIFT":
        detector = cv2.SIFT_create()
    elif method == "AKAZE":
        detector = cv2.AKAZE_create()
    elif method == "ORB":
        detector = cv2.ORB_create()

    # detect target keypoint & descriptor
    target_keypoint, target_descriptor = detector.detectAndCompute(gray_target, None)

    if target_descriptor is None:
        print("Descriptor target tidak ditemukan")
        continue

    # ubah tipe descriptor
    if method == "SIFT":
        target_descriptor = np.float32(target_descriptor) # agar sesuai dengan tipe data yang dibutuhkan FLANN untuk SIFT
    else:
        target_descriptor = np.uint8(target_descriptor) # agar sesuai dengan tipe data yang dibutuhkan FLANN untuk AKAZE dan ORB

    # coba beberapa nilai Lowe's ratio
    for ratio in ratio_list:
        print("  Lowe Ratio :", ratio)

        for index, img in enumerate(source_data):
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # preprocessing source
            gray_source = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_source = cv2.GaussianBlur(gray_source, (3, 3), 0)

            # detect source keypoint & descriptor
            source_keypoint, source_descriptor = detector.detectAndCompute(gray_source, None)

            if source_descriptor is None:
                continue

            # ubah tipe descriptor
            if method == "SIFT":
                source_descriptor = np.float32(source_descriptor)
            else:
                source_descriptor = np.uint8(source_descriptor)

            # FLANN matcher
            if method == "SIFT": # gunakan algoritma KD-Tree untuk SIFT
                flann = cv2.FlannBasedMatcher(
                    dict(algorithm=1, trees=5), # algorithm 1 untuk KD-Tree, trees itu jumlah pohon yang digunakan dalam indeks KD-Tree
                    dict(checks=50) # check itu jumlah pencarian yang dilakukan oleh FLANN, semakin tinggi semakin akurat tapi juga semakin lama
                )
            else:
                flann = cv2.FlannBasedMatcher( # gunakan algoritma LSH untuk AKAZE dan ORB
                    dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1), # algorithm 6 untuk LSH, table_number itu jumlah hash table yang digunakan, key_size itu ukuran hash key, multi_probe_level itu tingkat pencarian di sekitar hash key yang cocok
                    dict(checks=50) # sama kaya yg di atas
                )

            # matching
            matches = flann.knnMatch(target_descriptor, source_descriptor, k=2) # k=2 untuk mendapatkan 2 match terbaik untuk setiap keypoint target, yang dibutuhkan untuk Lowe's Ratio Test
            matchesmask = [[0, 0] for _ in range(len(matches))]

            current_matches = 0

            # Lowe's Ratio Test
            for i, match_pair in enumerate(matches):
                if len(match_pair) == 2:
                    fm, sm = match_pair
                    if fm.distance < ratio * sm.distance:
                        matchesmask[i] = [1, 0]
                        current_matches += 1

            print("   ", source_names[index], ":", current_matches, "good matches")

            # simpan hasil terbaik global
            if current_matches > best_matches:
                best_matches = current_matches
                best_method = method
                best_ratio = ratio
                best_matches_data = {
                    'image_data': img_rgb,
                    'keypoint': source_keypoint,
                    'descriptor': source_descriptor,
                    'match': matches,
                    'matchesmask': matchesmask,
                    'filename': source_names[index],
                    'target_keypoint': target_keypoint
                }

# =========================
# Step 6: Show best result
# =========================
if best_matches_data is not None:
    result = cv2.drawMatchesKnn(
        target_rgb,
        best_matches_data['target_keypoint'],
        best_matches_data['image_data'],
        best_matches_data['keypoint'],
        best_matches_data['match'],
        None, # karena kita tidak ingin menggambar semua matches, hanya yang lolos Lowe's Ratio Test
        matchesMask=best_matches_data['matchesmask'],
        matchColor=[255, 0, 0], # warna garis untuk matches yang lolos Lowe's Ratio Test (merah)
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS # agar hanya menggambar garis untuk matches yang lolos Lowe's Ratio Test, tanpa menggambar keypoint yang tidak match
    )

    plt.figure(figsize=(16, 8))
    plt.imshow(result)
    plt.title(
        "Best Match\n"
        + "Source Image: " + best_matches_data['filename']
        + " | Good Matches: " + str(best_matches)
        + " | Lowe Ratio: " + str(best_ratio)
        + " | Method: " + best_method
    )
    plt.axis('off')
    plt.show()

    print("\nBest Method       :", best_method)
    print("Best Lowe Ratio   :", best_ratio)
    print("Best Source Image :", best_matches_data['filename'])
    print("Good Matches      :", best_matches)
else:
    print("Tidak ada matching yang berhasil ditemukan")