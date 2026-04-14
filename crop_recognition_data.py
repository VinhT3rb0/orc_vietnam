import os
import json
import cv2
import numpy as np

archive_dir = "archive"
crop_dir = os.path.join(archive_dir, "crop_images")
os.makedirs(crop_dir, exist_ok=True)

def get_rotated_crop_image(img, points):
    points = np.array(points, dtype=np.float32)
    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    width = int(rect[1][0])
    height = int(rect[1][1])

    src_pts = points.astype("float32")
    # coordinate of the points in box points after the rectangle has been
    # straightened
    dst_pts = np.array([[0, 0],
                        [width - 1, 0],
                        [width - 1, height - 1],
                        [0, height - 1]], dtype="float32")

    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    # directly warp the rotated rectangle to get the straightened rectangle
    warped = cv2.warpPerspective(img, M, (width, height))
    return warped

def process_crop(det_file, rec_file):
    with open(det_file, 'r', encoding='utf-8') as f_det, open(rec_file, 'w', encoding='utf-8') as f_rec:
        for line in f_det:
            line = line.strip()
            if not line: continue
            img_path, labels = line.split('\t')
            labels = json.loads(labels)
            
            img = cv2.imread(img_path)
            if img is None: continue
            
            base_name = os.path.basename(img_path).split('.')[0]
            
            for i, label in enumerate(labels):
                transcription = label['transcription']
                # Bỏ qua những phần tử bị đánh dấu là ###
                if transcription == '###': continue
                
                points = label['points']
                try:
                    crop_img = get_rotated_crop_image(img, points)
                    crop_name = f"{base_name}_crop_{i}.jpg"
                    crop_path = os.path.join(crop_dir, crop_name)
                    
                    cv2.imwrite(crop_path, crop_img)
                    f_rec.write(f"{crop_path.replace(os.sep, '/')}\t{transcription}\n")
                except Exception as e:
                    print(f"Lỗi crop ảnh {img_path} - {i}: {e}")

print("Vui lòng đợi, quá trình crop sẽ mất vài phút...")
process_crop("train_det_list.txt", "train_rec_list.txt")
process_crop("val_det_list.txt", "val_rec_list.txt")
process_crop("test_det_list.txt", "test_rec_list.txt")
print("Hoàn thành crop data!")
