import os
import json

# Cấu hình đường dẫn
archive_dir = "archive"
labels_dir = os.path.join(archive_dir, "labels")

def get_image_dir(idx):
    if idx <= 1200:
        return os.path.join(archive_dir, "train_images")
    elif idx <= 1500:
        return os.path.join(archive_dir, "test_image")
    else:
        return os.path.join(archive_dir, "unseen_test_images")

# File đầu ra
train_det_file = "train_det_list.txt"
val_det_file = "val_det_list.txt"
test_det_file = "test_det_list.txt"

train_labels = []
val_labels = []
test_labels = []

label_files = os.listdir(labels_dir)
for lf in label_files:
    idx_str = lf.split('_')[1].split('.')[0]
    idx = int(idx_str)
    if idx <= 1200:
        train_labels.append(lf)
    elif idx <= 1500:
        val_labels.append(lf)
    else:
        test_labels.append(lf)

def process_detection(label_list, output_file):
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for label_file in label_list:
            idx = int(label_file.split('_')[1].split('.')[0])
            img_filename = f"im{idx:04d}.jpg"
            img_dir = get_image_dir(idx)
            img_path = os.path.join(img_dir, img_filename)
            
            if not os.path.exists(img_path):
                print(f"Không tìm thấy ảnh {img_path}")
                continue
                
            label_path = os.path.join(labels_dir, label_file)
            box_infos = []
            with open(label_path, 'r', encoding='utf-8-sig') as f_in:
                lines = f_in.readlines()
                for line in lines:
                    line = line.strip()
                    if not line: continue
                    parts = line.split(',')
                    if len(parts) >= 9:
                        try:
                            x1, y1 = float(parts[0]), float(parts[1])
                            x2, y2 = float(parts[2]), float(parts[3])
                            x3, y3 = float(parts[4]), float(parts[5])
                            x4, y4 = float(parts[6]), float(parts[7])
                            
                            transcription = ",".join(parts[8:])
                            points = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                            box_infos.append({
                                "transcription": transcription,
                                "points": points
                            })
                        except ValueError:
                            print(f"Lỗi parse tọa độ dòng: {line}")
            
            if len(box_infos) > 0:
                # Ghi định dạng path cần convert backslashes về forward slashes cho an toàn cross-platform
                f_out.write(f"{img_path.replace(os.sep, '/')}\t{json.dumps(box_infos, ensure_ascii=False)}\n")

print("Đang tạo file detection annotations...")
process_detection(train_labels, train_det_file)
process_detection(val_labels, val_det_file)
process_detection(test_labels, test_det_file)
print("Hoàn thành!")
