import os


label_dir = "C:/BAH_FINAL/labels/train"  
label_dir = "C:/BAH_FINAL/labels/val"

def filter_labels(label_dir, conf_thresh=0.5, max_box_area=0.3):
    for file in os.listdir(label_dir):
        if file.endswith(".txt"):
            new_lines = []
            with open(os.path.join(label_dir, file), "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = float(parts[0])
                        x, y, w, h = map(float, parts[1:5])
                        conf = float(parts[5]) if len(parts) == 6 else 1.0
                        if conf >= conf_thresh and (w * h) <= max_box_area:
                            new_lines.append(line + "\n")
            with open(os.path.join(label_dir, file), "w") as f:
                f.writelines(new_lines)

filter_labels(label_dir)
print("✅ Label filtering complete!")
