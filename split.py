import os

def split_file(file_path, output_dir, chunk_size):
    file_size = os.path.getsize(file_path)
    part_num = 0

    with open(file_path, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            part_filename = os.path.join(output_dir, f"best_model_weights.pth.part{part_num:04d}")
            with open(part_filename, 'wb') as part_file:
                part_file.write(chunk)
            print(f"Created: {part_filename}")
            part_num += 1

if __name__ == "__main__":
    file_path = "faster-rcnn-checkpoints/best_model_weights.pth"
    output_dir = "faster-rcnn-checkpoints/"
    chunk_size = 50 * 1024 * 1024  # 50MB

    split_file(file_path, output_dir, chunk_size)

