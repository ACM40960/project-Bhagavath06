import os

def concatenate_files(output_file_path, input_dir, part_prefix):
    with open(output_file_path, 'wb') as output_file:
        part_num = 0
        while True:
            part_filename = os.path.join(input_dir, f"{part_prefix}{part_num:04d}")
            if not os.path.exists(part_filename):
                break
            with open(part_filename, 'rb') as part_file:
                output_file.write(part_file.read())
            print(f"Added: {part_filename}")
            part_num += 1

if __name__ == "__main__":
    output_file_path = "faster-rcnn-checkpoints/best_model_weights.pth"
    input_dir = "faster-rcnn-checkpoints/"
    part_prefix = "best_model_weights.pth.part"

    concatenate_files(output_file_path, input_dir, part_prefix)
