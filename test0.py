import os
import re

# Folder containing all downloaded parts
parts_folder = r"C:\Users\Kiran\Downloads"

# Final merged file
output_file = os.path.join(parts_folder, "hi_mr_translation_model.zip")

# Get valid part files only
part_files = []

for f in os.listdir(parts_folder):

    # Ignore duplicate "(1)" files
    if "(1)" in f:
        continue

    # Match model_part_X
    if re.match(r"model_part_\d+", f):
        part_files.append(f)

# Sort numerically
part_files.sort(key=lambda x: int(re.findall(r"\d+", x)[0]))

print("\nFiles to merge:\n")
for p in part_files:
    print(p)

# Merge parts
with open(output_file, "wb") as outfile:

    for part in part_files:

        part_path = os.path.join(parts_folder, part)

        print(f"\nMerging {part} ...")

        with open(part_path, "rb") as infile:

            while True:
                chunk = infile.read(1024 * 1024)

                if not chunk:
                    break

                outfile.write(chunk)

print("\nMerge complete.")
print(f"\nCreated:\n{output_file}")

# Verify final size
size_gb = os.path.getsize(output_file) / (1024**3)
print(f"\nFinal ZIP Size: {size_gb:.2f} GB")