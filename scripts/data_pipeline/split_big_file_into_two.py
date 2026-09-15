import json
import os


def split_big_file_into_two(local_path: str):
    with open(local_path, "r", encoding="utf-8") as f_src:
        all_data = json.load(f_src)
        if not isinstance(all_data, list):
            print(f"Error: Expected a JSON list in {local_path}, but found {type(all_data)}.")
            return
        total_entries = len(all_data)
        split_index = total_entries // 2
        part1 = all_data[:split_index]
        part2 = all_data[split_index:]

        base_filename = os.path.basename(local_path)
        output_path1 = f"split1-{base_filename}"
        output_path2 = f"split2-{base_filename}"

        print(f"Writing {len(part1)} entries to {output_path1}...")
        with open(output_path1, "w", encoding="utf-8") as f_dest1:
            json.dump(part1, f_dest1, ensure_ascii=False, indent=4)

        print(f"Writing {len(part2)} entries to {output_path2}...")
        with open(output_path2, "w", encoding="utf-8") as f_dest2:
            json.dump(part2, f_dest2, ensure_ascii=False, indent=4)

        print(f"Successfully split '{local_path}' into '{output_path1}' and '{output_path2}'.")

if __name__ == "__main__":
    file_path = './fetched-wiki-bg-geo-po-oblast.json'
    split_big_file_into_two(file_path)
