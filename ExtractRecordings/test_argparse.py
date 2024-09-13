import argparse
import glob


def parse_inputs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--type", type=str, help="Specify recording type.")
    # va devoir aller chercher les fichiers .meta ou les fichiers .rhd (?)
    parser.add_argument("--path", type=str, help="Path to file(s). If left empty, will check in pwd.")
    return parser.parse_args()


def main():
    args = parse_inputs()
    recording_type = args.type
    path = args.path
    if recording_type == "npx":
        print("npx")
    elif recording_type == "intan":
        # extension.
        print("intan")
    else:
        print(f"Error: recording type unknown; {recording_type} is not available.")


if __name__ == "__main__":
    main()
