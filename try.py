import os

if __name__ == "__main__":
    # txt_path = r"F:\1234.txt"
    txt_path = r"F:\cpp_result.txt"
    with open(txt_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            line.replace('\n', '')
            a = float(line)
            if a >0.25 and a<1:
                print(a)