import sys
file_path = sys.argv[1]
# ファイルからデータを読み取る
data_dict = {}
with open(file_path, "r") as f:
    for line in f:
        key, value = line.strip().split(": ")
        data_dict[key] = float(value)  # 数値として扱う
    
# 変数として扱う
pre_accuracy = data_dict["Accuracy"]
pre_pre_time = data_dict["Pre Time"]
changed_acc = accuracy - pre_accuracy
changed_pre = pre_time - pre_pre_time
if changed_acc < 0:
    result = "モデルの性能が劣化しました"
elif changed_acc == 0:
    result = "モデルの性能は変化しませんでした"
else:
    result = "モデルの性能が向上しました"

with open("compare.txt", "w") as f:
    f.write(f"Accuracy: {changed_acc:.4f}\n")
    f.write(f"Pre Time: {changed_pre:.6f} sec\n")
    f.write(f"結果: {result}\n")
