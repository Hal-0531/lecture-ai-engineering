import os
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import random
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from mlflow.models.signature import infer_signature
import time
import sys

# データ準備
def prepare_data(test_size=0.2, random_state=42):
    # Titanicデータセットの読み込み
    path = "day5/演習1/data/Titanic.csv"
    data = pd.read_csv(path)

    # 必要な特徴量の選択と前処理
    data = data[["Pclass", "Sex", "Age", "Fare", "Survived"]].dropna()
    data["Sex"] = LabelEncoder().fit_transform(data["Sex"])  # 性別を数値に変換

    # 整数型の列を浮動小数点型に変換
    data["Pclass"] = data["Pclass"].astype(float)
    data["Sex"] = data["Sex"].astype(float)
    data["Age"] = data["Age"].astype(float)
    data["Fare"] = data["Fare"].astype(float)
    data["Survived"] = data["Survived"].astype(float)

    X = data[["Pclass", "Sex", "Age", "Fare"]]
    y = data["Survived"]

    # データ分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


# 学習と評価
def train_and_evaluate(
    X_train, X_test, y_train, y_test, n_estimators=100, max_depth=None, random_state=42
):
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
    )
    model.fit(X_train, y_train)
    start_time = time.time()
    predictions = model.predict(X_test)
    end_time = time.time()
    pre_time = end_time - start_time  # 秒単位
    accuracy = accuracy_score(y_test, predictions)
    return model, accuracy, pre_time


# モデル保存
def log_model(model, accuracy, pre_time, params):
    with mlflow.start_run():
        # パラメータをログ
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)

        # メトリクスをログ
        mlflow.log_metric("accuracy", accuracy)

        # モデルのシグネチャを推論
        signature = infer_signature(X_train, model.predict(X_train))

        # モデルを保存
        mlflow.sklearn.log_model(
            model,
            "model",
            signature=signature,
            input_example=X_test.iloc[:5],  # 入力例を指定
        )
        # accurecyとparmsは改行して表示
        #print(f"モデルのログ記録値 \naccuracy: {accuracy}\npre_time: {pre_time}\nparams: {params}")


# メイン処理
if __name__ == "__main__":
    # ランダム要素の設定
    test_size = round(
        random.uniform(0.1, 0.3), 2
    )  # 10%〜30%の範囲でテストサイズをランダム化
    data_random_state = random.randint(1, 100)
    model_random_state = random.randint(1, 100)
    n_estimators = random.randint(50, 200)
    max_depth = random.choice([None, 3, 5, 10, 15])

    # パラメータ辞書の作成
    params = {
        "test_size": test_size,
        "data_random_state": data_random_state,
        "model_random_state": model_random_state,
        "n_estimators": n_estimators,
        "max_depth": "None" if max_depth is None else max_depth,
    }

    # データ準備
    X_train, X_test, y_train, y_test = prepare_data(
        test_size=test_size, random_state=data_random_state
    )

    # 学習と評価
    model, accuracy, pre_time = train_and_evaluate(
        X_train,
        X_test,
        y_train,
        y_test,
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=model_random_state,
    )
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
    
    with open("result.txt", "w") as f:
        f.write(f"精度の変化: {changed_acc:.4f}\n")
        f.write(f"推論時間の変化: {changed_pre:.6f} sec\n")
        f.write(f"結果: {result}\n")
    # 結果を保存
    with open("result.txt", "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Pre Time: {pre_time:.6f}\n")

    #print("Execution complete. Check result.txt for details.")
    # モデル保存
    log_model(model, accuracy, pre_time, params)

    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"titanic_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    #print(f"モデルを {model_path} に保存しました")

    
