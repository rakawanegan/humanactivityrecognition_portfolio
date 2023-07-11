def create_experiment_memo(dir, title, date, content):
    file_name = f"{date.strftime('%Y%m%d')}_{title}.md"

    with open(file_name, "w") as f:
        f.write(content)


def generate_experiment_memo(dir, title, date, experiment_info):
    memo_content = f"# 実験メモ\n\n## 日付\n{date}\n"

    for key, value in experiment_info.items():
        memo_content += f"\n## {key}\n{value}\n"

    create_experiment_memo(dir, title, memo_content)


# Usage
if "__name__" == "__main__":
    import datetime

    dir = "./result"
    # 実験メモの日時、タイトル、内容を指定
    date = datetime.datetime.now()
    memo_title = "My Experiment"
    experiment_info = {
        "目的": "機械学習モデルの性能評価",
        "使用データセット": "MNIST",
        "モデル": "ニューラルネットワーク",
        "ハイパーパラメータ": "学習率: 0.001, バッチサイズ: 32, エポック数: 100",
        "実験結果": "正解率: 0.95, 損失関数の値: 0.2",
        "考察と課題": "モデルの学習が収束しきっていない可能性がある。",
        "追加実験の計画": "学習率を変更して再実験を行う。",
    }
    # 実験メモを生成
    generate_experiment_memo(dir, memo_title, date.strftime("%Y年%m月%d日"), experiment_info)
