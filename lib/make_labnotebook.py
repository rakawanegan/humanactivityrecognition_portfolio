import datetime

def create_experiment_memo(title, content):
    current_date = datetime.datetime.now().strftime("%Y%m%d")
    file_name = f"{current_date}_{title}.md"
    
    with open(file_name, "w") as f:
        f.write(content)

def generate_experiment_memo(title, experiment_info):
    memo_content = f"# 実験メモ\n\n## 日付\n{datetime.datetime.now().strftime('%Y年%m月%d日')}\n"
    
    for key, value in experiment_info.items():
        memo_content += f"\n## {key}\n{value}\n"

    create_experiment_memo(title, memo_content)


# Usage
if "__name__" == "__main__":
	# 実験メモのタイトルと内容を指定
	memo_title = "My Experiment"
	experiment_info = {
 	   "目的": "機械学習モデルの性能評価",
  	  "使用データセット": "MNIST",
  	  "モデル": "ニューラルネットワーク",
  	  "ハイパーパラメータ": "学習率: 0.001, バッチサイズ: 32, エポック数: 100",
  	  "実験結果": "正解率: 0.95, 損失関数の値: 0.2",
  	  "考察と課題": "モデルの学習が収束しきっていない可能性がある。",
   	 "追加実験の計画": "学習率を変更して再実験を行う。"
	}
	# 実験メモを生成
	generate_experiment_memo(memo_title, experiment_info)
