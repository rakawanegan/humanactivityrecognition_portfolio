# Data sets

you can download [here](https://www.cis.fordham.edu/wisdm/dataset.php)  

WISDM dataset

```
https://www.cis.fordham.edu/wisdm/dataset.php
```

![alt](./assets/axes-of-motion.png)

device in their front pants leg pocket.  
x is perpendicular to the direction of travel  
y is perpendicular to the thigh  
z is the direction of travel  

![alt](./assets/plot-six-activities.png)

sitting		: const. y is always bigger than y, z.  
standing	: const. z is always bigger than x, y.  
walking		: 1/2 seconds cycle. x is always smaller than y, z.  
jogging		: 1/4 seconds cycle. jogging-y is smaller than walking-y.   
stairs-up	: 1/2 seconds cycle. y is always smaller than z, x.  
stairs-down	: 3/4 seconds cycle.  

# Usage

```python3
python3 run.py --path [py file name]  
```

- mkdir "%m%d_[py file name]"
- run main py file
> study.pkl: optuna study dict  
> model.pkl: pretrained model  
> param.pkl: freature parameter  
> predict.csv: concat predict and y_test  
> experiment.log: execution log file  
- run result_process
> cross_tab.png: using labnotebook picture  
> labnotebook.md: experiment description  

# Tree

<pre>
.
├── README.md: this file
├── assets: image folder using README.md
├── config.ini: you have to set this file refer to sample.ini
├── cnn1d_tf.py: main file(1d-cnn)
├── convbbt.py: main file(conv.backbone transformer)
├── optuna_convbbt.py: main file(conv.backbone transformer with optuna)
├── vit1d.py: main file(1d-vision transformer)
├── optuna_vit1d.py: main file(1d-vision transformer with optuna)
├── data: WISDM dataset
│   ├── readme.txt: this dataset's readme(original)
│   ├── WISDM_ar_v1.1.csv
│   ├── WISDM_ar_v1.1_raw_about.txt
│   └── WISDM_ar_v1.1_raw.txt
├── lib: local modules
│   ├── model.py
│   ├── preprocess.py
│   └── result_process.py
├── processor
│   ├── data_format.py: convert raw.txt to csv
│   └── format.py: format py file
├── result
│   └── MMDD_main
│       ├── processed
│       │   ├── assets: image folder using lab_notebook.md
│       │   └── lab_notebook.md ## result of experiment ##
│       └── raw
│           ├── experiment.log: runing log file
│           ├── main.py: gurantee experiment envirnment
│           ├── lib
│           │   ├── model.py
│           │   ├── preprocess.py
│           │   └── result_process.py
│           ├── param.pkl: experiment feature dict object
│           ├── model.pkl: pre-trained model object written in Pytorch
│           ├── study.pkl: optuna's study object
│           ├── x_test.pkl: test input data
│           ├── y_test.pkl: test answer data
│           └── predict.csv: contain index, predict value, answer value
└── run.py: run script
</pre>

# MIT License

Copyright (C) 2023 NakagawaRen

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

