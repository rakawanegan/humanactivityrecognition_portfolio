# Data sets

you can download here  

WISDM dataset

```
https://www.cis.fordham.edu/wisdm/dataset.php
```

![alt](./assets/axes-of-motion.png)

in their front pants leg pocket.  
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

**don't run each py files !**

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
├── assets: image folder
├── cnn1d_tf.py: main file(1d-cnn)
├── convbbt.py: main file(conv.backbone transformer)
├── optuna_convbbt.py: main file(conv.backbone transformer with optuna)
├── vit1d.py: main file(1d-vision transformer)
├── optuna_vit1d.py: main file(1d-vision transformer with optuna)
├── data: WISDM dataset
│   ├── readme.txt
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
│       │   ├── cross-tab.png
│       │   └── lab_notebook.md # result of experiment
│       └── raw
│           ├── main.py
│           ├── experiment.log
│           ├── lib
│           │   ├── model.py
│           │   ├── preprocess.py
│           │   └── result_process.py
│           ├── param.pkl
│           └── predict.csv
└── run.py: run script
</pre>

# MIT License

Copyright (C) 2023 NakagawaRen

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

