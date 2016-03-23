# Project CatsDogs: Predictive analytics-model evaluation and selection - group2
author: yimin zhang,
#### - for cats versus dogs image data

Read [full project description](doc/project3_desc.md)

In this project, we will carry out model evaluation and selection for predictive analytics on image data. As data scientists, we often need to evaluate different modeling/analysis strategies and decide what is the best. Such decisions need to be supported with sound evidence in the form of model assessment, validation and comparison. In addition, we also need to communicate our decision and supporting evidence clearly and convincingly in an accessible fashion.

![image](https://i.ytimg.com/vi/8Ryo8Pf4NNE/hqdefault.jpg)

---
Following [suggestions](http://nicercode.github.io/blog/2013-04-05-projects/) by [RICH FITZJOHN](http://nicercode.github.io/about/#Team) (@richfitz). This folder is orgarnized as follows.

```
proj/
├── lib/
├── data/
├── doc/
├── figs/
└── output/
```

Please see each subfolder for a README file.


###  main.R 
the baseline model use svm and RGB for feature. predictor rate about 66%
the linear svm model use sift

### base_hsv
there is another baseline model with simillar predictor rate use HSV, this is reasonable because HSV is transfor from RGB.
there is also a model.R document, which has run different model use HSV for feature. But none of them work well. And the logistic regression shows that this feature has the linear boundary. So linear svm should work.



