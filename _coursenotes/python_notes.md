---
title: "Miscellaneous notes: `python`, `numpy`, `pandas`, `sklearn`"
categories:
toc: true
layout: single
permalink: /coursenotes/python_notes/
author_profile: true
read_time: true
---

## `python`

## `numpy`

## `pandas`

* `pandas.DataFrame.reindex(labels)` allows you to reorder the index of a dataframe to the order dictated by `labels`. If the corresponding label existed in the original dataframe, it will slot that particular row containing that index there. If that index does not exist, it will insert `NaN` unless the `fill_value` argument is provided. 

* `pandas.DataFrame.sample(n=)` gives us a random sample of `n` rows in the dataframe. 
## `sklearn`
