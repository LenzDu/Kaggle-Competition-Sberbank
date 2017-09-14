# Kaggle-Competition-Sberbank
Top 1% rankings (22/3270) code sharing for Kaggle competition Sberbank Russian Housing Market: https://www.kaggle.com/c/sberbank-russian-housing-market

* Data.py: Data cleaning and feature engineering
* Exploration.py: Used for unorganized data exploration
* Model.py: Main model (xgboost)
* BaseModel.py: Base models for ensembling
* lightGBM.py: lightgbm model for ensembling
* Stacking.py: model stacking (final model)

Final pipeline: Raw data → Data.py → Stacking.py. \
Model.py, BaseModel.py and lightGBM.py are not used for the final output. \
Original data set can be downloaded from the competition page.
