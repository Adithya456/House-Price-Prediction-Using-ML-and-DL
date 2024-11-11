# House Price Prediction

This project uses machine learning and deep learning to predict house prices based on a range of features such as size, location, and amenities. The project applies data preprocessing, feature engineering, exploratory data analysis, and several regression algorithms to develop an accurate prediction model for house pricing.

## Project Structure
- **Data Preprocessing**: Involves cleaning data, handling missing values, encoding categorical variables, feature scaling, and geospatial binning using K-means clustering.
- **Exploratory Data Analysis (EDA)**: Includes visualizations for feature distributions, correlation heatmaps, and insights on housing attributes.
- **Machine Learning and Deep Learning Models**: Various regression models, including Linear Regression, Decision Trees, Random Forest, Support Vector Machines (SVM), Gradient Boosting, Extreme Gradient Boosting (XGB), and a deep learning model Deep Neural Network(DNN), are trained and evaluated.

## Project Files
- **data/kc_house_data.csv**: The dataset containing attributes of houses such as bedrooms, bathrooms, square footage, etc.
- **notebooks/House_Price_Prediction.ipynb**: Contains Jupyter notebooks for EDA, preprocessing, model training, and evaluation.
- **report/House_Price_Prediction.pdf**: Detailed project report documenting the methodology, data analysis, model evaluation, and results.
- **README.md**: Project documentation (this file).

## Data Processing Steps
- **Cleaning**: Handle missing values, remove irrelevant features, and correct outliers.
- **Feature Engineering**: Extract new features from existing ones, including date-based attributes and geospatial clustering.
- **Encoding**: Convert categorical variables into numeric values.
- **Scaling**: Normalize features for better model performance using MinMax scaling.
- **Data Splitting**: Split data into training and testing sets, with K-fold cross-validation for model validation.

## Exploratory Data Analysis
Visualizations include:
- **Feature Distributions**: Histograms and count plots for features like price, bedrooms, and waterfront views.
- **Correlation Heatmap**: Shows the relationship between features and target (price).
- **Geospatial Clustering**: Grouping similar locations using K-means clustering for better analysis.

## Model Evaluation
Model evaluation was conducted using multiple preprocessing strategies to identify the best-performing configuration. The following evaluation approaches were tested:
- **Evaluation with All Features**: All features were included in model training, providing a baseline for model performance across various algorithms.
- **Evaluation with Feature Selection (Correlation Matrix)**: Features with low correlation to the target variable were removed to reduce complexity.
- **Evaluation with Transformed Target Variable**: The target variable (house price) was log-transformed to correct skewness, enhancing model accuracy.


## Machine Learning Models and Results

The following models were trained and evaluated using three different preprocessing methods. Hyperparameter tuning was performed using GridSearchCV to optimize each model’s performance. The table below shows the R² scores achieved by each model across these methods.

| Model                           | All Features | Feature Selection (Correlation Matrix) | Log-Transformed Target |
|---------------------------------|--------------|----------------------------------------|-------------------------|
| **Linear Regression**           | 0.67         | 0.62                                   | 0.73                    |
| **Decision Tree**               | 0.76         | 0.75                                   | 0.79                    |
| **Random Forest**               | 0.83         | 0.79                                   | 0.85                    |
| **Support Vector Machine (SVM)**| 0.84         | 0.79                                   | 0.83                    |
| **K-Nearest Neighbors (KNN)**   | 0.71         | 0.76                                   | 0.81                    |
| **Gradient Boosting**           | 0.84         | 0.79                                   | 0.84                    |
| **Extreme Gradient Boosting (XGB)** | 0.86     | 0.79                                   | 0.88                    |

### Summary of Results
- **All Features**: Using all features provided a strong baseline, with XGB achieving the highest R² score of 0.86.
- **Feature Selection (Correlation Matrix)**: This approach reduced model complexity by removing features with low correlation to the target variable. Although it simplified the model, performance slightly decreased, showing that low-correlation features may still contribute useful information.
- **Log-Transformed Target**: Applying a log transformation to the target variable significantly improved model performance, with XGB achieving the highest R² score of 0.88, making it the best-performing model across all evaluation methods.

This comprehensive evaluation highlights that the **Extreme Gradient Boosting (XGB) model** with a log-transformed target variable and all features yields the most accurate predictions, demonstrating its effectiveness for house price prediction.


## Deep Learning Models and Results

For the deep learning model, all features were used without additional feature selection or transformation. A grid search was conducted to optimize the neural network’s hyperparameters, including the number of neurons, batch size, learning rate, and activation functions.

- **Final Model Configuration**:
  - **Activation Functions**: ReLU in the hidden layers; Sigmoid in the output layer.
  - **Loss Function**: Mean Squared Error (MSE).
  - **Evaluation Metrics**: Mean Absolute Error (MAE) and R² score.
  - **Early Stopping**: Implemented to prevent overfitting by monitoring validation loss.

![Training vs Validation Loss](./Train_vs_Val_Loss.png)
  
The graph above shows the model’s training loss vs. validation loss, with a minimal difference indicating the model is not overfitting. The optimized neural network achieved a final performance with an **MSE** of **0.0023** and an **R² score** of **0.83** on unseen test data, indicating strong predictive capability for the house pricing dataset.

| Hyperparameter                  | Final Value   |
|---------------------------------|---------------|
| **Number of Neurons**           | **544** (Optimized via grid search) |
| **Batch Size**                  | **8** (Optimized via grid search) |
| **Learning Rate**               | **0.001** (Optimized via grid search) |
| **Activation Functions**        | **ReLU** (hidden), **Sigmoid** (output) |

This configuration yielded an effective model, demonstrating reliable performance and generalization on test data.

## Dependencies
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `tensorflow`
- `xgboost`

Install dependencies with:
```bash
pip install -r requirements.txt
```

## Results and Conclusion
- **Best Model**: The best results were obtained using the Extreme Gradient Boosting (XGB) model with a log-transformed target variable and all features. This model achieved an R² score of 0.88, demonstrating robust performance for house price prediction on this dataset.
- **Conclusion**: The findings indicate that the XGB model with log-transformed data and all features provides the most accurate predictions, making it a reliable choice for house price estimation. This project highlights the importance of preprocessing and feature selection in enhancing model accuracy for regression tasks.
