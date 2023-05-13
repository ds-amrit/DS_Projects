# DS_Projects
In this repository I have included various machine learning and deep learning projects for reference.

# Project Experience:

1) Credit Card Fraud Detection:

•	Classified credit card transactions into fraud or non-fraud by building a model based on ensemble methods like Random Forest Classifier and Voting Classifier and 
  further fine-tuned them to optimize their performance.
•	Calculated the ratio of fraud and non-fraud transactions in the dataset. The fraudulent transactions were less than 1% of the dataset, leading to extremely skewed 
  class distribution.
•	Treated the imbalanced dataset using a combination of Synthetic Minority Oversampling Technique (SMOTE) and Random Under Sampler.
•	Created a pipeline to preprocess the data using the above techniques along with a model to fit the training dataset.  

3) Flight Price Prediction:

•	Applied feature engineering to create new features based on current features and ensured significant categorical features were labeled encoded and a random forest
  regressor was applied to predict flight price.
•	 Hyperparameter tuning was applied using the Randomized Search CV to get the best parameters for the above model.
•	Label encoded categorical features using pandas get dummies function and extracted important features using Extra Tree Regressor.
•	Split the data into train, and test and applied the mean squared error and r2 score technique to evaluate the machine learning model. 

4) Pneumonia Lung Detection using OpenCV and Machine Learning:

 Classified chest X-ray into "normal(0)" or "pneumonia(1)" by building a model based on Medical Image Segmentation. Visualizing lung x-rays for different categories
 Integrated 2 functions, which included img_to_vector and extract_color_histogram, to convert images.
 img_to_vector function resizes the image to a fixed height and width and then flattens the RGB pixels into a single list of numbers.
 extract_color_histogram function accepts images and constructs a color histogram to characterize the image’s color distribution.
 Applied different machine learning algorithms (KNN and Logistic Regression) on the image array, achieving a 93% accuracy level.




