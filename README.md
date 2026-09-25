SCT_ML_3 – Cat vs Dog Classifier (SVM)
A Support Vector Machine model that classifies images as Cat or Dog, built as Task 3 of my Machine Learning Internship at SkillCraft Technology.

🎯 Task
Implement an SVM to classify images of cats and dogs using the Kaggle Dogs vs Cats dataset.

⚙️ How it works
Images are converted to grayscale, resized, and turned into HOG (shape/edge) features. These features are fed into an SVM classifier trained to tell cats and dogs apart.

📊 Results
Test Accuracy: 72.2%
🚀 Try it yourself
python predict_image.py your_photo.jpg

Give it any photo — it'll tell you Cat or Dog with a confidence score.

🛠️ Built with
Python · scikit-learn · scikit-image · NumPy · Matplotlib

📁 Files
File	Description
task03_svm.py	--Trains the SVM model
predict_image.py --Test the model on any image
confusion_matrix.png --Model performance
sample_predictions.png --Example predictions
