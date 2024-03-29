-------------------------------------------------------------------------------
		HOW TO RUN THE CODE FROM THIS FOLDER
-------------------------------------------------------------------------------

-----------------------------------------------------------
1. TYPICAL WORKFLOW
-----------------------------------------------------------

>>> pip install -r ../requirements.txt

-----------------------
Step1 : pre-processing
-----------------------
Script : pre_processing.py		
First, run this script to preprocess the train set. 
The preprocessed dataframe will be available in csv format as train_df_processed.csv.

-----------------------
Step2 : pipeline
-----------------------
Script: validation_pipeline_heldout.py or validation_pipeline_kfold.py
Depending on the validation strategy, run either of these two scripts compare the model accuracies on the preprocessed train set.

-----------------------
Step3 : generate test 
	predictions
-----------------------
Script: generate_predictions_kaggle.py
Run this script to generate the a csv that can be uploaded to Kaggle

-----------------------------------------------------------
2. AVAILABLE MODELS
-----------------------------------------------------------
naive_bayes/
	naive_bayes.py 		Multiclass Bernouilli Naive Bayes model, from scratch
	test_naive_bayes.py	Compares the performance of the custom naive bayes model with Sklearn's version

multinomialNB.py		Experiment with Sklear's Multinomial Naive Bayes 
svm.py				Experiment with Sklear's Linear SVC

-----------------------------------------------------------
3. AVAILABLE VALIDATION PIPELINES
-----------------------------------------------------------

validation_pipeline_heldout.py	Validation pipeline, using a held out test set
validation_pipeline_kfold.py	Validation pipeline, using k-fold cross validation

