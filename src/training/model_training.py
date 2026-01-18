# internal library imports
import os
import re

# third-party imports
from dotenv import load_dotenv
from loguru import logger
import mlflow

# GPU configuration - set USE_GPU=true to enable CUDA acceleration
USE_GPU = os.getenv("USE_GPU", "false").lower() == "true"
DEVICE = "cuda" if USE_GPU else "cpu"
logger.info(f"XGBoost device configured: {DEVICE}")

# Hyperparameter configuration - configurable via environment variables
OPTUNA_N_TRIALS = int(os.getenv("OPTUNA_N_TRIALS", "200"))
XGBOOST_N_ESTIMATORS_MAX = int(os.getenv("XGBOOST_N_ESTIMATORS_MAX", "200"))
logger.info(f"Optuna trials: {OPTUNA_N_TRIALS}, XGBoost max estimators: {XGBOOST_N_ESTIMATORS_MAX}")
from mlflow.models.signature import infer_signature
import numpy as np
import optuna
from optuna.integration import XGBoostPruningCallback
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import pandas as pd
import polars as pl
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
import xgboost as xgb

# custom/internal imports
import src.utils as utils

# ------------------------------------------------------------------------
# supporting functions
# ------------------------------------------------------------------------

def unpack_engineered(
	engineered: pl.DataFrame,
	engineered_schema: pl.DataFrame
) -> pl.DataFrame:
	"""
	flattens out all array columns

	:param engineered: engineered DataFrame with categorical list columns
		stored as sparse array
	:param engineered_schema: column mapping of exactly how to unpack the
		sparse array columns
	:return: DataFrame containing the unpacked columns
	"""
	# Get the column names and convert list column to struct, then unnest
	engineered_unpacked = engineered.clone()

	for col in engineered_schema['original_column']:
		column_names = (engineered_schema.filter(
			pl.col('original_column') == col)['exploded_mapping']
				.item()
				.to_list()
		)

		if len(column_names) > 1:
			engineered_unpacked = (
				engineered_unpacked.with_columns(
					pl.col(col)
						.list
						.to_struct(fields=column_names)
						.alias(col + "_struct")
				).unnest(col + "_struct")
				.drop(col)
			)

	return engineered_unpacked


def prep_engineered(
	df: pl.DataFrame
) -> pd.DataFrame:
	"""
	performs all other processing functions on the dataframe before model
		ingestion

	:param df: engineered DataFrame before final pre-processing
	:return: DataFrame ready for model ingestion
	"""
	# Get the column names and convert list column to struct, then unnest
	df_prepped = df.clone()

	# drop unneeded columns
	df_prepped = df_prepped.drop([
		'media_title',
		'created_at',
		'updated_at'
	])

	pdf_prepped = df_prepped.to_pandas()
	pdf_prepped.set_index('imdb_id', inplace=True)

	# convert categoricals
	categorical_cols = ['production_status', 'original_language']
	for col in categorical_cols:
		pdf_prepped[col] = pdf_prepped[col].astype('category')

	# convert numerics
	num_cols = ['imdb_rating', 'tmdb_rating']
	for col in num_cols:
		pdf_prepped[col] = pdf_prepped[col].astype(float)

	return pdf_prepped


def data_split(
	pdf: pd.DataFrame,
	split_size: float = 0.2,
	random_state: int = 42
) -> tuple[
	pd.DataFrame,
	pd.DataFrame,
	pd.DataFrame,
	pd.Series,
	pd.Series,
	pd.Series
]:
	"""
	take full df and performs test/train/val split based off of size param

	:param pdf: whole dataset including the label column
	:param split_size: the size in decimal percentage of the training and
		the validation samples
	:param random_state: random state param for reproducibility
	:return: tuple containing[
		the full X DataFrame
		X_train DataFrame
		X_test DataFrame
		the full y Series
		y_train Series
		y_test Series
	]
	"""
	pdf_split = pdf.copy()

	# features/label split
	X = pdf_split.drop('label', axis=1)
	y = pdf_split['label']

	# train/test split
	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=split_size,
		random_state=random_state
	)

	logger.info(f"split complete on {len(X)} data elements with {len(X.columns)} features")
	logger.info(f"{len(X_train)} elements to be used for training")
	logger.info(f"{len(X_test)} elements to be used for testing")

	return (
		X,
		X_train,
		X_test,
		y,
		y_train,
		y_test
	)


def track_input_metrics(
	X: pd.DataFrame,
	X_train: pd.DataFrame,
	X_test: pd.DataFrame,
):
	"""
	logs all metrics recorded by MLFlow before the model is executed

	:param X: full feature set
	:param X_train: feature training set
	:param X_test:  feature test set
	:return: None
	"""
	# Log data info
	mlflow.log_param("train_size", X_train.shape[0])
	mlflow.log_param("test_size", X_test.shape[0])
	mlflow.log_param("features", list(X.columns))

	return


def optuna_objective(
	trial: optuna.trial.Trial,
	X_train: pd.DataFrame,
	y_train: pd.Series,
	random_seed: int = 42
) -> float:
	"""
	Optuna objective function for XGBoost hyperparameter optimization

	:param trial: Optuna trial object for parameter suggestions
	:param X_train: training features
	:param y_train: training labels
	:param random_seed: random state for reproducibility
	:return: mean CV score (f1) to maximize
	"""

	# Suggest hyperparameters - full ranges for exploration
	params = {
		'objective': 'binary:logistic',
		'enable_categorical': True,
		'device': DEVICE,
		'random_state': random_seed,
		'scale_pos_weight': trial.suggest_int('scale_pos_weight', 1, 15),
		'max_depth': trial.suggest_int('max_depth', 3, 7),
		'n_estimators': trial.suggest_int('n_estimators', min(50, XGBOOST_N_ESTIMATORS_MAX), XGBOOST_N_ESTIMATORS_MAX),
		'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
		'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2,
											 log=True),
		'gamma': trial.suggest_float('gamma', 0, 0.2),
		'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
		'subsample': trial.suggest_float('subsample', 0.8, 1.0),
		'max_delta_step': trial.suggest_int('max_delta_step', 0, 5),
		'colsample_bytree': trial.suggest_float('colsample_bytree', 0.8, 1.0),
		'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.8, 1.0)
	}

	# Setup cross-validation with pruning
	cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
	scores = []

	for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
		X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[
			val_idx]
		y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[
			val_idx]

		# Create model with pruning callback
		model = xgb.XGBClassifier(**params)

		# Add pruning callback
		pruning_callback = XGBoostPruningCallback(trial, f'validation_{fold}')

		model.fit(
			X_fold_train, y_fold_train,
			eval_set=[(X_fold_val, y_fold_val)],
			verbose=False
		)

		# Get predictions and calculate F1 score
		y_pred = model.predict(X_fold_val)
		fold_score = f1_score(y_fold_val, y_pred)
		scores.append(fold_score)

		# Report intermediate score for pruning decision
		trial.report(fold_score, fold)

		# Check if trial should be pruned (after 2 poor folds as requested)
		if fold >= 1 and trial.should_prune():
			raise optuna.TrialPruned()

	return np.mean(scores)


def track_output_metrics(
	study: optuna.study.Study,
	model: xgb.XGBClassifier,
	X_test: pd.DataFrame,
	y_test: pd.Series
):
	"""
	generates all relevant model output metrics and logs in MLflow

	:param study: Optuna study object with optimization results
	:param model: trained XGBoost model
	:param X_test: test features
	:param y_test: label output subset
	:return: None
	"""

	# Log Optuna study results
	study_results = {
		'best_trial_number': study.best_trial.number,
		'best_value': study.best_value,
		'best_params': study.best_params,
		'n_trials': len(study.trials),
		'n_complete_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
		'n_pruned_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
	}

	mlflow.log_dict(study_results, "optuna_study_summary.json")

	# Log detailed trial results
	trials_df = study.trials_dataframe()
	if not trials_df.empty:
		mlflow.log_dict(trials_df.to_dict(orient='records'), "optuna_trials_detailed.json")

	# Log parameter importance if available
	try:
		param_importance = optuna.importance.get_param_importances(study)
		mlflow.log_dict(param_importance, "optuna_param_importance.json")
	except Exception as e:
		logger.warning(f"Could not calculate parameter importance: {e}")

	# Log best parameters as MLflow params
	for param, value in study.best_params.items():
		mlflow.log_param(param, value)

	logger.info(f"Best trial: {study.best_trial.number}")
	logger.info(f"Best F1 score: {study.best_value:.4f}")
	logger.info(f"Best params: {study.best_params}")

	# Generate predictions using the trained model
	y_pred = model.predict(X_test)
	y_pred_proba = model.predict_proba(X_test)[:, 1]

	# Calculate metrics
	if len(np.unique(y_test)) > 1:
		tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
		mlflow.log_metric("true_positives", tp)
		mlflow.log_metric("false_positives", fp)
		mlflow.log_metric("true_negatives", tn)
		mlflow.log_metric("false_negatives", fn)
		logger.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

		if (tp + fp > 0) and (tp + fn > 0):
			f1 = f1_score(y_test, y_pred)
			mlflow.log_metric("f1", f1)
			logger.info(f"F1 Score: {f1:.4f}")

		accuracy = accuracy_score(y_test, y_pred)
		mlflow.log_metric("accuracy", accuracy)
		logger.info(f"Accuracy: {accuracy:.4f}")

		if tp + fp > 0:
			precision = precision_score(y_test, y_pred)
			mlflow.log_metric("precision", precision)
			logger.info(f"Precision: {precision:.4f}")

		if tp + fn > 0:
			recall = recall_score(y_test, y_pred)
			mlflow.log_metric("recall", recall)
			logger.info(f"Recall: {recall:.4f}")

		if len(np.unique(y_test)) > 1:
			auc = roc_auc_score(y_test, y_pred_proba)
			mlflow.log_metric("auc", auc)
			logger.info(f"AUC-ROC: {auc:.4f}")

	return


def log_tree_visualizations(
	model: xgb.XGBClassifier,
	num_trees: int = 5
):
	"""
	Logs XGBoost tree visualizations as MLflow artifacts.

	Stores DOT format representations of decision trees and feature importance
	for later visualization in Streamlit via st.graphviz_chart().

	:param model: trained XGBoost model
	:param num_trees: number of trees to export (default 5)
	:return: None
	"""
	booster = model.get_booster()

	# Get total number of trees in the model
	total_trees = booster.num_boosted_rounds()
	trees_to_export = min(num_trees, total_trees)

	logger.info(f"Exporting {trees_to_export} of {total_trees} trees as DOT format")

	# Get DOT format for trees
	dot_trees = booster.get_dump(dump_format='dot')

	# Build feature importance dict
	feature_names = booster.feature_names
	importance_scores = model.feature_importances_.tolist()
	feature_importance = dict(zip(feature_names, importance_scores))

	# Sort by importance descending
	feature_importance_sorted = dict(
		sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
	)

	# Create artifact structure
	tree_artifacts = {
		'metadata': {
			'total_trees': total_trees,
			'exported_trees': trees_to_export,
			'feature_count': len(feature_names)
		},
		'trees': dot_trees[:trees_to_export],
		'feature_importance': feature_importance_sorted
	}

	mlflow.log_dict(tree_artifacts, "xgboost_trees.json")

	logger.info(f"Tree visualizations logged to MLflow: {trees_to_export} trees, {len(feature_names)} features")

	return


# ------------------------------------------------------------------------
# primary function
# ------------------------------------------------------------------------

def xgb_hyp_op(
	random_seed: int = 42
):
	"""
	end-to-end stateless hyperparameter optimization service

	:param random_seed: random_state var to be passed to all functions
	:return: None
	"""
	# load dotenv at the module level if running locally
	if os.getenv("LOCAL_DEVELOPMENT", '') == "true":
		load_dotenv(override=True)
		random_seed = 42

	# set random seeds
	np.random.seed(random_seed)

	# set minio env vars
	os.environ['MLFLOW_S3_ENDPOINT_URL'] = str(
		os.environ['REEL_DRIVER_MINIO_ENDPOINT'] + ":" + os.environ['REEL_DRIVER_MINIO_PORT']
	)
	os.environ['AWS_ACCESS_KEY_ID'] = os.environ['REEL_DRIVER_MINIO_ACCESS_KEY']
	os.environ['AWS_SECRET_ACCESS_KEY'] = os.environ['REEL_DRIVER_MINIO_SECRET_KEY']

	# connect to k8s mlflow service
	mlflow_uri = "http://" + os.getenv('REEL_DRIVER_MLFLOW_HOST') + ":" + os.getenv('REEL_DRIVER_MLFLOW_PORT')
	mlflow.set_tracking_uri(mlflow_uri)

	# Set experiment
	mlflow.set_experiment(experiment_name=os.getenv('REEL_DRIVER_MLFLOW_EXPERIMENT'))

	logger.info("extracting engineered features")

	# read in training data and metadata
	engineered = utils.select_star(table="engineered")
	engineered_schema = utils.select_star(table="engineered_schema")
	normalization_table = utils.select_star(table="engineered_normalization_table")

	# perform prep on engineered data
	df = unpack_engineered(
		engineered=engineered,
		engineered_schema=engineered_schema
	)
	pdf = prep_engineered(df=df)

	# separate anomalous records from training data
	# anomalous records are excluded from training but included in final predictions
	if 'anomalous' in pdf.columns:
		anomalous_mask = pdf['anomalous'].fillna(False).astype(bool)
		pdf_train = pdf[~anomalous_mask].drop(columns=['anomalous'])
		pdf_all = pdf.drop(columns=['anomalous'])
		logger.info(f"filtered {anomalous_mask.sum()} anomalous records from training")
		logger.info(f"training on {len(pdf_train)} records, predicting on {len(pdf_all)} records")
	else:
		pdf_train = pdf
		pdf_all = pdf
		logger.info("no anomalous column found, using all records for training")

	# split (using non-anomalous data only)
	X, X_train, X_test, y, y_train, y_test = data_split(
		pdf=pdf_train,
		split_size=0.2,
		random_state=random_seed
	)

	# Start MLflow run with variant suffix if specified
	training_variant = os.environ.get('TRAINING_VARIANT', '')
	run_name = f"xgboost_optuna_search_{training_variant}" if training_variant else "xgboost_optuna_search"
	with mlflow.start_run(run_name=run_name):

		# enter notes
		notes = "automated training run"
		mlflow.set_tag("notes", notes)

		# store all input model values in MLFlow
		track_input_metrics(X, X_train, X_test)

		# Setup Optuna study with medium pruning
		sampler = TPESampler(seed=random_seed, n_startup_trials=20)
		pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=1,
							  interval_steps=1)

		study = optuna.create_study(
			direction='maximize',
			sampler=sampler,
			pruner=pruner,
			study_name=f"xgboost_optimization_{random_seed}"
		)

		logger.info(
			f"Starting Optuna hyperparameter optimization with {OPTUNA_N_TRIALS} trials")

		# Run optimization
		study.optimize(
			lambda trial: optuna_objective(trial, X_train, y_train,
										   random_seed),
			n_trials=OPTUNA_N_TRIALS,
			show_progress_bar=True
		)

		logger.info(
			f"Optimization completed. Best trial: {study.best_trial.number}")
		logger.info(f"Best F1 score: {study.best_value:.4f}")
		logger.info(f"Best parameters: {study.best_params}")

		logger.info('building full_model')

		# generate model with best params and train with all data
		best_params = study.best_params.copy()
		best_params.update({
			'objective': 'binary:logistic',
			'enable_categorical': True,
			'device': DEVICE,
			'random_state': random_seed
		})

		full_model = xgb.XGBClassifier(**best_params)
		full_model.fit(X, y)

		# track all output metrics
		track_output_metrics(
			study=study,
			model=full_model,
			X_test=X_test,
			y_test=y_test
		)

		# log tree visualizations for Streamlit display
		log_tree_visualizations(model=full_model, num_trees=5)

		# prepare full dataset (including anomalous) for predictions
		X_all = pdf_all.drop('label', axis=1)
		y_all = pdf_all['label']

		# run predictions on full data set (including anomalous records)
		y_pred = full_model.predict(X_all)
		y_pred_proba = full_model.predict_proba(X_all)

		# create model signature
		if len(X_all) < 1000:
			signature = infer_signature(
				X_all,
				y_pred
			)
		else:
			signature = infer_signature(
				X_all.head(1000),
				y_pred[:1000]
			)

		# store model
		mlflow.sklearn.log_model(
			sk_model=full_model,
			artifact_path="model",
			registered_model_name=os.getenv("REEL_DRIVER_MLFLOW_MODEL"),
			signature=signature,
			conda_env=None
		)

		logger.info('full_model stored in MLflow model registry')

		# Save normalization table and engineered schema as JSON artifacts (in-memory)
		mlflow.log_dict(
			normalization_table.to_pandas().to_dict(orient='records'),
			"engineered_normalization_table.json"
		)

		mlflow.log_dict(
			engineered_schema.to_pandas().to_dict(orient='records'),
			"engineered_schema.json"
		)

		logger.info('normalization table and engineered schema saved as MLflow artifacts')

		# create atp.prediction dataframe
		prediction = pl.DataFrame({
			'imdb_id': X_all.index,
			'prediction': y_pred,
			'probability': y_pred_proba[:, 1],
			'actual': y_all
		}).with_columns(
			cm_value = pl.when(pl.col('prediction') == 1)
				.then(
					pl.when(pl.col('actual') == 1)
						.then(pl.lit("tp"))
						.otherwise(pl.lit("fp"))
				).otherwise(
					pl.when(pl.col('actual') == 0)
						.then(pl.lit("tn"))
					.otherwise(pl.lit("fn"))
				)
		).drop('actual')

		# add full prediction table to db
		utils.trunc_and_load(
			df=prediction,
			table_name="prediction"
		)

# main guard
def __main__():
	xgb_hyp_op()

# -----------------------------------------------------------------------------
# end of _03_xgb_binomial_classifier_build_model.py
# -----------------------------------------------------------------------------

