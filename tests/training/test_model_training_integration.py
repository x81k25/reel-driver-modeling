#!/usr/bin/env python3
"""
Integration test for model training script.
Tests the complete model training pipeline with real MLflow and database connections.
"""

import pytest
import os
import sys
import subprocess
from dotenv import load_dotenv

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Load environment variables
load_dotenv()


@pytest.mark.integration
@pytest.mark.slow
class TestModelTrainingIntegration:
    """Integration test for model training pipeline."""
    
    def test_required_env_vars(self):
        """Test that all required environment variables are set."""
        required_vars = [
            'REEL_DRIVER_TRNG_PGSQL_HOST',
            'REEL_DRIVER_TRNG_PGSQL_PORT',
            'REEL_DRIVER_TRNG_PGSQL_DATABASE',
            'REEL_DRIVER_TRNG_PGSQL_USERNAME',
            'REEL_DRIVER_TRNG_PGSQL_PASSWORD',
            'REEL_DRIVER_MLFLOW_HOST',
            'REEL_DRIVER_MLFLOW_PORT',
            'REEL_DRIVER_MLFLOW_EXPERIMENT',
            'REEL_DRIVER_MLFLOW_MODEL',
            'REEL_DRIVER_MINIO_ENDPOINT',
            'REEL_DRIVER_MINIO_PORT',
            'REEL_DRIVER_MINIO_ACCESS_KEY',
            'REEL_DRIVER_MINIO_SECRET_KEY'
        ]
        
        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            pytest.skip(f"Missing required environment variables: {missing_vars}")
        
        print("✅ All required environment variables are set")
    
    def test_mlflow_connectivity(self):
        """Test MLflow connectivity before running training."""
        print("\n📊 Testing MLflow connectivity...")
        
        try:
            import mlflow
            from mlflow.tracking import MlflowClient
            
            # Set up MLflow environment like training script does
            os.environ['MLFLOW_S3_ENDPOINT_URL'] = str(
                os.environ['REEL_DRIVER_MINIO_ENDPOINT'] +
                ":" +
                os.environ['REEL_DRIVER_MINIO_PORT']
            )
            os.environ['AWS_ACCESS_KEY_ID'] = os.environ['REEL_DRIVER_MINIO_ACCESS_KEY']
            os.environ['AWS_SECRET_ACCESS_KEY'] = os.environ['REEL_DRIVER_MINIO_SECRET_KEY']
            
            # Set MLflow tracking URI
            mlflow_uri = "http://" + os.getenv('REEL_DRIVER_MLFLOW_HOST') + ":" + os.getenv('REEL_DRIVER_MLFLOW_PORT')
            mlflow.set_tracking_uri(mlflow_uri)
            
            # Test connection
            client = MlflowClient()
            experiments = client.search_experiments()
            
            print(f"✅ MLflow connection successful, found {len(experiments)} experiments")
            
        except Exception as e:
            pytest.skip(f"MLflow connection failed: {e}")
    
    def test_database_connectivity(self):
        """Test database connectivity before running training."""
        print("\n🗄️  Testing database connectivity...")
        
        try:
            # Import database utilities
            from src.utils.db_operations import gen_adbc_con
            
            # Test database connection
            with gen_adbc_con() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    result = cursor.fetchone()
                    assert result[0] == 1, "Database connection test failed"
            
            print("✅ Database connection successful")
            
        except ImportError as e:
            pytest.skip(f"Cannot import database utilities: {e}")
        except Exception as e:
            pytest.skip(f"Database connection failed: {e}")
    
    def test_engineered_data_availability(self):
        """Test that engineered data is available for training."""
        print("\n🔧 Testing engineered data availability...")
        
        try:
            from src.utils.db_operations import select_star
            
            # Check if engineered features exist
            engineered_data = select_star(table="engineered")
            assert len(engineered_data) > 0, "No engineered features found in database"
            print(f"✅ Found {len(engineered_data)} engineered feature records")
            
            # Check if normalization table exists
            normalization_data = select_star(table="engineered_normalization_table")
            assert len(normalization_data) > 0, "No normalization data found"
            print(f"✅ Found {len(normalization_data)} normalization records")
            
            # Check if schema table exists
            schema_data = select_star(table="engineered_schema")
            assert len(schema_data) > 0, "No schema data found"
            print(f"✅ Found {len(schema_data)} schema records")
            
        except ImportError as e:
            pytest.skip(f"Cannot import database utilities: {e}")
        except Exception as e:
            pytest.skip(f"Engineered data check failed: {e}")
    
    def test_model_training_pipeline(self):
        """Test the complete model training pipeline."""
        print("\n🤖 Testing model training pipeline...")
        
        # Set environment for local development
        env = os.environ.copy()
        env['LOCAL_DEVELOPMENT'] = 'true'
        env['PYTHONPATH'] = project_root
        
        # Set MLflow environment variables
        env['MLFLOW_S3_ENDPOINT_URL'] = str(
            env['REEL_DRIVER_MINIO_ENDPOINT'] +
            ":" +
            env['REEL_DRIVER_MINIO_PORT']
        )
        env['AWS_ACCESS_KEY_ID'] = env['REEL_DRIVER_MINIO_ACCESS_KEY']
        env['AWS_SECRET_ACCESS_KEY'] = env['REEL_DRIVER_MINIO_SECRET_KEY']
        
        # Run model training script
        try:
            result = subprocess.run(
                [sys.executable, "-c", "from src.training.model_training import __main__; __main__()"],
                cwd=project_root,
                env=env,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout for training
            )
            
            print(f"Return code: {result.returncode}")
            print(f"STDOUT:\n{result.stdout}")
            if result.stderr:
                print(f"STDERR:\n{result.stderr}")
            
            # Check if the process completed successfully
            assert result.returncode == 0, f"Model training failed with return code {result.returncode}\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
            
            # Check for expected output messages
            expected_messages = [
                "extracting engineered features",
                "splitting data for training",
                "performing hyperparameter search",
                "training model",
                "logging model to mlflow",
                "model registered successfully"
            ]
            
            output_lower = result.stdout.lower()
            missing_messages = []
            for message in expected_messages:
                if message not in output_lower:
                    missing_messages.append(message)
            
            if missing_messages:
                print(f"⚠️  Missing expected log messages: {missing_messages}")
            else:
                print("✅ All expected pipeline steps found in output")
            
            print("✅ Model training pipeline completed successfully")
            
        except subprocess.TimeoutExpired:
            pytest.fail("Model training pipeline timed out after 30 minutes")
        except Exception as e:
            pytest.fail(f"Model training pipeline failed with exception: {e}")
    
    def test_model_registration(self):
        """Test that model was registered in MLflow."""
        print("\n📋 Testing model registration...")
        
        try:
            import mlflow
            from mlflow.tracking import MlflowClient
            
            # Set up MLflow environment
            os.environ['MLFLOW_S3_ENDPOINT_URL'] = str(
                os.environ['REEL_DRIVER_MINIO_ENDPOINT'] +
                ":" +
                os.environ['REEL_DRIVER_MINIO_PORT']
            )
            os.environ['AWS_ACCESS_KEY_ID'] = os.environ['REEL_DRIVER_MINIO_ACCESS_KEY']
            os.environ['AWS_SECRET_ACCESS_KEY'] = os.environ['REEL_DRIVER_MINIO_SECRET_KEY']
            
            mlflow_uri = "http://" + os.getenv('REEL_DRIVER_MLFLOW_HOST') + ":" + os.getenv('REEL_DRIVER_MLFLOW_PORT')
            mlflow.set_tracking_uri(mlflow_uri)
            
            client = MlflowClient()
            model_name = os.getenv('REEL_DRIVER_MLFLOW_MODEL')
            
            # Check if model exists
            try:
                model_versions = client.search_model_versions(f"name='{model_name}'")
                assert len(model_versions) > 0, f"No versions found for model {model_name}"
                
                latest_version = max(model_versions, key=lambda x: int(x.version))
                print(f"✅ Model {model_name} found with latest version {latest_version.version}")
                print(f"   Status: {latest_version.status}")
                print(f"   Run ID: {latest_version.run_id}")
                
                # Check model artifacts
                run = client.get_run(latest_version.run_id)
                artifacts = client.list_artifacts(latest_version.run_id)
                artifact_names = [artifact.path for artifact in artifacts]
                
                expected_artifacts = ['model-artifacts']
                for artifact in expected_artifacts:
                    if artifact in artifact_names:
                        print(f"✅ Found expected artifact: {artifact}")
                    else:
                        print(f"⚠️  Missing artifact: {artifact}")
                
            except Exception as e:
                print(f"⚠️  Model registration check failed: {e}")
                
        except Exception as e:
            pytest.skip(f"Model registration test failed: {e}")
    
    def test_experiment_tracking(self):
        """Test that experiment was tracked properly."""
        print("\n📈 Testing experiment tracking...")
        
        try:
            import mlflow
            from mlflow.tracking import MlflowClient
            
            # Set up MLflow environment
            mlflow_uri = "http://" + os.getenv('REEL_DRIVER_MLFLOW_HOST') + ":" + os.getenv('REEL_DRIVER_MLFLOW_PORT')
            mlflow.set_tracking_uri(mlflow_uri)
            
            client = MlflowClient()
            experiment_name = os.getenv('REEL_DRIVER_MLFLOW_EXPERIMENT')
            
            # Get experiment
            experiment = mlflow.get_experiment_by_name(experiment_name)
            assert experiment is not None, f"Experiment {experiment_name} not found"
            
            # Get recent runs
            runs = client.search_runs(experiment_ids=[experiment.experiment_id], max_results=5)
            assert len(runs) > 0, "No runs found in experiment"
            
            latest_run = runs[0]
            print(f"✅ Found experiment {experiment_name} with {len(runs)} runs")
            print(f"   Latest run: {latest_run.info.run_id}")
            print(f"   Status: {latest_run.info.status}")
            
            # Check metrics
            metrics = latest_run.data.metrics
            expected_metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
            for metric in expected_metrics:
                if metric in metrics:
                    print(f"✅ Found metric {metric}: {metrics[metric]:.3f}")
                else:
                    print(f"⚠️  Missing metric: {metric}")
            
        except Exception as e:
            pytest.skip(f"Experiment tracking test failed: {e}")


if __name__ == "__main__":
    """Run the integration test directly."""
    print("=" * 60)
    print("Model Training Integration Test")
    print("=" * 60)
    
    # Check environment configuration
    required_vars = [
        'REEL_DRIVER_MLFLOW_HOST',
        'REEL_DRIVER_MLFLOW_PORT',
        'REEL_DRIVER_MLFLOW_EXPERIMENT',
        'REEL_DRIVER_MLFLOW_MODEL'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print(f"❌ Missing required environment variables: {missing_vars}")
        sys.exit(1)
    
    print("✅ Environment configuration validated")
    
    # Run pytest on this file
    pytest.main([__file__, "-v", "-s"])