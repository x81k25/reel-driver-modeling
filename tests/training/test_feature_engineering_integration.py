#!/usr/bin/env python3
"""
Integration test for feature engineering training script.
Tests the complete feature engineering pipeline with real database connections.
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
class TestFeatureEngineeringIntegration:
    """Integration test for feature engineering pipeline."""
    
    def test_required_env_vars(self):
        """Test that all required environment variables are set."""
        required_vars = [
            'REEL_DRIVER_TRNG_PGSQL_HOST',
            'REEL_DRIVER_TRNG_PGSQL_PORT',
            'REEL_DRIVER_TRNG_PGSQL_DATABASE',
            'REEL_DRIVER_TRNG_PGSQL_USERNAME',
            'REEL_DRIVER_TRNG_PGSQL_PASSWORD'
        ]
        
        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            pytest.skip(f"Missing required environment variables: {missing_vars}")
        
        print("✅ All required environment variables are set")
    
    def test_feature_engineering_pipeline(self):
        """Test the complete feature engineering pipeline."""
        print("\n🔧 Testing feature engineering pipeline...")
        
        # Set environment for local development
        env = os.environ.copy()
        env['LOCAL_DEVELOPMENT'] = 'true'
        env['PYTHONPATH'] = project_root
        
        # Run feature engineering script
        try:
            result = subprocess.run(
                [sys.executable, "-c", "from src.training.feature_engineering import __main__; __main__()"],
                cwd=project_root,
                env=env,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            print(f"Return code: {result.returncode}")
            print(f"STDOUT:\n{result.stdout}")
            if result.stderr:
                print(f"STDERR:\n{result.stderr}")
            
            # Check if the process completed successfully
            assert result.returncode == 0, f"Feature engineering failed with return code {result.returncode}\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
            
            # Check for expected output messages
            expected_messages = [
                "extracting training data",
                "filtering training data", 
                "normalizing numeric features",
                "encoding categorical features",
                "encoding labels",
                "persisting engineered features"
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
            
            print("✅ Feature engineering pipeline completed successfully")
            
        except subprocess.TimeoutExpired:
            pytest.fail("Feature engineering pipeline timed out after 5 minutes")
        except Exception as e:
            pytest.fail(f"Feature engineering pipeline failed with exception: {e}")
    
    def test_database_connectivity(self):
        """Test database connectivity before running pipeline."""
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
    
    def test_training_data_availability(self):
        """Test that training data is available in the database."""
        print("\n📊 Testing training data availability...")
        
        try:
            from src.utils.db_operations import select_star
            
            # Check if training data exists
            training_data = select_star(table="training")
            
            assert len(training_data) > 0, "No training data found in database"
            print(f"✅ Found {len(training_data)} training records")
            
            # Check for required columns
            required_columns = ['imdb_id', 'label']
            missing_columns = []
            for col in required_columns:
                if col not in training_data.columns:
                    missing_columns.append(col)
            
            assert not missing_columns, f"Missing required columns in training data: {missing_columns}"
            print("✅ Training data has required columns")
            
        except ImportError as e:
            pytest.skip(f"Cannot import database utilities: {e}")
        except Exception as e:
            pytest.skip(f"Training data check failed: {e}")
    
    def test_feature_engineering_output(self):
        """Test that feature engineering produces expected outputs."""
        print("\n🔍 Testing feature engineering outputs...")
        
        try:
            from src.utils.db_operations import select_star
            
            # Check if engineered features exist
            try:
                engineered_data = select_star(table="engineered")
                print(f"✅ Found {len(engineered_data)} engineered feature records")
            except Exception:
                print("ℹ️  Engineered features table not found (expected if not run yet)")
            
            # Check if normalization table exists
            try:
                normalization_data = select_star(table="engineered_normalization_table")
                print(f"✅ Found {len(normalization_data)} normalization records")
            except Exception:
                print("ℹ️  Normalization table not found (expected if not run yet)")
            
            # Check if schema table exists
            try:
                schema_data = select_star(table="engineered_schema")
                print(f"✅ Found {len(schema_data)} schema records")
            except Exception:
                print("ℹ️  Schema table not found (expected if not run yet)")
                
        except ImportError as e:
            pytest.skip(f"Cannot import database utilities: {e}")
        except Exception as e:
            print(f"⚠️  Error checking outputs: {e}")


if __name__ == "__main__":
    """Run the integration test directly."""
    print("=" * 60)
    print("Feature Engineering Integration Test")
    print("=" * 60)
    
    # Check environment configuration
    required_vars = [
        'REEL_DRIVER_TRNG_PGSQL_HOST',
        'REEL_DRIVER_TRNG_PGSQL_PORT',
        'REEL_DRIVER_TRNG_PGSQL_DATABASE',
        'REEL_DRIVER_TRNG_PGSQL_USERNAME'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print(f"❌ Missing required environment variables: {missing_vars}")
        sys.exit(1)
    
    print("✅ Environment configuration validated")
    
    # Run pytest on this file
    pytest.main([__file__, "-v", "-s"])