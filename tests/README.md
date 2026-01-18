# Reel Driver API Tests

Comprehensive test suite for the Reel Driver FastAPI application.

## Test Structure

- `conftest.py` - Test fixtures and configuration
- `test_endpoints.py` - Comprehensive endpoint tests
- `pytest.ini` - Pytest configuration

## Running Tests

### Install Test Dependencies

```bash
pip install -r test-requirements.txt
```

### Run All Tests

```bash
pytest tests/
```

### Run with Verbose Output

```bash
pytest tests/ -v
```

### Run Specific Test Classes

```bash
pytest tests/test_endpoints.py::TestRootEndpoint -v
pytest tests/test_endpoints.py::TestHealthEndpoint -v
pytest tests/test_endpoints.py::TestPredictionEndpoint -v
pytest tests/test_endpoints.py::TestBatchPredictionEndpoint -v
```

## Test Coverage

### Endpoints Tested

1. **Root Endpoint** (`GET /reel-driver/`)
   - Basic functionality and response structure

2. **Health Check** (`GET /reel-driver/health`)
   - Model loaded state
   - Model not loaded state
   - Component health verification

3. **Single Prediction** (`POST /reel-driver/api/predict`)
   - Successful predictions
   - Input validation (IMDB ID format, ratings, years, etc.)
   - Error handling (model not loaded, prediction errors)
   - Minimal vs. full input scenarios

4. **Batch Prediction** (`POST /reel-driver/api/predict_batch`)
   - Multiple item predictions
   - Empty batch handling
   - Invalid items in batch
   - Error scenarios

5. **Documentation Endpoints**
   - OpenAPI JSON schema
   - Swagger UI docs
   - ReDoc documentation

6. **Error Handling**
   - 404 for non-existent endpoints
   - 405 for invalid HTTP methods
   - 422 for malformed JSON/validation errors

### Input Validation Tests

- IMDB ID format validation
- Country code length validation (2 characters)
- Language code validation (2 characters)
- Genre name length limits
- Numeric range validation (years, ratings, budgets)
- Negative value rejection

## Test Fixtures

### Mock Predictor (`mock_predictor`)
Provides a fully mocked XGBMediaPredictor with:
- Sample model metadata
- Feature names and schemas
- Normalization parameters
- Predefined prediction responses

### Test Clients
- `client_with_mock_predictor` - Client with working model
- `client_no_predictor` - Client with no model loaded

### Sample Data
- `sample_media_input` - Complete media metadata example
- `minimal_media_input` - Minimal valid input
- `batch_media_input` - Multiple items for batch testing

## Test Categories

- **Unit Tests**: Individual endpoint functionality
- **Integration Tests**: Full request/response cycles
- **Validation Tests**: Input validation edge cases
- **Error Handling**: Exception scenarios and error responses