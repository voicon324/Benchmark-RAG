"""
Unit tests for experiment configuration classes.

Tests configuration validation, serialization, and error handling.
"""

import pytest
import tempfile
import json
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from newaibench.experiment.config import (
    ExperimentConfig,
    ModelConfiguration,
    DatasetConfiguration,
    EvaluationConfiguration,
    OutputConfiguration,
    ConfigFormat
)


class TestModelConfiguration:
    """Test ModelConfiguration class."""
    
    def test_model_config_basic(self):
        """Test basic model configuration creation."""
        config = ModelConfiguration(
            name="test_model",
            type="sparse",
            model_name_or_path="",
            device="cpu",
            batch_size=32,
            max_seq_length=512,
            parameters={"k1": 1.2, "b": 0.75}
        )
        
        assert config.name == "test_model"
        assert config.type == "sparse"
        assert config.device == "cpu"
        assert config.batch_size == 32
        assert config.parameters["k1"] == 1.2
    
    def test_model_config_validation(self):
        """Test model configuration validation."""
        # Valid configuration
        config = ModelConfiguration(
            name="valid_model",
            type="dense",
            model_name_or_path="sentence-transformers/all-MiniLM-L6-v2"
        )
        config.validate()  # Should not raise
        
        # Invalid type
        config.type = "invalid_type"
        with pytest.raises(ValueError, match="Invalid model type"):
            config.validate()
    
    def test_model_config_device_validation(self):
        """Test device validation."""
        config = ModelConfiguration(
            name="test_model",
            type="sparse", 
            model_name_or_path=""
        )
        
        # Valid devices
        for device in ["cpu", "cuda", "auto"]:
            config.device = device
            config.validate()  # Should not raise
        
        # Invalid device
        config.device = "invalid_device"
        with pytest.raises(ValueError, match="Invalid device"):
            config.validate()
    
    def test_model_config_serialization(self):
        """Test model configuration serialization."""
        config = ModelConfiguration(
            name="test_model",
            type="dense",
            model_name_or_path="test/model",
            parameters={"param1": "value1"}
        )
        
        # Test to_dict
        config_dict = config.to_dict()
        assert config_dict["name"] == "test_model"
        assert config_dict["type"] == "dense"
        assert config_dict["parameters"]["param1"] == "value1"
        
        # Test from_dict
        restored_config = ModelConfiguration.from_dict(config_dict)
        assert restored_config.name == config.name
        assert restored_config.type == config.type
        assert restored_config.parameters == config.parameters


class TestDatasetConfiguration:
    """Test DatasetConfiguration class."""
    
    def test_dataset_config_basic(self):
        """Test basic dataset configuration creation."""
        config = DatasetConfiguration(
            name="test_dataset",
            type="text",
            data_dir="/path/to/data",
            config_overrides={"cache_enabled": True}
        )
        
        assert config.name == "test_dataset"
        assert config.type == "text"
        assert config.data_dir == "/path/to/data"
        assert config.config_overrides["cache_enabled"] is True
    
    def test_dataset_config_validation(self):
        """Test dataset configuration validation."""
        # Valid configuration
        config = DatasetConfiguration(
            name="valid_dataset",
            type="text",
            data_dir="/valid/path"
        )
        
        # Mock path existence
        with patch('pathlib.Path.exists', return_value=True):
            config.validate()  # Should not raise
        
        # Invalid type
        config.type = "invalid_type"
        with pytest.raises(ValueError, match="Invalid dataset type"):
            config.validate()
    
    def test_dataset_config_path_validation(self):
        """Test path validation for dataset configuration."""
        config = DatasetConfiguration(
            name="test_dataset",
            type="text",
            data_dir="/nonexistent/path"
        )
        
        # Mock path not existing
        with patch('pathlib.Path.exists', return_value=False):
            with pytest.raises(ValueError, match="Dataset directory does not exist"):
                config.validate()


class TestEvaluationConfiguration:
    """Test EvaluationConfiguration class."""
    
    def test_evaluation_config_basic(self):
        """Test basic evaluation configuration creation."""
        config = EvaluationConfiguration(
            metrics=["ndcg", "map", "recall"],
            k_values=[1, 5, 10],
            relevance_threshold=1,
            top_k=1000
        )
        
        assert config.metrics == ["ndcg", "map", "recall"]
        assert config.k_values == [1, 5, 10]
        assert config.relevance_threshold == 1
        assert config.top_k == 1000
    
    def test_evaluation_config_validation(self):
        """Test evaluation configuration validation."""
        # Valid configuration
        config = EvaluationConfiguration(
            metrics=["ndcg", "map"],
            k_values=[1, 5, 10]
        )
        config.validate()  # Should not raise
        
        # Invalid metric
        config.metrics = ["invalid_metric"]
        with pytest.raises(ValueError, match="Invalid metric"):
            config.validate()
        
        # Invalid k_values
        config.metrics = ["ndcg"]
        config.k_values = []
        with pytest.raises(ValueError, match="k_values cannot be empty"):
            config.validate()
    
    def test_evaluation_config_defaults(self):
        """Test evaluation configuration defaults."""
        config = EvaluationConfiguration()
        
        assert "ndcg" in config.metrics
        assert "map" in config.metrics
        assert 1 in config.k_values
        assert 5 in config.k_values
        assert 10 in config.k_values
        assert config.relevance_threshold == 1
        assert config.top_k == 1000


class TestOutputConfiguration:
    """Test OutputConfiguration class."""
    
    def test_output_config_basic(self):
        """Test basic output configuration creation."""
        config = OutputConfiguration(
            output_dir="./results",
            experiment_name="test_experiment",
            log_level="INFO",
            overwrite=False
        )
        
        assert config.output_dir == "./results"
        assert config.experiment_name == "test_experiment"
        assert config.log_level == "INFO"
        assert config.overwrite is False
    
    def test_output_config_validation(self):
        """Test output configuration validation."""
        # Valid configuration
        config = OutputConfiguration(
            output_dir="./results",
            experiment_name="valid_experiment"
        )
        config.validate()  # Should not raise
        
        # Invalid log level
        config.log_level = "INVALID"
        with pytest.raises(ValueError, match="Invalid log level"):
            config.validate()
        
        # Invalid experiment name
        config.log_level = "INFO"
        config.experiment_name = "invalid/name"
        with pytest.raises(ValueError, match="Invalid experiment name"):
            config.validate()


class TestExperimentConfig:
    """Test ExperimentConfig class."""
    
    def test_experiment_config_basic(self):
        """Test basic experiment configuration creation."""
        models = [ModelConfiguration(
            name="test_model", 
            type="sparse", 
            model_name_or_path=""
        )]
        datasets = [DatasetConfiguration(
            name="test_dataset", 
            type="text", 
            data_dir="/path"
        )]
        evaluation = EvaluationConfiguration()
        output = OutputConfiguration(
            output_dir="./results",
            experiment_name="test_experiment"
        )
        
        config = ExperimentConfig(
            models=models,
            datasets=datasets,
            evaluation=evaluation,
            output=output
        )
        
        assert len(config.models) == 1
        assert len(config.datasets) == 1
        assert config.models[0].name == "test_model"
        assert config.datasets[0].name == "test_dataset"
    
    def test_experiment_config_validation(self):
        """Test experiment configuration validation."""
        models = [ModelConfiguration(
            name="test_model", 
            type="sparse", 
            model_name_or_path=""
        )]
        datasets = [DatasetConfiguration(
            name="test_dataset", 
            type="text", 
            data_dir="/path"
        )]
        evaluation = EvaluationConfiguration()
        output = OutputConfiguration(
            output_dir="./results",
            experiment_name="test_experiment"
        )
        
        config = ExperimentConfig(
            models=models,
            datasets=datasets,
            evaluation=evaluation,
            output=output
        )
        
        # Mock validations
        with patch.object(ModelConfiguration, 'validate'), \
             patch.object(DatasetConfiguration, 'validate'), \
             patch.object(EvaluationConfiguration, 'validate'), \
             patch.object(OutputConfiguration, 'validate'):
            config.validate()  # Should not raise
    
    def test_experiment_config_cross_compatibility(self):
        """Test cross-compatibility validation."""
        # Compatible configurations
        models = [ModelConfiguration(
            name="text_model", 
            type="dense", 
            model_name_or_path="sentence-transformers/all-MiniLM-L6-v2"
        )]
        datasets = [DatasetConfiguration(
            name="text_dataset", 
            type="text", 
            data_dir="/path"
        )]
        evaluation = EvaluationConfiguration()
        output = OutputConfiguration(
            output_dir="./results",
            experiment_name="test_experiment"
        )
        
        config = ExperimentConfig(
            models=models,
            datasets=datasets,
            evaluation=evaluation,
            output=output
        )
        
        config.validate_cross_compatibility()  # Should not raise
    
    def test_experiment_config_serialization(self):
        """Test experiment configuration serialization."""
        models = [ModelConfiguration(
            name="test_model", 
            type="sparse", 
            model_name_or_path=""
        )]
        datasets = [DatasetConfiguration(
            name="test_dataset", 
            type="text", 
            data_dir="/path"
        )]
        evaluation = EvaluationConfiguration()
        output = OutputConfiguration(
            output_dir="./results",
            experiment_name="test_experiment"
        )
        
        config = ExperimentConfig(
            models=models,
            datasets=datasets,
            evaluation=evaluation,
            output=output,
            description="Test experiment"
        )
        
        # Test to_dict
        config_dict = config.to_dict()
        assert "models" in config_dict
        assert "datasets" in config_dict
        assert "evaluation" in config_dict
        assert "output" in config_dict
        assert config_dict["description"] == "Test experiment"
        
        # Test from_dict
        restored_config = ExperimentConfig.from_dict(config_dict)
        assert len(restored_config.models) == 1
        assert len(restored_config.datasets) == 1
        assert restored_config.description == "Test experiment"


class TestConfigFormats:
    """Test configuration file format handling."""
    
    def test_config_format_enum(self):
        """Test ConfigFormat enum."""
        assert ConfigFormat.YAML.value == "yaml"
        assert ConfigFormat.JSON.value == "json"
    
    def test_config_serialization_yaml(self):
        """Test YAML serialization."""
        config = ModelConfiguration(
            name="test_model",
            type="sparse",
            model_name_or_path=""
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config.to_dict(), f)
            temp_path = f.name
        
        try:
            # Read back and verify
            with open(temp_path, 'r') as f:
                loaded_dict = yaml.safe_load(f)
            
            restored_config = ModelConfiguration.from_dict(loaded_dict)
            assert restored_config.name == config.name
            assert restored_config.type == config.type
        finally:
            os.unlink(temp_path)
    
    def test_config_serialization_json(self):
        """Test JSON serialization."""
        config = ModelConfiguration(
            name="test_model",
            type="sparse",
            model_name_or_path=""
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config.to_dict(), f)
            temp_path = f.name
        
        try:
            # Read back and verify
            with open(temp_path, 'r') as f:
                loaded_dict = json.load(f)
            
            restored_config = ModelConfiguration.from_dict(loaded_dict)
            assert restored_config.name == config.name
            assert restored_config.type == config.type
        finally:
            os.unlink(temp_path)


# Fixtures for testing
@pytest.fixture
def sample_model_config():
    """Sample model configuration for testing."""
    return ModelConfiguration(
        name="test_model",
        type="sparse",
        model_name_or_path="",
        device="cpu",
        batch_size=32,
        parameters={"k1": 1.2, "b": 0.75}
    )


@pytest.fixture  
def sample_dataset_config():
    """Sample dataset configuration for testing."""
    return DatasetConfiguration(
        name="test_dataset",
        type="text",
        data_dir="/test/path",
        config_overrides={"cache_enabled": True}
    )


@pytest.fixture
def sample_evaluation_config():
    """Sample evaluation configuration for testing."""
    return EvaluationConfiguration(
        metrics=["ndcg", "map"],
        k_values=[1, 5, 10],
        relevance_threshold=1,
        top_k=1000
    )


@pytest.fixture
def sample_output_config():
    """Sample output configuration for testing."""
    return OutputConfiguration(
        output_dir="./test_results",
        experiment_name="test_experiment",
        log_level="INFO",
        overwrite=False
    )


@pytest.fixture
def sample_experiment_config(sample_model_config, sample_dataset_config, 
                           sample_evaluation_config, sample_output_config):
    """Sample experiment configuration for testing."""
    return ExperimentConfig(
        models=[sample_model_config],
        datasets=[sample_dataset_config],
        evaluation=sample_evaluation_config,
        output=sample_output_config,
        description="Test experiment configuration"
    )
