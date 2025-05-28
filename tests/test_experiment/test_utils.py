"""
Unit tests for experiment utility functions.

Tests configuration loading/saving, template creation, device validation,
and other utility functions.
"""

import pytest
import tempfile
import json
import yaml
import os
from pathlib import Path
from unittest.mock import patch, Mock

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from newaibench.experiment.utils import (
    load_experiment_config,
    save_experiment_config,
    save_experiment_results,
    merge_configs,
    validate_device_availability,
    create_experiment_template,
    load_and_validate_paths,
    setup_experiment_logging
)
from newaibench.experiment.config import (
    ExperimentConfig,
    ModelConfiguration,
    DatasetConfiguration,
    EvaluationConfiguration,
    OutputConfiguration,
    ConfigFormat
)


class TestConfigLoading:
    """Test configuration loading functionality."""
    
    def test_load_yaml_config(self):
        """Test loading YAML configuration file."""
        config_data = {
            'description': 'Test experiment',
            'models': [{
                'name': 'test_model',
                'type': 'sparse',
                'model_name_or_path': '',
                'device': 'cpu'
            }],
            'datasets': [{
                'name': 'test_dataset',
                'type': 'text',
                'data_dir': '/test/path'
            }],
            'evaluation': {
                'metrics': ['ndcg', 'map'],
                'k_values': [1, 5, 10]
            },
            'output': {
                'output_dir': './results',
                'experiment_name': 'test_experiment'
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            config = load_experiment_config(temp_path)
            
            assert isinstance(config, ExperimentConfig)
            assert config.description == 'Test experiment'
            assert len(config.models) == 1
            assert config.models[0].name == 'test_model'
            assert len(config.datasets) == 1
            assert config.datasets[0].name == 'test_dataset'
        finally:
            os.unlink(temp_path)
    
    def test_load_json_config(self):
        """Test loading JSON configuration file."""
        config_data = {
            'description': 'Test experiment',
            'models': [{
                'name': 'test_model',
                'type': 'sparse',
                'model_name_or_path': '',
                'device': 'cpu'
            }],
            'datasets': [{
                'name': 'test_dataset',
                'type': 'text',
                'data_dir': '/test/path'
            }],
            'evaluation': {
                'metrics': ['ndcg', 'map'],
                'k_values': [1, 5, 10]
            },
            'output': {
                'output_dir': './results',
                'experiment_name': 'test_experiment'
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name
        
        try:
            config = load_experiment_config(temp_path)
            
            assert isinstance(config, ExperimentConfig)
            assert config.description == 'Test experiment'
            assert len(config.models) == 1
            assert len(config.datasets) == 1
        finally:
            os.unlink(temp_path)
    
    def test_load_config_missing_file(self):
        """Test loading non-existent configuration file."""
        with pytest.raises(FileNotFoundError):
            load_experiment_config('/nonexistent/config.yaml')
    
    def test_load_config_missing_required_section(self):
        """Test loading configuration with missing required sections."""
        config_data = {
            'models': [{
                'name': 'test_model',
                'type': 'sparse'
            }]
            # Missing 'datasets' and 'output' sections
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Missing required section"):
                load_experiment_config(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_load_config_invalid_yaml(self):
        """Test loading invalid YAML configuration."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            temp_path = f.name
        
        try:
            with pytest.raises(Exception):  # YAML parsing error
                load_experiment_config(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_load_config_yaml_import_error(self):
        """Test YAML loading when PyYAML is not available."""
        config_data = {'test': 'data'}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            # Write valid YAML manually
            f.write("test: data\n")
            temp_path = f.name
        
        try:
            with patch('newaibench.experiment.utils.yaml', None):
                with pytest.raises(ImportError, match="PyYAML is required"):
                    load_experiment_config(temp_path)
        finally:
            os.unlink(temp_path)


class TestConfigSaving:
    """Test configuration saving functionality."""
    
    def test_save_config_yaml(self):
        """Test saving configuration in YAML format."""
        config = ExperimentConfig(
            models=[ModelConfiguration(
                name='test_model',
                type='sparse',
                model_name_or_path=''
            )],
            datasets=[DatasetConfiguration(
                name='test_dataset',
                type='text',
                data_dir='/test/path'
            )],
            evaluation=EvaluationConfiguration(),
            output=OutputConfiguration(
                output_dir='./results',
                experiment_name='test_experiment'
            ),
            description='Test experiment'
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            save_experiment_config(config, temp_path, ConfigFormat.YAML)
            
            # Verify file was created and is valid YAML
            assert Path(temp_path).exists()
            with open(temp_path, 'r') as f:
                loaded_data = yaml.safe_load(f)
            
            assert loaded_data['description'] == 'Test experiment'
            assert len(loaded_data['models']) == 1
            assert loaded_data['models'][0]['name'] == 'test_model'
        finally:
            os.unlink(temp_path)
    
    def test_save_config_json(self):
        """Test saving configuration in JSON format."""
        config = ExperimentConfig(
            models=[ModelConfiguration(
                name='test_model',
                type='sparse',
                model_name_or_path=''
            )],
            datasets=[DatasetConfiguration(
                name='test_dataset',
                type='text',
                data_dir='/test/path'
            )],
            evaluation=EvaluationConfiguration(),
            output=OutputConfiguration(
                output_dir='./results',
                experiment_name='test_experiment'
            ),
            description='Test experiment'
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            save_experiment_config(config, temp_path, ConfigFormat.JSON)
            
            # Verify file was created and is valid JSON
            assert Path(temp_path).exists()
            with open(temp_path, 'r') as f:
                loaded_data = json.load(f)
            
            assert loaded_data['description'] == 'Test experiment'
            assert len(loaded_data['models']) == 1
            assert loaded_data['models'][0]['name'] == 'test_model'
        finally:
            os.unlink(temp_path)


class TestResultsSaving:
    """Test results saving functionality."""
    
    def test_save_experiment_results(self):
        """Test saving experiment results."""
        results = [
            {
                'model_name': 'test_model',
                'dataset_name': 'test_dataset',
                'metrics': {'ndcg@10': 0.5, 'map@10': 0.3},
                'success': True
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            save_experiment_results(results, temp_path)
            
            # Verify file was created and contains correct data
            assert Path(temp_path).exists()
            with open(temp_path, 'r') as f:
                loaded_results = json.load(f)
            
            assert len(loaded_results) == 1
            assert loaded_results[0]['model_name'] == 'test_model'
            assert loaded_results[0]['metrics']['ndcg@10'] == 0.5
        finally:
            os.unlink(temp_path)


class TestConfigMerging:
    """Test configuration merging functionality."""
    
    def test_merge_configs_basic(self):
        """Test basic configuration merging."""
        base_config = {
            'models': [{'name': 'model1', 'type': 'sparse'}],
            'datasets': [{'name': 'dataset1', 'type': 'text'}],
            'evaluation': {'metrics': ['ndcg'], 'k_values': [10]}
        }
        
        override_config = {
            'evaluation': {'metrics': ['ndcg', 'map'], 'top_k': 1000},
            'output': {'experiment_name': 'new_experiment'}
        }
        
        merged = merge_configs(base_config, override_config)
        
        assert merged['models'] == base_config['models']
        assert merged['datasets'] == base_config['datasets']
        assert merged['evaluation']['metrics'] == ['ndcg', 'map']
        assert merged['evaluation']['k_values'] == [10]  # Preserved from base
        assert merged['evaluation']['top_k'] == 1000  # Added from override
        assert merged['output']['experiment_name'] == 'new_experiment'
    
    def test_merge_configs_nested(self):
        """Test merging nested configurations."""
        base_config = {
            'models': [{
                'name': 'model1',
                'type': 'dense',
                'parameters': {'batch_size': 32, 'device': 'cpu'}
            }]
        }
        
        override_config = {
            'models': [{
                'parameters': {'batch_size': 64, 'max_length': 512}
            }]
        }
        
        merged = merge_configs(base_config, override_config)
        
        model_params = merged['models'][0]['parameters']
        assert model_params['batch_size'] == 64  # Overridden
        assert model_params['device'] == 'cpu'  # Preserved
        assert model_params['max_length'] == 512  # Added
    
    def test_merge_configs_list_replacement(self):
        """Test that lists are replaced, not merged."""
        base_config = {
            'evaluation': {'metrics': ['ndcg', 'map']}
        }
        
        override_config = {
            'evaluation': {'metrics': ['recall']}
        }
        
        merged = merge_configs(base_config, override_config)
        
        assert merged['evaluation']['metrics'] == ['recall']  # Completely replaced


class TestDeviceValidation:
    """Test device validation functionality."""
    
    @patch('torch.cuda.is_available')
    def test_validate_device_cuda_available(self, mock_cuda_available):
        """Test device validation when CUDA is available."""
        mock_cuda_available.return_value = True
        
        assert validate_device_availability('cpu') == 'cpu'
        assert validate_device_availability('cuda') == 'cuda'
        assert validate_device_availability('auto') == 'cuda'
    
    @patch('torch.cuda.is_available')
    def test_validate_device_cuda_unavailable(self, mock_cuda_available):
        """Test device validation when CUDA is unavailable."""
        mock_cuda_available.return_value = False
        
        assert validate_device_availability('cpu') == 'cpu'
        assert validate_device_availability('cuda') == 'cpu'  # Falls back to CPU
        assert validate_device_availability('auto') == 'cpu'
    
    def test_validate_device_invalid(self):
        """Test validation of invalid device."""
        with pytest.raises(ValueError, match="Invalid device"):
            validate_device_availability('invalid_device')
    
    @patch('newaibench.experiment.utils.torch', None)
    def test_validate_device_no_torch(self):
        """Test device validation when PyTorch is not available."""
        # Should default to CPU when torch is not available
        assert validate_device_availability('auto') == 'cpu'
        assert validate_device_availability('cuda') == 'cpu'


class TestTemplateCreation:
    """Test experiment template creation."""
    
    def test_create_basic_template(self):
        """Test creating basic experiment template."""
        template = create_experiment_template('basic')
        
        assert 'models' in template
        assert 'datasets' in template
        assert 'evaluation' in template
        assert 'output' in template
        
        assert len(template['models']) == 1
        assert template['models'][0]['type'] == 'sparse'
        assert template['models'][0]['name'] == 'bm25_baseline'
        
        assert len(template['datasets']) == 1
        assert template['datasets'][0]['type'] == 'text'
    
    def test_create_text_comparison_template(self):
        """Test creating text comparison template."""
        template = create_experiment_template('text_comparison')
        
        assert len(template['models']) == 2  # BM25 + Dense
        
        model_types = [model['type'] for model in template['models']]
        assert 'sparse' in model_types
        assert 'dense' in model_types
        
        assert template['evaluation']['metrics'] == ['ndcg', 'map', 'recall', 'precision']
    
    def test_create_image_experiment_template(self):
        """Test creating image experiment template."""
        template = create_experiment_template('image_experiment')
        
        assert len(template['models']) >= 1
        assert any(model['type'] == 'image_retrieval' for model in template['models'])
        
        assert len(template['datasets']) == 1
        assert template['datasets'][0]['type'] == 'image'
        assert template['datasets'][0]['config_overrides']['require_ocr_text'] is True
    
    def test_create_multimodal_template(self):
        """Test creating multimodal template."""
        template = create_experiment_template('multimodal')
        
        # Should have both text and image models
        model_types = [model['type'] for model in template['models']]
        assert 'dense' in model_types  # Text model
        assert 'image_retrieval' in model_types  # Image model
        
        # Should have both text and image datasets
        dataset_types = [dataset['type'] for dataset in template['datasets']]
        assert 'text' in dataset_types
        assert 'image' in dataset_types
    
    def test_create_invalid_template(self):
        """Test creating invalid template type."""
        with pytest.raises(ValueError, match="Unknown template type"):
            create_experiment_template('invalid_template_type')


class TestPathValidation:
    """Test path validation functionality."""
    
    def test_load_and_validate_paths_valid(self):
        """Test path validation with valid paths."""
        config_dict = {
            'datasets': [{
                'data_dir': '/valid/path'
            }],
            'output': {
                'output_dir': '/valid/output'
            }
        }
        
        with patch('pathlib.Path.exists', return_value=True):
            validated_config = load_and_validate_paths(config_dict)
            
            assert validated_config == config_dict  # Should be unchanged
    
    def test_load_and_validate_paths_expand_user(self):
        """Test path validation with user home expansion."""
        config_dict = {
            'datasets': [{
                'data_dir': '~/data'
            }],
            'output': {
                'output_dir': '~/results'
            }
        }
        
        validated_config = load_and_validate_paths(config_dict)
        
        # Paths should be expanded
        assert not validated_config['datasets'][0]['data_dir'].startswith('~')
        assert not validated_config['output']['output_dir'].startswith('~')
    
    def test_load_and_validate_paths_relative(self):
        """Test path validation with relative paths."""
        config_dict = {
            'datasets': [{
                'data_dir': './data'
            }],
            'output': {
                'output_dir': './results'
            }
        }
        
        validated_config = load_and_validate_paths(config_dict)
        
        # Paths should be converted to absolute
        assert Path(validated_config['datasets'][0]['data_dir']).is_absolute()
        assert Path(validated_config['output']['output_dir']).is_absolute()


class TestLoggingSetup:
    """Test logging setup functionality."""
    
    def test_setup_experiment_logging_basic(self):
        """Test basic logging setup."""
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            setup_experiment_logging('INFO')
            
            mock_logger.setLevel.assert_called_with(20)  # INFO level
            assert mock_logger.addHandler.call_count >= 1
    
    def test_setup_experiment_logging_with_file(self):
        """Test logging setup with file output."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            log_file = f.name
        
        try:
            with patch('logging.getLogger') as mock_get_logger:
                mock_logger = Mock()
                mock_get_logger.return_value = mock_logger
                
                setup_experiment_logging('DEBUG', log_file)
                
                mock_logger.setLevel.assert_called_with(10)  # DEBUG level
                assert mock_logger.addHandler.call_count >= 2  # Console + File
        finally:
            os.unlink(log_file)
    
    def test_setup_experiment_logging_invalid_level(self):
        """Test logging setup with invalid log level."""
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            # Should default to INFO for invalid levels
            setup_experiment_logging('INVALID_LEVEL')
            
            # Should still set up logging (may use default level)
            assert mock_logger.addHandler.call_count >= 1


# Integration tests
class TestUtilsIntegration:
    """Integration tests for utility functions."""
    
    def test_config_roundtrip_yaml(self):
        """Test saving and loading configuration roundtrip in YAML."""
        original_config = ExperimentConfig(
            models=[ModelConfiguration(
                name='test_model',
                type='dense',
                model_name_or_path='sentence-transformers/all-MiniLM-L6-v2',
                parameters={'batch_size': 32}
            )],
            datasets=[DatasetConfiguration(
                name='test_dataset',
                type='text',
                data_dir='/test/path',
                config_overrides={'cache_enabled': True}
            )],
            evaluation=EvaluationConfiguration(
                metrics=['ndcg', 'map'],
                k_values=[1, 5, 10]
            ),
            output=OutputConfiguration(
                output_dir='./results',
                experiment_name='roundtrip_test'
            ),
            description='Roundtrip test'
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save configuration
            save_experiment_config(original_config, temp_path, ConfigFormat.YAML)
            
            # Load configuration back
            loaded_config = load_experiment_config(temp_path)
            
            # Compare key attributes
            assert loaded_config.description == original_config.description
            assert len(loaded_config.models) == len(original_config.models)
            assert loaded_config.models[0].name == original_config.models[0].name
            assert loaded_config.models[0].type == original_config.models[0].type
            assert loaded_config.models[0].model_name_or_path == original_config.models[0].model_name_or_path
            
            assert len(loaded_config.datasets) == len(original_config.datasets)
            assert loaded_config.datasets[0].name == original_config.datasets[0].name
            assert loaded_config.datasets[0].type == original_config.datasets[0].type
            
            assert loaded_config.evaluation.metrics == original_config.evaluation.metrics
            assert loaded_config.evaluation.k_values == original_config.evaluation.k_values
            
            assert loaded_config.output.experiment_name == original_config.output.experiment_name
        finally:
            os.unlink(temp_path)
    
    def test_template_validation(self):
        """Test that created templates are valid configurations."""
        for template_type in ['basic', 'text_comparison', 'image_experiment', 'multimodal']:
            template_dict = create_experiment_template(template_type)
            
            # Should be able to create ExperimentConfig from template
            config = ExperimentConfig.from_dict(template_dict)
            
            # Basic validation
            assert len(config.models) >= 1
            assert len(config.datasets) >= 1
            assert config.evaluation is not None
            assert config.output is not None
            
            # Should have valid model and dataset types
            for model in config.models:
                assert model.type in ['sparse', 'dense', 'image_retrieval']
            
            for dataset in config.datasets:
                assert dataset.type in ['text', 'image']
