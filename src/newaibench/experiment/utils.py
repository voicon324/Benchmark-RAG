"""
Utility functions for experiment configuration and result management.

This module provides helper functions for loading configurations, saving results,
and other experiment-related utilities.
"""

import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Union, List

from .config import ExperimentConfig, ConfigFormat


logger = logging.getLogger(__name__)


def load_experiment_config(config_path: Union[str, Path]) -> ExperimentConfig:
    """Load experiment configuration from file.
    
    Args:
        config_path: Path to configuration file (YAML or JSON)
        
    Returns:
        ExperimentConfig instance
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config format is invalid
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Determine format from extension
    if config_path.suffix.lower() in ['.yaml', '.yml']:
        format_type = ConfigFormat.YAML
    elif config_path.suffix.lower() == '.json':
        format_type = ConfigFormat.JSON
    else:
        raise ValueError(f"Unsupported config format. Use .yaml, .yml, or .json: {config_path}")
    
    # Load configuration
    with open(config_path, 'r', encoding='utf-8') as f:
        if format_type == ConfigFormat.YAML:
            try:
                import yaml
                config_dict = yaml.safe_load(f)
            except ImportError:
                raise ImportError("PyYAML is required for YAML config files. Install with: pip install PyYAML")
        else:
            config_dict = json.load(f)
    
    # Validate required sections
    required_sections = ['models', 'datasets', 'output']
    for section in required_sections:
        if section not in config_dict:
            raise ValueError(f"Missing required section '{section}' in config file")
    
    # Create ExperimentConfig
    try:
        return ExperimentConfig.from_dict(config_dict)
    except Exception as e:
        raise ValueError(f"Invalid configuration: {str(e)}")


def save_experiment_config(config: ExperimentConfig, save_path: Union[str, Path], 
                          format_type: ConfigFormat = ConfigFormat.YAML):
    """Save experiment configuration to file.
    
    Args:
        config: ExperimentConfig to save
        save_path: Path to save configuration
        format_type: Format to save in (YAML or JSON)
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    config_dict = config.to_dict()
    
    with open(save_path, 'w', encoding='utf-8') as f:
        if format_type == ConfigFormat.YAML:
            try:
                import yaml
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, indent=2)
            except ImportError:
                raise ImportError("PyYAML is required for YAML output. Install with: pip install PyYAML")
        else:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved experiment configuration to {save_path}")


def save_experiment_results(results: List[Dict[str, Any]], save_path: Union[str, Path]):
    """Save experiment results to JSON file.
    
    Args:
        results: List of experiment results
        save_path: Path to save results
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved experiment results to {save_path}")


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two configuration dictionaries recursively.
    
    Args:
        base_config: Base configuration
        override_config: Configuration to override with
        
    Returns:
        Merged configuration
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def validate_device_availability(device: str) -> str:
    """Validate and potentially adjust device configuration.
    
    Args:
        device: Requested device ('cpu', 'cuda', 'auto')
        
    Returns:
        Validated device string
    """
    if device == 'auto':
        try:
            import torch
            if torch.cuda.is_available():
                return 'cuda'
            else:
                return 'cpu'
        except ImportError:
            logger.warning("PyTorch not available, defaulting to CPU")
            return 'cpu'
    
    elif device == 'cuda':
        try:
            import torch
            if not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                return 'cpu'
        except ImportError:
            logger.warning("PyTorch not available, falling back to CPU")
            return 'cpu'
    
    return device


def create_experiment_template(template_type: str = 'basic') -> Dict[str, Any]:
    """Create a template experiment configuration.
    
    Args:
        template_type: Type of template ('basic', 'text_comparison', 'image_experiment')
        
    Returns:
        Template configuration dictionary
    """
    if template_type == 'basic':
        return {
            'description': 'Basic experiment template',
            'models': [
                {
                    'name': 'bm25_baseline',
                    'type': 'sparse',
                    'model_name_or_path': '',
                    'parameters': {
                        'k1': 1.2,
                        'b': 0.75
                    },
                    'device': 'auto'
                }
            ],
            'datasets': [
                {
                    'name': 'sample_dataset',
                    'type': 'text',
                    'data_dir': '/path/to/dataset',
                    'config_overrides': {}
                }
            ],
            'evaluation': {
                'metrics': ['ndcg', 'map', 'recall'],
                'k_values': [1, 5, 10, 100],
                'top_k': 1000
            },
            'output': {
                'output_dir': './experiment_results',
                'experiment_name': 'basic_experiment'
            }
        }
    
    elif template_type == 'text_comparison':
        return {
            'description': 'Compare sparse and dense text retrieval models',
            'models': [
                {
                    'name': 'bm25_baseline',
                    'type': 'sparse',
                    'model_name_or_path': '',
                    'parameters': {
                        'k1': 1.2,
                        'b': 0.75
                    }
                },
                {
                    'name': 'sentence_bert',
                    'type': 'dense',
                    'model_name_or_path': 'sentence-transformers/all-MiniLM-L6-v2',
                    'batch_size': 32,
                    'parameters': {
                        'normalize_embeddings': True
                    }
                }
            ],
            'datasets': [
                {
                    'name': 'text_dataset',
                    'type': 'text',
                    'data_dir': '/path/to/text_dataset',
                    'config_overrides': {
                        'cache_enabled': True
                    }
                }
            ],
            'evaluation': {
                'metrics': ['ndcg', 'map', 'recall', 'precision'],
                'k_values': [1, 3, 5, 10, 20, 100],
                'top_k': 1000,
                'save_run_file': True
            },
            'output': {
                'output_dir': './experiment_results',
                'experiment_name': 'text_model_comparison',
                'save_intermediate': True
            }
        }
    
    elif template_type == 'image_experiment':
        return {
            'description': 'Document image retrieval experiment with OCR',
            'models': [
                {
                    'name': 'ocr_bm25',
                    'type': 'image_retrieval',
                    'model_name_or_path': '',
                    'parameters': {
                        'retrieval_method': 'ocr',
                        'ocr_engine': 'tesseract',
                        'sparse_model_params': {
                            'k1': 1.2,
                            'b': 0.75
                        }
                    }
                },
                {
                    'name': 'image_embedding',
                    'type': 'image_retrieval',
                    'model_name_or_path': 'sentence-transformers/clip-ViT-B-32',
                    'parameters': {
                        'retrieval_method': 'embedding',
                        'image_preprocessing': {
                            'resize': [224, 224],
                            'normalize': True
                        }
                    }
                }
            ],
            'datasets': [
                {
                    'name': 'document_images',
                    'type': 'image',
                    'data_dir': '/path/to/image_dataset',
                    'config_overrides': {
                        'require_ocr_text': True,
                        'supported_image_formats': ['.jpg', '.png', '.pdf'],
                        'cache_enabled': True
                    }
                }
            ],
            'evaluation': {
                'metrics': ['ndcg', 'map', 'recall'],
                'k_values': [1, 5, 10, 50],
                'top_k': 100
            },
            'output': {
                'output_dir': './image_experiments',
                'experiment_name': 'image_retrieval_comparison'
            }
        }
    
    else:
        raise ValueError(f"Unknown template type: {template_type}")


def load_and_validate_paths(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Load configuration and validate that all paths exist.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        Validated configuration dictionary
        
    Raises:
        FileNotFoundError: If required paths don't exist
    """
    # Check dataset paths
    for dataset in config_dict.get('datasets', []):
        data_dir = Path(dataset['data_dir'])
        if not data_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    
    # Check output directory can be created
    output_dir = Path(config_dict['output']['output_dir'])
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise ValueError(f"Cannot create output directory {output_dir}: {e}")
    
    return config_dict


def setup_experiment_logging(log_level: str = 'INFO', log_file: Union[str, Path, None] = None):
    """Setup logging for experiments.
    
    Args:
        log_level: Logging level
        log_file: Optional log file path
    """
    # Setup root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logging.getLogger().addHandler(file_handler)
