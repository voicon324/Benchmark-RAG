"""
Unit tests for experiment runner functionality.

Tests the main ExperimentRunner class, experiment execution, result handling, 
and error scenarios.
"""

import pytest
import tempfile
import json
import logging
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from newaibench.experiment.runner import (
    ExperimentRunner,
    ExperimentResult,
    ExperimentError
)
from newaibench.experiment.config import (
    ExperimentConfig,
    ModelConfiguration,
    DatasetConfiguration,
    EvaluationConfiguration,
    OutputConfiguration
)


class TestExperimentResult:
    """Test ExperimentResult dataclass."""
    
    def test_experiment_result_creation(self):
        """Test creating experiment result."""
        result = ExperimentResult(
            model_name="test_model",
            dataset_name="test_dataset",
            metrics={"ndcg@10": 0.5, "map@10": 0.3},
            execution_time=45.6,
            success=True
        )
        
        assert result.model_name == "test_model"
        assert result.dataset_name == "test_dataset"
        assert result.metrics["ndcg@10"] == 0.5
        assert result.execution_time == 45.6
        assert result.success is True
        assert result.error is None
    
    def test_experiment_result_failure(self):
        """Test creating failed experiment result."""
        result = ExperimentResult(
            model_name="test_model",
            dataset_name="test_dataset",
            metrics={},
            execution_time=10.0,
            success=False,
            error="Test error message"
        )
        
        assert result.success is False
        assert result.error == "Test error message"
        assert result.metrics == {}


class TestExperimentRunner:
    """Test ExperimentRunner class."""
    
    @pytest.fixture
    def sample_config(self):
        """Create sample experiment configuration."""
        models = [ModelConfiguration(
            name="test_model",
            type="sparse",
            model_name_or_path="",
            device="cpu"
        )]
        
        datasets = [DatasetConfiguration(
            name="test_dataset",
            type="text",
            data_dir="/test/path"
        )]
        
        evaluation = EvaluationConfiguration(
            metrics=["ndcg", "map"],
            k_values=[1, 5, 10]
        )
        
        output = OutputConfiguration(
            output_dir="./test_results",
            experiment_name="test_experiment"
        )
        
        return ExperimentConfig(
            models=models,
            datasets=datasets,
            evaluation=evaluation,
            output=output
        )
    
    @pytest.fixture
    def mock_dataset_components(self):
        """Mock dataset components."""
        corpus = {
            "doc1": {"id": "doc1", "text": "Sample document 1"},
            "doc2": {"id": "doc2", "text": "Sample document 2"}
        }
        
        queries = {
            "q1": {"id": "q1", "text": "Sample query 1"},
            "q2": {"id": "q2", "text": "Sample query 2"}
        }
        
        qrels = {
            "q1": {"doc1": 1, "doc2": 0},
            "q2": {"doc1": 0, "doc2": 1}
        }
        
        return corpus, queries, qrels
    
    def test_experiment_runner_initialization(self, sample_config):
        """Test experiment runner initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            sample_config.output.output_dir = temp_dir
            
            with patch.object(ExperimentConfig, 'validate_cross_compatibility'):
                runner = ExperimentRunner(sample_config)
                
                assert runner.config == sample_config
                assert runner.logger is not None
                assert runner.results == []
                assert runner.experiment_dir.exists()
    
    @patch('newaibench.experiment.runner.create_dataset_loader')
    @patch('newaibench.experiment.runner.BM25Model')
    @patch('newaibench.experiment.runner.Evaluator')
    def test_single_experiment_execution(self, mock_evaluator, mock_model, 
                                       mock_loader, sample_config, mock_dataset_components):
        """Test single experiment execution."""
        corpus, queries, qrels = mock_dataset_components
        
        # Setup mocks
        mock_loader_instance = Mock()
        mock_loader_instance.load_corpus.return_value = corpus
        mock_loader_instance.load_queries.return_value = queries
        mock_loader_instance.load_qrels.return_value = qrels
        mock_loader_instance.validate_data.return_value = None
        mock_loader_instance.get_statistics.return_value = {
            'total_documents': 2,
            'total_queries': 2,
            'total_qrels': 2
        }
        mock_loader.return_value = mock_loader_instance
        
        mock_model_instance = Mock()
        mock_model_instance.predict.return_value = {
            "q1": {"doc1": 0.8, "doc2": 0.2},
            "q2": {"doc1": 0.3, "doc2": 0.7}
        }
        mock_model.return_value = mock_model_instance
        
        mock_evaluator_instance = Mock()
        mock_evaluator_instance.evaluate.return_value = {
            'metrics': {
                'ndcg@10': 0.5,
                'map@10': 0.3
            }
        }
        mock_evaluator.return_value = mock_evaluator_instance
        
        with tempfile.TemporaryDirectory() as temp_dir:
            sample_config.output.output_dir = temp_dir
            
            with patch.object(ExperimentConfig, 'validate_cross_compatibility'):
                runner = ExperimentRunner(sample_config)
                
                model_config = sample_config.models[0]
                dataset_config = sample_config.datasets[0]
                
                result = runner._run_single_experiment(model_config, dataset_config)
                
                assert result.success is True
                assert result.model_name == "test_model"
                assert result.dataset_name == "test_dataset"
                assert "ndcg@10" in result.metrics
                assert result.execution_time > 0
    
    @patch('newaibench.experiment.runner.create_dataset_loader')
    def test_single_experiment_failure(self, mock_loader, sample_config):
        """Test single experiment failure handling."""
        # Setup mock to raise exception
        mock_loader.side_effect = Exception("Dataset loading failed")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            sample_config.output.output_dir = temp_dir
            
            with patch.object(ExperimentConfig, 'validate_cross_compatibility'):
                runner = ExperimentRunner(sample_config)
                
                model_config = sample_config.models[0]
                dataset_config = sample_config.datasets[0]
                
                result = runner._run_single_experiment(model_config, dataset_config)
                
                assert result.success is False
                assert result.error is not None
                assert "Dataset loading failed" in result.error
                assert result.metrics == {}
    
    def test_model_creation_sparse(self, sample_config):
        """Test sparse model creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            sample_config.output.output_dir = temp_dir
            
            with patch.object(ExperimentConfig, 'validate_cross_compatibility'), \
                 patch('newaibench.experiment.runner.BM25Model') as mock_bm25:
                
                runner = ExperimentRunner(sample_config)
                model_config = sample_config.models[0]
                
                runner._create_model(model_config)
                
                mock_bm25.assert_called_once()
                call_args = mock_bm25.call_args[0][0]
                assert call_args['name'] == "test_model"
                assert call_args['device'] == "cpu"
    
    def test_model_creation_dense(self, sample_config):
        """Test dense model creation."""
        sample_config.models[0].type = "dense"
        sample_config.models[0].model_name_or_path = "sentence-transformers/all-MiniLM-L6-v2"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            sample_config.output.output_dir = temp_dir
            
            with patch.object(ExperimentConfig, 'validate_cross_compatibility'), \
                 patch('newaibench.experiment.runner.DenseTextRetriever') as mock_dense:
                
                runner = ExperimentRunner(sample_config)
                model_config = sample_config.models[0]
                
                runner._create_model(model_config)
                
                mock_dense.assert_called_once()
                call_args = mock_dense.call_args[0][0]
                assert call_args['name'] == "test_model"
                assert call_args['model_name_or_path'] == "sentence-transformers/all-MiniLM-L6-v2"
    
    def test_model_creation_image_ocr(self, sample_config):
        """Test OCR-based image model creation."""
        sample_config.models[0].type = "image_retrieval"
        sample_config.models[0].parameters = {"retrieval_method": "ocr"}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            sample_config.output.output_dir = temp_dir
            
            with patch.object(ExperimentConfig, 'validate_cross_compatibility'), \
                 patch('newaibench.experiment.runner.OCRBasedDocumentRetriever') as mock_ocr:
                
                runner = ExperimentRunner(sample_config)
                model_config = sample_config.models[0]
                
                runner._create_model(model_config)
                
                mock_ocr.assert_called_once()
    
    def test_model_creation_image_embedding(self, sample_config):
        """Test embedding-based image model creation."""
        sample_config.models[0].type = "image_retrieval"
        sample_config.models[0].model_name_or_path = "sentence-transformers/clip-ViT-B-32"
        sample_config.models[0].parameters = {"retrieval_method": "embedding"}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            sample_config.output.output_dir = temp_dir
            
            with patch.object(ExperimentConfig, 'validate_cross_compatibility'), \
                 patch('newaibench.experiment.runner.ImageEmbeddingDocumentRetriever') as mock_embedding:
                
                runner = ExperimentRunner(sample_config)
                model_config = sample_config.models[0]
                
                runner._create_model(model_config)
                
                mock_embedding.assert_called_once()
                call_args = mock_embedding.call_args[0][0]
                assert call_args['model_name_or_path'] == "sentence-transformers/clip-ViT-B-32"
    
    def test_model_creation_invalid_type(self, sample_config):
        """Test invalid model type handling."""
        sample_config.models[0].type = "invalid_type"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            sample_config.output.output_dir = temp_dir
            
            with patch.object(ExperimentConfig, 'validate_cross_compatibility'):
                runner = ExperimentRunner(sample_config)
                model_config = sample_config.models[0]
                
                with pytest.raises(ExperimentError, match="Failed to create model"):
                    runner._create_model(model_config)
    
    @patch('newaibench.experiment.runner.create_dataset_loader')
    def test_dataset_creation(self, mock_loader, sample_config, mock_dataset_components):
        """Test dataset creation."""
        corpus, queries, qrels = mock_dataset_components
        
        mock_loader_instance = Mock()
        mock_loader_instance.load_corpus.return_value = corpus
        mock_loader_instance.load_queries.return_value = queries
        mock_loader_instance.load_qrels.return_value = qrels
        mock_loader_instance.validate_data.return_value = None
        mock_loader_instance.get_statistics.return_value = {
            'total_documents': 2,
            'total_queries': 2,
            'total_qrels': 2
        }
        mock_loader.return_value = mock_loader_instance
        
        with tempfile.TemporaryDirectory() as temp_dir:
            sample_config.output.output_dir = temp_dir
            
            with patch.object(ExperimentConfig, 'validate_cross_compatibility'):
                runner = ExperimentRunner(sample_config)
                dataset_config = sample_config.datasets[0]
                
                loader, corpus_result, queries_result, qrels_result = runner._create_dataset(dataset_config)
                
                assert loader == mock_loader_instance
                assert corpus_result == corpus
                assert queries_result == queries
                assert qrels_result == qrels
    
    def test_retrieval_execution(self, sample_config):
        """Test retrieval execution."""
        mock_model = Mock()
        mock_model.predict.return_value = {
            "q1": {"doc1": 0.8, "doc2": 0.2},
            "q2": {"doc1": 0.3, "doc2": 0.7}
        }
        
        corpus = {"doc1": {}, "doc2": {}}
        queries = {"q1": {}, "q2": {}}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            sample_config.output.output_dir = temp_dir
            
            with patch.object(ExperimentConfig, 'validate_cross_compatibility'):
                runner = ExperimentRunner(sample_config)
                
                results = runner._run_retrieval(mock_model, corpus, queries, top_k=10)
                
                assert "q1" in results
                assert "q2" in results
                assert results["q1"]["doc1"] == 0.8
                assert results["q2"]["doc2"] == 0.7
    
    @patch('newaibench.experiment.runner.Evaluator')
    def test_results_evaluation(self, mock_evaluator, sample_config):
        """Test results evaluation."""
        mock_evaluator_instance = Mock()
        mock_evaluator_instance.evaluate.return_value = {
            'metrics': {
                'ndcg@10': 0.5,
                'map@10': 0.3,
                'recall@10': 0.4
            }
        }
        mock_evaluator.return_value = mock_evaluator_instance
        
        results = {
            "q1": {"doc1": 0.8, "doc2": 0.2},
            "q2": {"doc1": 0.3, "doc2": 0.7}
        }
        
        qrels = {
            "q1": {"doc1": 1, "doc2": 0},
            "q2": {"doc1": 0, "doc2": 1}
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            sample_config.output.output_dir = temp_dir
            
            with patch.object(ExperimentConfig, 'validate_cross_compatibility'):
                runner = ExperimentRunner(sample_config)
                
                metrics = runner._evaluate_results(results, qrels)
                
                assert 'ndcg@10' in metrics
                assert 'map@10' in metrics
                assert metrics['ndcg@10'] == 0.5
    
    def test_run_file_saving_trec(self, sample_config):
        """Test run file saving in TREC format."""
        sample_config.evaluation.save_run_file = True
        sample_config.evaluation.run_file_format = "trec"
        
        results = {
            "q1": {"doc1": 0.8, "doc2": 0.2},
            "q2": {"doc1": 0.3, "doc2": 0.7}
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            sample_config.output.output_dir = temp_dir
            
            with patch.object(ExperimentConfig, 'validate_cross_compatibility'):
                runner = ExperimentRunner(sample_config)
                
                run_file_path = runner._save_run_file(results, "test_model", "test_dataset")
                
                assert Path(run_file_path).exists()
                
                # Verify TREC format
                with open(run_file_path, 'r') as f:
                    lines = f.readlines()
                
                assert len(lines) == 4  # 2 queries × 2 docs
                assert "q1 Q0 doc1 1 0.800000 test_model" in lines[0]
    
    def test_run_file_saving_json(self, sample_config):
        """Test run file saving in JSON format."""
        sample_config.evaluation.save_run_file = True
        sample_config.evaluation.run_file_format = "json"
        
        results = {
            "q1": {"doc1": 0.8, "doc2": 0.2},
            "q2": {"doc1": 0.3, "doc2": 0.7}
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            sample_config.output.output_dir = temp_dir
            
            with patch.object(ExperimentConfig, 'validate_cross_compatibility'):
                runner = ExperimentRunner(sample_config)
                
                run_file_path = runner._save_run_file(results, "test_model", "test_dataset")
                
                assert Path(run_file_path).exists()
                
                # Verify JSON format
                with open(run_file_path, 'r') as f:
                    loaded_results = json.load(f)
                
                assert loaded_results == results
    
    def test_results_saving(self, sample_config):
        """Test experiment results saving."""
        with tempfile.TemporaryDirectory() as temp_dir:
            sample_config.output.output_dir = temp_dir
            
            with patch.object(ExperimentConfig, 'validate_cross_compatibility'):
                runner = ExperimentRunner(sample_config)
                
                # Add some mock results
                runner.results = [
                    ExperimentResult(
                        model_name="test_model",
                        dataset_name="test_dataset",
                        metrics={"ndcg@10": 0.5},
                        execution_time=10.0,
                        success=True
                    )
                ]
                
                runner._save_results()
                
                results_file = runner.experiment_dir / "results.json"
                assert results_file.exists()
                
                with open(results_file, 'r') as f:
                    saved_data = json.load(f)
                
                assert "experiment_config" in saved_data
                assert "results" in saved_data
                assert "summary" in saved_data
                assert len(saved_data["results"]) == 1
    
    def test_summary_generation(self, sample_config):
        """Test experiment summary generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            sample_config.output.output_dir = temp_dir
            
            with patch.object(ExperimentConfig, 'validate_cross_compatibility'):
                runner = ExperimentRunner(sample_config)
                
                # Add mock results
                runner.results = [
                    ExperimentResult(
                        model_name="model1",
                        dataset_name="dataset1",
                        metrics={"ndcg@10": 0.5, "map@10": 0.3},
                        execution_time=10.0,
                        success=True
                    ),
                    ExperimentResult(
                        model_name="model2",
                        dataset_name="dataset1",
                        metrics={"ndcg@10": 0.6, "map@10": 0.4},
                        execution_time=15.0,
                        success=True
                    ),
                    ExperimentResult(
                        model_name="model3",
                        dataset_name="dataset1",
                        metrics={},
                        execution_time=5.0,
                        success=False,
                        error="Test error"
                    )
                ]
                
                summary = runner._generate_summary()
                
                assert summary['total_experiments'] == 3
                assert summary['successful_experiments'] == 2
                assert summary['failed_experiments'] == 1
                assert summary['average_execution_time'] == 10.0
                assert 'average_metrics' in summary
                assert summary['average_metrics']['ndcg@10'] == 0.55  # (0.5 + 0.6) / 2
                assert 'failed_experiments_details' in summary


class TestExperimentError:
    """Test ExperimentError exception."""
    
    def test_experiment_error(self):
        """Test ExperimentError creation."""
        error = ExperimentError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)


# Integration tests with mocked dependencies
class TestExperimentRunnerIntegration:
    """Integration tests for ExperimentRunner."""
    
    @pytest.fixture
    def full_config(self):
        """Create full experiment configuration."""
        models = [
            ModelConfiguration(
                name="bm25_model",
                type="sparse", 
                model_name_or_path="",
                parameters={"k1": 1.2, "b": 0.75}
            ),
            ModelConfiguration(
                name="dense_model",
                type="dense",
                model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
                batch_size=16
            )
        ]
        
        datasets = [
            DatasetConfiguration(
                name="text_dataset",
                type="text",
                data_dir="/test/path1"
            ),
            DatasetConfiguration(
                name="image_dataset", 
                type="image",
                data_dir="/test/path2",
                config_overrides={"require_ocr_text": True}
            )
        ]
        
        evaluation = EvaluationConfiguration(
            metrics=["ndcg", "map", "recall"],
            k_values=[1, 5, 10],
            save_run_file=True
        )
        
        output = OutputConfiguration(
            output_dir="./test_results",
            experiment_name="integration_test",
            save_intermediate=True
        )
        
        return ExperimentConfig(
            models=models,
            datasets=datasets,
            evaluation=evaluation,
            output=output,
            description="Integration test experiment"
        )
    
    @patch('newaibench.experiment.runner.create_dataset_loader')
    @patch('newaibench.experiment.runner.BM25Model')
    @patch('newaibench.experiment.runner.DenseTextRetriever')
    @patch('newaibench.experiment.runner.Evaluator')
    def test_full_experiment_run(self, mock_evaluator, mock_dense, mock_bm25, 
                               mock_loader, full_config):
        """Test full experiment run with multiple models and datasets."""
        # Setup dataset loader mock
        mock_loader_instance = Mock()
        mock_loader_instance.load_corpus.return_value = {"doc1": {}, "doc2": {}}
        mock_loader_instance.load_queries.return_value = {"q1": {}, "q2": {}}
        mock_loader_instance.load_qrels.return_value = {"q1": {"doc1": 1}}
        mock_loader_instance.validate_data.return_value = None
        mock_loader_instance.get_statistics.return_value = {
            'total_documents': 2, 'total_queries': 2, 'total_qrels': 1
        }
        mock_loader.return_value = mock_loader_instance
        
        # Setup model mocks
        mock_bm25_instance = Mock()
        mock_bm25_instance.predict.return_value = {"q1": {"doc1": 0.8}}
        mock_bm25.return_value = mock_bm25_instance
        
        mock_dense_instance = Mock()
        mock_dense_instance.predict.return_value = {"q1": {"doc1": 0.9}}
        mock_dense.return_value = mock_dense_instance
        
        # Setup evaluator mock
        mock_evaluator_instance = Mock()
        mock_evaluator_instance.evaluate.return_value = {
            'metrics': {'ndcg@10': 0.5, 'map@10': 0.3}
        }
        mock_evaluator.return_value = mock_evaluator_instance
        
        with tempfile.TemporaryDirectory() as temp_dir:
            full_config.output.output_dir = temp_dir
            
            with patch.object(ExperimentConfig, 'validate_cross_compatibility'):
                runner = ExperimentRunner(full_config)
                results = runner.run()
                
                # Should have 4 experiments (2 models × 2 datasets)
                assert len(results) == 4
                
                # All should be successful in this mock scenario
                assert all(r.success for r in results)
                
                # Check that results file was created
                results_file = runner.experiment_dir / "results.json"
                assert results_file.exists()
