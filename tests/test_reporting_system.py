"""
Test suite for the reporting system components.
"""

import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

# Import reporting modules
import sys
sys.path.append('/home/hkduy/NewAI/new_bench/src')

from newaibench.reporting.storage import ResultsStorage, ExperimentMetadata, RunResult
from newaibench.reporting.aggregator import ResultsAggregator, AggregationConfig, create_aggregation_config
from newaibench.reporting.reporter import ReportGenerator, ReportConfig
from newaibench.reporting.integration import ReportingIntegration


class TestResultsStorage(unittest.TestCase):
    """Test cases for ResultsStorage."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage = ResultsStorage(self.temp_dir)
        self.test_timestamp = datetime.now()
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_store_and_retrieve_result(self):
        """Test storing and retrieving a single result."""
        # Create test data
        metadata = ExperimentMetadata(
            experiment_id="test_exp_001",
            experiment_name="test_experiment",
            description="Test experiment for validation",
            created_at=self.test_timestamp,
            author="test_user",
            tags=["test", "validation"]
        )
        
        run_result = RunResult(
            run_id="run_001",
            experiment_id="test_exp_001",
            model_name="test_model",
            dataset_name="test_dataset",
            metrics={"ndcg@10": 0.85, "map@10": 0.75},
            execution_time=120.5,
            created_at=self.test_timestamp,
            config={"param1": "value1"},
            metadata={"notes": "test run"},
            success=True
        )
        
        # Store experiment metadata
        self.storage.store_experiment_metadata(metadata)
        
        # Store result
        self.storage.store_run_result(run_result)
        
        # Retrieve and verify
        stored_experiments = self.storage.list_experiments()
        
        self.assertGreater(len(stored_experiments), 0)
        
        # Get run results
        run_results = self.storage.get_run_results("test_exp_001")
        
        self.assertEqual(len(run_results), 1)
        retrieved_run = run_results[0]
        self.assertEqual(retrieved_run.model_name, "test_model")
        self.assertEqual(retrieved_run.dataset_name, "test_dataset")
        self.assertEqual(retrieved_run.metrics["ndcg@10"], 0.85)
    
    def test_list_experiments(self):
        """Test listing experiments."""
        # Initially empty
        experiments = self.storage.list_experiments()
        self.assertEqual(len(experiments), 0)
        
        # Add some experiments
        for i in range(3):
            metadata = ExperimentMetadata(
                experiment_name=f"experiment_{i}",
                model=f"model_{i}",
                dataset=f"dataset_{i}",
                timestamp=self.test_timestamp,
                config={}
            )
            
            run_result = RunResult(
                metadata=metadata,
                results=EvaluationResults(metrics={}, execution_time=0),
                status="completed"
            )
            
            self.storage.store_result(run_result)
        
        # Check experiments are listed
        experiments = self.storage.list_experiments()
        self.assertEqual(len(experiments), 3)
        self.assertIn("experiment_0", experiments)
        self.assertIn("experiment_1", experiments)
        self.assertIn("experiment_2", experiments)
    
    def test_query_results_functionality(self):
        """Test query_results with pandas simulation."""
        # Create test data
        for model in ["model_a", "model_b"]:
            for dataset in ["dataset_1", "dataset_2"]:
                metadata = ExperimentMetadata(
                    experiment_name="test_exp",
                    model=model,
                    dataset=dataset,
                    timestamp=self.test_timestamp,
                    config={}
                )
                
                run_result = RunResult(
                    metadata=metadata,
                    results=EvaluationResults(
                        metrics={"ndcg@10": 0.8 if model == "model_a" else 0.7},
                        execution_time=100
                    ),
                    status="completed"
                )
                
                self.storage.store_result(run_result)
        
        # Test query (without pandas, should work)
        try:
            results_df = self.storage.query_results(
                experiments=["test_exp"],
                models=["model_a"],
                datasets=None,
                metrics=["ndcg@10"]
            )
            # Should return something (empty df or raise ImportError)
            self.assertIsNotNone(results_df)
        except ImportError:
            # Expected if pandas not available
            pass


class TestResultsAggregator(unittest.TestCase):
    """Test cases for ResultsAggregator."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage = ResultsStorage(self.temp_dir)
        self.aggregator = ResultsAggregator(self.storage)
        
        # Create sample data
        self._create_sample_data()
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_sample_data(self):
        """Create sample data for testing."""
        models = ["bert", "roberta", "gpt"]
        datasets = ["dataset_a", "dataset_b"]
        
        for model in models:
            for dataset in datasets:
                for run_id in range(2):  # 2 runs per combination
                    metadata = ExperimentMetadata(
                        experiment_name="test_experiment",
                        model=model,
                        dataset=dataset,
                        timestamp=datetime.now() - timedelta(days=run_id),
                        config={"run_id": run_id}
                    )
                    
                    # Simulate different performance levels
                    base_score = 0.7
                    if model == "bert":
                        base_score = 0.75
                    elif model == "roberta":
                        base_score = 0.80
                    elif model == "gpt":
                        base_score = 0.85
                    
                    # Add dataset difficulty factor
                    if dataset == "dataset_b":
                        base_score *= 0.9
                    
                    # Add some noise
                    import random
                    random.seed(42 + hash(model + dataset + str(run_id)))
                    noise = random.uniform(-0.05, 0.05)
                    
                    metrics = {
                        "ndcg@10": base_score + noise,
                        "map@10": (base_score + noise) * 0.9,
                        "recall@10": (base_score + noise) * 1.1
                    }
                    
                    run_result = RunResult(
                        metadata=metadata,
                        results=EvaluationResults(
                            metrics=metrics,
                            execution_time=random.uniform(50, 200)
                        ),
                        status="completed"
                    )
                    
                    self.storage.store_result(run_result)
    
    def test_aggregation_config_creation(self):
        """Test creating aggregation configurations."""
        config = create_aggregation_config(
            experiments=["test_experiment"],
            models=["bert", "roberta"],
            metrics=["ndcg@10", "map@10"]
        )
        
        self.assertEqual(config.experiments, ["test_experiment"])
        self.assertEqual(config.models, ["bert", "roberta"])
        self.assertEqual(config.metrics, ["ndcg@10", "map@10"])
        self.assertFalse(config.include_failed)
    
    def test_time_series_analysis(self):
        """Test time series analysis functionality."""
        config = AggregationConfig(
            experiments=["test_experiment"],
            metrics=["ndcg@10"]
        )
        
        analysis = self.aggregator.get_time_series_analysis(config)
        
        self.assertIsInstance(analysis, dict)
        self.assertIn('time_series', analysis)
        self.assertIn('total_runs', analysis)
        self.assertIn('date_range', analysis)
        
        # Should have some data
        self.assertGreater(analysis['total_runs'], 0)
    
    def test_export_aggregated_results(self):
        """Test exporting aggregated results."""
        config = AggregationConfig(
            experiments=["test_experiment"],
            metrics=["ndcg@10", "map@10"]
        )
        
        # Test JSON export
        output_path = Path(self.temp_dir) / "test_export.json"
        self.aggregator.export_aggregated_results(config, output_path, format='json')
        
        self.assertTrue(output_path.exists())
        
        # Verify content
        with open(output_path, 'r') as f:
            exported_data = json.load(f)
        
        self.assertIsInstance(exported_data, list)
        self.assertGreater(len(exported_data), 0)
        
        # Check structure of first record
        first_record = exported_data[0]
        self.assertIn('experiment', first_record)
        self.assertIn('model', first_record)
        self.assertIn('dataset', first_record)
        self.assertIn('metrics', first_record)


class TestReportGenerator(unittest.TestCase):
    """Test cases for ReportGenerator."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.generator = ReportGenerator()
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_basic_report_config(self):
        """Test creating basic report configuration."""
        config = ReportConfig(
            title="Test Report",
            formats=["markdown", "csv"],
            output_dir=self.temp_dir
        )
        
        self.assertEqual(config.title, "Test Report")
        self.assertIn("markdown", config.formats)
        self.assertIn("csv", config.formats)
    
    def test_generate_empty_report(self):
        """Test generating report with empty data."""
        config = ReportConfig(
            title="Empty Report",
            formats=["markdown"],
            output_dir=self.temp_dir
        )
        
        # Generate report with empty data
        output_files = self.generator.generate_report([], config)
        
        self.assertIsInstance(output_files, list)
        # Should still create files even with empty data
        self.assertGreater(len(output_files), 0)
    
    def test_summary_statistics_calculation(self):
        """Test summary statistics calculation."""
        # Create sample data
        sample_data = [
            {
                'model': 'bert',
                'dataset': 'test',
                'ndcg@10': 0.8,
                'map@10': 0.7
            },
            {
                'model': 'roberta',
                'dataset': 'test',
                'ndcg@10': 0.85,
                'map@10': 0.75
            }
        ]
        
        stats = self.generator._calculate_summary_statistics(sample_data)
        
        self.assertIsInstance(stats, dict)
        self.assertIn('total_runs', stats)
        self.assertIn('unique_models', stats)
        self.assertIn('unique_datasets', stats)
        
        self.assertEqual(stats['total_runs'], 2)
        self.assertEqual(stats['unique_models'], 2)
        self.assertEqual(stats['unique_datasets'], 1)


class TestReportingIntegration(unittest.TestCase):
    """Test cases for ReportingIntegration."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.integration = ReportingIntegration(storage_path=self.temp_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_capture_experiment_result(self):
        """Test capturing experiment results."""
        # Mock experiment result
        mock_result = {
            'experiment_name': 'test_exp',
            'model_name': 'test_model',
            'dataset_name': 'test_dataset',
            'metrics': {'ndcg@10': 0.8},
            'execution_time': 120,
            'config': {'param1': 'value1'},
            'status': 'completed'
        }
        
        # Capture result
        self.integration.capture_experiment_result(mock_result)
        
        # Verify it was stored
        experiments = self.integration.storage.list_experiments()
        self.assertIn('test_exp', experiments)
        
        exp_results = self.integration.storage.get_experiment_results('test_exp')
        self.assertEqual(len(exp_results.runs), 1)
        self.assertEqual(exp_results.runs[0].metadata.model, 'test_model')
    
    def test_generate_experiment_report(self):
        """Test generating report for specific experiment."""
        # First capture some results
        for i in range(3):
            mock_result = {
                'experiment_name': 'report_test',
                'model_name': f'model_{i}',
                'dataset_name': 'test_dataset',
                'metrics': {'ndcg@10': 0.8 + i * 0.05},
                'execution_time': 100 + i * 10,
                'config': {},
                'status': 'completed'
            }
            self.integration.capture_experiment_result(mock_result)
        
        # Generate report
        output_files = self.integration.generate_experiment_report(
            experiment_name='report_test',
            output_dir=self.temp_dir,
            formats=['markdown', 'csv']
        )
        
        self.assertIsInstance(output_files, list)
        self.assertGreater(len(output_files), 0)
        
        # Check files were created
        for file_path in output_files:
            self.assertTrue(Path(file_path).exists())


def run_reporting_tests():
    """Run all reporting tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestResultsStorage,
        TestResultsAggregator,
        TestReportGenerator,
        TestReportingIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    # Run tests when script is executed directly
    success = run_reporting_tests()
    if success:
        print("\n✅ All reporting tests passed!")
        exit(0)
    else:
        print("\n❌ Some tests failed!")
        exit(1)
