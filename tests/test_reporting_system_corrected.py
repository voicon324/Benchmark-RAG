"""
Test suite for the reporting system components - Updated to match actual implementation.
"""

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch
import uuid

# Import reporting modules
import sys
sys.path.append('/home/hkduy/NewAI/new_bench/src')

from newaibench.reporting.storage import ResultsStorage, ExperimentMetadata, RunResult
from newaibench.reporting.aggregator import ResultsAggregator, AggregationConfig, create_aggregation_config
from newaibench.reporting.reporter import ReportGenerator, ReportConfig


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
    
    def test_store_and_retrieve_experiment(self):
        """Test storing and retrieving experiment metadata."""
        # Create test metadata
        metadata = ExperimentMetadata(
            experiment_id="test_exp_001",
            experiment_name="test_experiment",
            description="Test experiment for validation",
            created_at=self.test_timestamp,
            author="test_user",
            tags=["test", "validation"]
        )
        
        # Store experiment
        self.storage.store_experiment_metadata(metadata)
        
        # Retrieve and verify
        stored_experiments = self.storage.list_experiments()
        self.assertGreater(len(stored_experiments), 0)
        
        retrieved_metadata = self.storage.get_experiment_metadata("test_exp_001")
        self.assertIsNotNone(retrieved_metadata)
        self.assertEqual(retrieved_metadata.experiment_name, "test_experiment")
        self.assertEqual(retrieved_metadata.author, "test_user")
    
    def test_store_and_retrieve_run_results(self):
        """Test storing and retrieving run results."""
        # First create experiment
        exp_id = "test_exp_002"
        metadata = ExperimentMetadata(
            experiment_id=exp_id,
            experiment_name="test_experiment_2",
            description="Test experiment 2",
            created_at=self.test_timestamp,
            author="test_user"
        )
        self.storage.store_experiment_metadata(metadata)
        
        # Create run result
        run_result = RunResult(
            run_id=str(uuid.uuid4()),
            experiment_id=exp_id,
            model_name="test_model",
            dataset_name="test_dataset",
            metrics={"ndcg@10": 0.85, "map@10": 0.75},
            execution_time=120.5,
            created_at=self.test_timestamp,
            config={"param1": "value1"},
            metadata={"notes": "test run"},
            success=True
        )
        
        # Store run result
        self.storage.store_run_result(run_result)
        
        # Retrieve and verify
        run_results = self.storage.get_run_results(exp_id)
        self.assertEqual(len(run_results), 1)
        
        retrieved_run = run_results[0]
        self.assertEqual(retrieved_run.model_name, "test_model")
        self.assertEqual(retrieved_run.dataset_name, "test_dataset")
        self.assertEqual(retrieved_run.metrics["ndcg@10"], 0.85)
    
    def test_list_experiments(self):
        """Test listing experiments."""
        # Initially empty or with existing data
        initial_count = len(self.storage.list_experiments())
        
        # Add some experiments
        for i in range(3):
            exp_id = f"experiment_{i}"
            metadata = ExperimentMetadata(
                experiment_id=exp_id,
                experiment_name=f"experiment_{i}",
                description=f"Test experiment {i}",
                created_at=self.test_timestamp,
                author="test_user"
            )
            self.storage.store_experiment_metadata(metadata)
        
        # Check experiments are listed
        experiments = self.storage.list_experiments()
        self.assertEqual(len(experiments), initial_count + 3)


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
        # Create experiment
        exp_id = "test_experiment"
        metadata = ExperimentMetadata(
            experiment_id=exp_id,
            experiment_name="test_experiment",
            description="Test experiment for aggregation",
            created_at=datetime.now(),
            author="test_user"
        )
        self.storage.store_experiment_metadata(metadata)
        
        models = ["bert", "roberta", "gpt"]
        datasets = ["dataset_a", "dataset_b"]
        
        for model in models:
            for dataset in datasets:
                for run_id in range(2):  # 2 runs per combination
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
                        run_id=str(uuid.uuid4()),
                        experiment_id=exp_id,
                        model_name=model,
                        dataset_name=dataset,
                        metrics=metrics,
                        execution_time=random.uniform(50, 200),
                        created_at=datetime.now() - timedelta(days=run_id),
                        config={"run_id": run_id},
                        metadata={},
                        success=True
                    )
                    
                    self.storage.store_run_result(run_result)
    
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
        self.storage = ResultsStorage(self.temp_dir)
        self.aggregator = ResultsAggregator(self.storage)
        self.generator = ReportGenerator(self.storage, self.aggregator)
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_basic_report_config(self):
        """Test creating basic report configuration."""
        config = ReportConfig(
            title="Test Report",
            include_metadata=True,
            include_statistics=True,
            metrics_precision=3
        )
        
        self.assertEqual(config.title, "Test Report")
        self.assertTrue(config.include_metadata)
        self.assertTrue(config.include_statistics)
        self.assertEqual(config.metrics_precision, 3)
    
    def test_generate_empty_report(self):
        """Test generating report with empty data."""
        config = ReportConfig(
            title="Empty Report",
            include_metadata=True
        )
        
        # Create aggregation config for empty results
        agg_config = AggregationConfig(
            experiments=[],
            metrics=["ndcg@10"],
            include_failed=False
        )
        
        # Test CSV generation with empty data
        output_path = os.path.join(self.temp_dir, "empty_report.csv")
        try:
            result = self.generator.generate_csv_report(output_path, config, agg_config)
            # Should handle empty data gracefully
            self.assertTrue(os.path.exists(output_path) or "No results found" in str(result))
        except ValueError as e:
            # Expected behavior for empty data
            self.assertIn("No results found", str(e))
    
    def test_summary_statistics_calculation(self):
        """Test summary statistics through aggregator."""
        # Store some sample data
        exp_metadata = ExperimentMetadata(
            experiment_id="test_exp",
            experiment_name="test_exp",
            description="Test experiment",
            created_at=datetime.now(),
            author="test",
            tags=["test"]
        )
        self.storage.store_experiment_metadata(exp_metadata)
        
        # Store sample run results
        for i, model in enumerate(['bert', 'roberta']):
            run_result = RunResult(
                run_id=f"run_{i}",
                experiment_id="test_exp",
                model_name=model,
                dataset_name="test_dataset",
                metrics={'ndcg@10': 0.8 + i * 0.05, 'map@10': 0.7 + i * 0.05},
                execution_time=10.5,
                created_at=datetime.now(),
                config={},
                metadata={},
                success=True
            )
            self.storage.store_run_result(run_result)
        
        # Test aggregation functionality
        config = AggregationConfig(
            experiments=["test_exp"],
            metrics=["ndcg@10", "map@10"],
            include_failed=False
        )
        
        time_analysis = self.aggregator.get_time_series_analysis(config)
        self.assertIsInstance(time_analysis, dict)
        self.assertIn('total_runs', time_analysis)
        self.assertEqual(time_analysis['total_runs'], 2)


def run_reporting_tests():
    """Run all reporting tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestResultsStorage,
        TestResultsAggregator,
        TestReportGenerator,
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
