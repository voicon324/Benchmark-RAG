#!/usr/bin/env python3
"""
Test suite for NewAIBench Reporting System.

This script provides comprehensive tests for all reporting functionality.
"""

import unittest
import tempfile
import shutil
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from newaibench.reporting import (
    ExperimentMetadata, RunResult,
    FileSystemStorage, SQLiteStorage, ResultsStorage,
    AggregationConfig, ResultsAggregator,
    ReportConfig, ReportGenerator
)
from newaibench.reporting.integration import ReportingIntegration, setup_default_reporting


class TestStorage(unittest.TestCase):
    """Test storage functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir)
    
    def test_filesystem_storage(self):
        """Test filesystem storage operations."""
        storage = FileSystemStorage(self.temp_dir)
        
        # Test storing result
        storage.store_result(
            experiment_name="test_exp",
            model="test_model",
            dataset="test_dataset", 
            metrics={"accuracy": 0.85},
            metadata={"timestamp": "2024-12-01T12:00:00"}
        )
        
        # Test listing experiments
        experiments = storage.list_experiments()
        self.assertIn("test_exp", experiments)
        
        # Test getting results
        results = storage.get_results(experiment_name="test_exp")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].model, "test_model")
        self.assertEqual(results[0].metrics["accuracy"], 0.85)
    
    def test_sqlite_storage(self):
        """Test SQLite storage operations."""
        db_path = Path(self.temp_dir) / "test.db"
        storage = SQLiteStorage(str(db_path))
        
        # Test storing result
        storage.store_result(
            experiment_name="test_exp",
            model="test_model",
            dataset="test_dataset",
            metrics={"accuracy": 0.85, "f1_score": 0.82},
            metadata={"timestamp": "2024-12-01T12:00:00"}
        )
        
        # Test listing experiments
        experiments = storage.list_experiments()
        self.assertIn("test_exp", experiments)
        
        # Test getting results
        results = storage.get_results(experiment_name="test_exp")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].model, "test_model")
        self.assertEqual(results[0].metrics["accuracy"], 0.85)
        
        # Test filtering
        filtered_results = storage.get_results(
            experiment_name="test_exp",
            model="test_model"
        )
        self.assertEqual(len(filtered_results), 1)
        
        # Test with non-existent filter
        empty_results = storage.get_results(
            experiment_name="test_exp",
            model="nonexistent"
        )
        self.assertEqual(len(empty_results), 0)
    
    def test_dual_storage(self):
        """Test dual storage backend."""
        fs_storage = FileSystemStorage(self.temp_dir)
        db_path = Path(self.temp_dir) / "test.db"
        sqlite_storage = SQLiteStorage(str(db_path))
        
        storage = ResultsStorage(
            filesystem_backend=fs_storage,
            sqlite_backend=sqlite_storage
        )
        
        # Store result
        storage.store_result(
            experiment_name="dual_test",
            model="model1",
            dataset="dataset1",
            metrics={"accuracy": 0.9},
            metadata={"config": {"temp": 0.7}}
        )
        
        # Check both backends have the data
        fs_experiments = fs_storage.list_experiments()
        sqlite_experiments = sqlite_storage.list_experiments()
        
        self.assertIn("dual_test", fs_experiments)
        self.assertIn("dual_test", sqlite_experiments)
        
        # Check results are consistent
        fs_results = fs_storage.get_results(experiment_name="dual_test")
        sqlite_results = sqlite_storage.get_results(experiment_name="dual_test")
        
        self.assertEqual(len(fs_results), 1)
        self.assertEqual(len(sqlite_results), 1)
        self.assertEqual(fs_results[0].metrics, sqlite_results[0].metrics)


class TestAggregation(unittest.TestCase):
    """Test aggregation functionality."""
    
    def setUp(self):
        """Set up test environment with sample data."""
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir)
        
        # Create storage with sample data
        self.storage = ResultsStorage(
            filesystem_backend=FileSystemStorage(self.temp_dir)
        )
        
        # Add sample results
        sample_data = [
            ("exp1", "model1", "dataset1", {"accuracy": 0.85, "f1_score": 0.82}),
            ("exp1", "model1", "dataset2", {"accuracy": 0.83, "f1_score": 0.80}),
            ("exp1", "model2", "dataset1", {"accuracy": 0.87, "f1_score": 0.84}),
            ("exp1", "model2", "dataset2", {"accuracy": 0.84, "f1_score": 0.81}),
            ("exp2", "model1", "dataset1", {"accuracy": 0.86, "f1_score": 0.83}),
        ]
        
        for exp, model, dataset, metrics in sample_data:
            self.storage.store_result(
                experiment_name=exp,
                model=model,
                dataset=dataset,
                metrics=metrics,
                metadata={"timestamp": datetime.now().isoformat()}
            )
        
        self.aggregator = ResultsAggregator(self.storage)
    
    def test_comparison_table(self):
        """Test comparison table generation."""
        config = AggregationConfig(experiments=["exp1"])
        
        df = self.aggregator.get_comparison_table(config)
        
        # Should have 4 rows for exp1 (2 models x 2 datasets)
        self.assertEqual(len(df), 4)
        
        # Should have expected columns
        expected_cols = ["experiment", "model", "dataset", "accuracy", "f1_score"]
        for col in expected_cols:
            self.assertIn(col, df.columns)
        
        # Check filtering by model
        config_filtered = AggregationConfig(
            experiments=["exp1"],
            models=["model1"]
        )
        df_filtered = self.aggregator.get_comparison_table(config_filtered)
        self.assertEqual(len(df_filtered), 2)  # model1 on 2 datasets
    
    def test_summary_statistics(self):
        """Test summary statistics generation."""
        config = AggregationConfig(experiments=["exp1"])
        
        stats_df = self.aggregator.get_summary_statistics(config)
        
        # Should have statistics for accuracy and f1_score
        self.assertIn("accuracy", stats_df.columns)
        self.assertIn("f1_score", stats_df.columns)
        
        # Should have standard statistics rows
        expected_stats = ["count", "mean", "std", "min", "max"]
        for stat in expected_stats:
            self.assertIn(stat, stats_df.index)
    
    def test_best_models(self):
        """Test best models finding."""
        config = AggregationConfig(experiments=["exp1"])
        
        best_models = self.aggregator.find_best_models(config, "accuracy")
        
        # Should return dict with model scores
        self.assertIsInstance(best_models, dict)
        self.assertIn("model1", best_models)
        self.assertIn("model2", best_models)
        
        # model2 should have higher average accuracy (0.855 vs 0.84)
        self.assertGreater(best_models["model2"], best_models["model1"])
    
    def test_performance_trends(self):
        """Test performance trends analysis."""
        config = AggregationConfig(models=["model1"])
        
        trends = self.aggregator.analyze_performance_trends(config)
        
        # Should have data for model1
        self.assertIn("model1", trends)
        
        # Should have trends for metrics
        model1_trends = trends["model1"]
        self.assertIn("accuracy", model1_trends)
        self.assertIn("f1_score", model1_trends)
        
        # Should have 2 data points (exp1 and exp2)
        self.assertEqual(len(model1_trends["accuracy"]), 2)


class TestReporting(unittest.TestCase):
    """Test report generation functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir)
        
        # Create storage with sample data
        self.storage = ResultsStorage(
            filesystem_backend=FileSystemStorage(self.temp_dir)
        )
        
        # Add sample results
        self.storage.store_result(
            experiment_name="report_test",
            model="test_model",
            dataset="test_dataset",
            metrics={"accuracy": 0.85, "f1_score": 0.82, "precision": 0.88},
            metadata={"timestamp": datetime.now().isoformat()}
        )
        
        self.aggregator = ResultsAggregator(self.storage)
        self.reporter = ReportGenerator(self.storage, self.aggregator)
    
    def test_csv_report_generation(self):
        """Test CSV report generation."""
        output_path = Path(self.temp_dir) / "test_report.csv"
        
        config = ReportConfig(title="Test Report")
        agg_config = AggregationConfig(experiments=["report_test"])
        
        generated_path = self.reporter.generate_csv_report(
            output_path=output_path,
            config=config,
            aggregation_config=agg_config
        )
        
        # Check file was created
        self.assertTrue(Path(generated_path).exists())
        
        # Check content
        import pandas as pd
        df = pd.read_csv(generated_path)
        self.assertEqual(len(df), 1)
        self.assertIn("accuracy", df.columns)
        self.assertEqual(df.iloc[0]["model"], "test_model")
    
    def test_markdown_report_generation(self):
        """Test Markdown report generation."""
        output_path = Path(self.temp_dir) / "test_report.md"
        
        config = ReportConfig(
            title="Test Markdown Report",
            include_metadata=True,
            include_statistics=True
        )
        agg_config = AggregationConfig(experiments=["report_test"])
        
        generated_path = self.reporter.generate_markdown_report(
            output_path=output_path,
            config=config,
            aggregation_config=agg_config
        )
        
        # Check file was created
        self.assertTrue(Path(generated_path).exists())
        
        # Check content
        with open(generated_path, 'r') as f:
            content = f.read()
        
        self.assertIn("# Test Markdown Report", content)
        self.assertIn("## Report Metadata", content)
        self.assertIn("## Summary Statistics", content)
        self.assertIn("## Results", content)
        self.assertIn("test_model", content)
    
    def test_latex_report_generation(self):
        """Test LaTeX report generation."""
        output_path = Path(self.temp_dir) / "test_report.tex"
        
        config = ReportConfig(title="Test LaTeX Report")
        agg_config = AggregationConfig(experiments=["report_test"])
        
        generated_path = self.reporter.generate_latex_report(
            output_path=output_path,
            config=config,
            aggregation_config=agg_config
        )
        
        # Check file was created
        self.assertTrue(Path(generated_path).exists())
        
        # Check content
        with open(generated_path, 'r') as f:
            content = f.read()
        
        self.assertIn("\\documentclass{article}", content)
        self.assertIn("\\title{Test LaTeX Report}", content)
        self.assertIn("\\begin{document}", content)
        self.assertIn("\\end{document}", content)
        self.assertIn("test\\_model", content)  # Escaped underscore
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_visualization_generation(self, mock_close, mock_savefig):
        """Test visualization generation (mocked)."""
        # Only test if matplotlib is available
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            self.skipTest("Matplotlib/Seaborn not available")
        
        output_dir = Path(self.temp_dir) / "charts"
        
        config = ReportConfig()
        agg_config = AggregationConfig(experiments=["report_test"])
        
        # Mock savefig to avoid actual file operations
        mock_savefig.return_value = None
        
        chart_files = self.reporter.generate_visualization(
            output_dir=output_dir,
            config=config,
            aggregation_config=agg_config,
            chart_types=["bar"]
        )
        
        # Should attempt to create charts
        self.assertTrue(mock_savefig.called)
        self.assertTrue(mock_close.called)
    
    def test_full_report_generation(self):
        """Test full report generation."""
        output_dir = Path(self.temp_dir) / "full_reports"
        
        config = ReportConfig(title="Full Test Report")
        agg_config = AggregationConfig(experiments=["report_test"])
        
        generated_files = self.reporter.generate_full_report(
            output_dir=output_dir,
            formats=["csv", "markdown"],  # Skip latex and viz for speed
            config=config,
            aggregation_config=agg_config,
            include_visualizations=False
        )
        
        # Should have generated CSV and Markdown
        self.assertIn("csv", generated_files)
        self.assertIn("markdown", generated_files)
        
        # Check files exist
        for file_list in generated_files.values():
            for file_path in file_list:
                self.assertTrue(Path(file_path).exists())


class TestIntegration(unittest.TestCase):
    """Test integration functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir)
    
    def test_setup_default_reporting(self):
        """Test default reporting setup."""
        integration = setup_default_reporting(self.temp_dir)
        
        self.assertIsInstance(integration, ReportingIntegration)
        self.assertIsInstance(integration.storage, ResultsStorage)
        
        # Should have both backends
        self.assertIsNotNone(integration.storage.filesystem_backend)
        self.assertIsNotNone(integration.storage.sqlite_backend)
    
    def test_capture_experiment_result(self):
        """Test experiment result capture."""
        integration = setup_default_reporting(self.temp_dir)
        
        integration.capture_experiment_result(
            experiment_name="capture_test",
            model="test_model",
            dataset="test_dataset",
            metrics={"accuracy": 0.9},
            config={"temperature": 0.7}
        )
        
        # Check result was stored
        experiments = integration.storage.list_experiments()
        self.assertIn("capture_test", experiments)
        
        results = integration.storage.get_results(experiment_name="capture_test")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].metrics["accuracy"], 0.9)
    
    def test_capture_from_output_dir(self):
        """Test capturing from output directory."""
        # Create mock output directory structure
        output_dir = Path(self.temp_dir) / "mock_output"
        run_dir = output_dir / "model1_dataset1_20241201_120000"
        run_dir.mkdir(parents=True)
        
        # Create mock evaluation.json
        eval_data = {
            "metrics": {"accuracy": 0.85, "f1_score": 0.82},
            "config": {"temperature": 0.7}
        }
        
        with open(run_dir / "evaluation.json", "w") as f:
            json.dump(eval_data, f)
        
        # Test capture
        integration = setup_default_reporting(self.temp_dir)
        captured_count = integration.capture_from_output_dir(
            str(output_dir),
            experiment_name="mock_experiment"
        )
        
        self.assertEqual(captured_count, 1)
        
        # Check result was stored
        experiments = integration.storage.list_experiments()
        self.assertIn("mock_experiment", experiments)
        
        results = integration.storage.get_results(experiment_name="mock_experiment")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].model, "model1")
        self.assertEqual(results[0].dataset, "dataset1")
    
    def test_auto_capture_setup(self):
        """Test automatic capture setup."""
        integration = setup_default_reporting(self.temp_dir)
        
        # Create mock experiment runner
        class MockRunner:
            def evaluate(self, model, dataset, output_dir=None):
                class MockResult:
                    def __init__(self):
                        self.metrics = {"accuracy": 0.85}
                return MockResult()
        
        runner = MockRunner()
        original_evaluate = runner.evaluate
        
        # Setup auto capture
        integration.setup_auto_capture(runner, "auto_test")
        
        # Method should be wrapped
        self.assertNotEqual(runner.evaluate, original_evaluate)
        
        # Test wrapped method
        class MockModel:
            name = "test_model"
        
        class MockDataset:
            name = "test_dataset"
        
        result = runner.evaluate(MockModel(), MockDataset())
        
        # Original functionality should work
        self.assertEqual(result.metrics["accuracy"], 0.85)
        
        # Result should be captured
        experiments = integration.storage.list_experiments()
        self.assertIn("auto_test", experiments)


class TestCLI(unittest.TestCase):
    """Test CLI functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir)
        
        # Create storage with sample data
        integration = setup_default_reporting(self.temp_dir)
        integration.capture_experiment_result(
            experiment_name="cli_test",
            model="cli_model",
            dataset="cli_dataset",
            metrics={"accuracy": 0.8}
        )
    
    def test_cli_imports(self):
        """Test CLI module imports."""
        try:
            from newaibench.reporting.cli import setup_storage, main
            self.assertTrue(True)  # Import successful
        except ImportError as e:
            self.fail(f"CLI import failed: {e}")
    
    def test_setup_storage_function(self):
        """Test CLI storage setup function."""
        from newaibench.reporting.cli import setup_storage
        
        # Test dual storage setup
        storage = setup_storage("dual", self.temp_dir)
        self.assertIsInstance(storage, ResultsStorage)
        
        # Should have both backends
        self.assertIsNotNone(storage.filesystem_backend)
        self.assertIsNotNone(storage.sqlite_backend)


def run_tests():
    """Run all tests."""
    # Create test suite
    test_classes = [
        TestStorage,
        TestAggregation, 
        TestReporting,
        TestIntegration,
        TestCLI
    ]
    
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Test Summary:")
    print(f"Ran {result.testsRun} tests")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nOverall Result: {'PASS' if success else 'FAIL'}")
    
    return success


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
