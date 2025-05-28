#!/bin/bash
"""
Setup script for NewAIBench Reporting System.

This script installs dependencies and sets up the reporting system.
"""

set -e  # Exit on any error

echo "NewAIBench Reporting System Setup"
echo "=================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check Python version
print_status "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
print_status "Python version: $python_version"

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    print_error "requirements.txt not found. Please run this script from the NewAIBench root directory."
    exit 1
fi

# Install basic dependencies
print_status "Installing basic dependencies..."
if pip3 install pandas > /dev/null 2>&1; then
    print_success "Basic dependencies installed"
else
    print_error "Failed to install basic dependencies"
    exit 1
fi

# Try to install visualization dependencies
print_status "Installing visualization dependencies (optional)..."
if pip3 install matplotlib seaborn > /dev/null 2>&1; then
    print_success "Visualization dependencies installed"
    visualization_available=true
else
    print_warning "Failed to install visualization dependencies. Charts will not be available."
    visualization_available=false
fi

# Create results directory
print_status "Creating results directory..."
mkdir -p results
mkdir -p reports
mkdir -p demo_reports
print_success "Directories created"

# Run tests
print_status "Running reporting system tests..."
if python3 tests/test_reporting.py > /dev/null 2>&1; then
    print_success "All tests passed"
else
    print_warning "Some tests failed. Check tests/test_reporting.py for details."
fi

# Run demo
print_status "Running reporting system demo..."
if python3 examples/reporting_demo.py; then
    print_success "Demo completed successfully"
else
    print_error "Demo failed"
    exit 1
fi

# Display summary
echo ""
echo "Setup Summary:"
echo "=============="
print_success "NewAIBench Reporting System is ready!"
echo ""
echo "Available features:"
echo "  ✓ Storage System (filesystem + SQLite)"
echo "  ✓ Results Aggregation and Analysis"
echo "  ✓ Report Generation (CSV, Markdown, LaTeX)"
if [ "$visualization_available" = true ]; then
    echo "  ✓ Visualization and Charts"
else
    echo "  ✗ Visualization and Charts (dependencies missing)"
fi
echo "  ✓ Command-line Interface"
echo "  ✓ Integration with ExperimentRunner"
echo ""
echo "Quick start commands:"
echo "  # Import existing results:"
echo "  python -m newaibench.reporting.cli import ./old_results"
echo ""
echo "  # Generate report:"
echo "  python -m newaibench.reporting.cli report --format all --output-dir ./reports"
echo ""
echo "  # Run demo again:"
echo "  python examples/reporting_demo.py"
echo ""
echo "  # Run tests:"
echo "  python tests/test_reporting.py"
echo ""
echo "Check ./demo_reports/ for sample generated reports!"
