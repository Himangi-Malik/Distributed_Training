#!/usr/bin/env python3
"""
Validation script: tests all models and gradient sync algorithms.
"""
import sys
import torch
import traceback

sys.path.insert(0, "src")

from models import ann_model, cnn_model, rnn_model
from gradient_sync import ring, tree, parameter_server


class MockEndpoint:
    """Mock socket endpoint for local testing."""
    def __init__(self, name):
        self.name = name
        self.data = None

    def send(self, payload):
        self.data = payload
        print(f"  [mock send] {self.name} sent {type(payload).__name__}")

    def recv(self):
        print(f"  [mock recv] {self.name} received")
        if self.data is None:
            return {"gradients": torch.tensor([1.0, 2.0, 3.0])}
        return self.data

    def close(self):
        pass


# ============================================================================
# MODEL TESTS
# ============================================================================
def test_model(model_name, model_module):
    """Test a single model: build, train_step, output shape."""
    print(f"\n{'='*60}")
    print(f"Testing model: {model_name}")
    print(f"{'='*60}")
    try:
        config = {"lr": 0.01, "rank": 0}
        
        # Build model
        model_obj = model_module.build_model(config)
        print(f"✓ build_model succeeded")
        
        # Train step
        result = model_module.train_step(model_obj, config)
        print(f"✓ train_step succeeded")
        
        # Validate output
        assert isinstance(result, dict), "train_step must return dict"
        assert "gradients" in result, "result must have 'gradients' field"
        assert "loss" in result, "result must have 'loss' field"
        assert isinstance(result["gradients"], torch.Tensor), "gradients must be tensor"
        assert isinstance(result["loss"], (float, int)), "loss must be numeric"
        
        print(f"✓ output shape: {result['gradients'].shape}, loss: {result['loss']:.4f}")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        traceback.print_exc()
        return False


# ============================================================================
# GRADIENT SYNC TESTS
# ============================================================================
def test_algo_local(algo_name, algo_module):
    """Test a gradient sync algorithm in local mode."""
    print(f"\n{'='*60}")
    print(f"Testing algorithm (local mode): {algo_name}")
    print(f"{'='*60}")
    try:
        config = {
            "mode": "local",
            "rank": 0,
            "world_size": 2,
            "algo": algo_name,
            "lr": 0.01,
        }
        
        # Mock connections for local mode
        if algo_name == "ring":
            left_ep = MockEndpoint("left")
            right_ep = MockEndpoint("right")
            config["left_conn"] = left_ep
            config["right_conn"] = right_ep
            config["left_endpoint_info"] = {"peer_rank": 1}
            config["right_endpoint_info"] = {"peer_rank": 1}
        elif algo_name == "tree":
            config["left_child_conn"] = MockEndpoint("left_child")
            config["right_child_conn"] = MockEndpoint("right_child")
            config["parent_conn"] = MockEndpoint("parent")
        elif algo_name == "parameter_server":
            config["server_conn"] = MockEndpoint("server")
        
        # Setup
        comm_ctx = algo_module.setup(config)
        print(f"✓ setup succeeded")
        
        # Average
        local_grad = {
            "rank": 0,
            "gradients": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
            "loss": 0.5,
        }
        result = algo_module.average(local_grad, comm_ctx, config)
        print(f"✓ average succeeded")
        
        # Validate output
        assert isinstance(result, dict), "average must return dict"
        assert "gradients" in result, "result must have 'gradients' field"
        assert isinstance(result["gradients"], torch.Tensor), "gradients must be tensor"
        print(f"✓ output shape: {result['gradients'].shape}")
        
        # Teardown
        algo_module.teardown(comm_ctx)
        print(f"✓ teardown succeeded")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        traceback.print_exc()
        return False


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("\n" + "="*60)
    print("VALIDATION SUITE: Models and Gradient Sync Algorithms")
    print("="*60)
    
    results = {}
    
    # Test models
    print("\n" + "█"*60)
    print("SECTION: MODELS")
    print("█"*60)
    results["ann"] = test_model("ann", ann_model)
    results["cnn"] = test_model("cnn", cnn_model)
    results["rnn"] = test_model("rnn", rnn_model)
    
    # Test algorithms
    print("\n" + "█"*60)
    print("SECTION: GRADIENT SYNC ALGORITHMS (local mode)")
    print("█"*60)
    results["ring"] = test_algo_local("ring", ring)
    results["tree"] = test_algo_local("tree", tree)
    results["parameter_server"] = test_algo_local("parameter_server", parameter_server)
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} {name}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n✓ ALL VALIDATIONS PASSED - Ready for metrics collection!")
        return 0
    else:
        print("\n✗ SOME VALIDATIONS FAILED - Fix issues before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
