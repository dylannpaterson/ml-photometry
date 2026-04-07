import torch
import numpy as np
import onnxruntime as ort
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Verify ONNX Parity for Pollux Handover")
    parser.add_argument("--artifact_dir", default="artifacts", help="Directory containing artifacts")
    parser.add_argument("--git_hash", required=True, help="Git hash of the artifact")
    parser.add_argument("--stage", type=int, default=0, help="Stage index")
    
    args = parser.parse_args()
    
    onnx_path = os.path.join(args.artifact_dir, f"stage{args.stage}_{args.git_hash}.onnx")
    input_path = os.path.join(args.artifact_dir, f"test_input_{args.git_hash}.npy")
    output_stars_path = os.path.join(args.artifact_dir, f"test_output_stars_{args.git_hash}.npy")
    output_bg_path = os.path.join(args.artifact_dir, f"test_output_bg_{args.git_hash}.npy")
    
    # 1. Load Data
    test_input = np.load(input_path)
    expected_stars = np.load(output_stars_path)
    expected_bg = np.load(output_bg_path)
    
    # 2. Run ONNX Inference
    print(f"🧪 Running ONNX inference on {onnx_path}...")
    session = ort.InferenceSession(onnx_path)
    onnx_inputs = {session.get_inputs()[0].name: test_input}
    onnx_outs = session.run(None, onnx_inputs)
    
    onnx_stars, onnx_bg = onnx_outs
    
    # 3. Compare
    print("⚖️ Comparing PyTorch vs ONNX outputs...")
    
    stars_match = np.allclose(expected_stars, onnx_stars, atol=1e-5)
    bg_match = np.allclose(expected_bg, onnx_bg, atol=1e-5)
    
    if stars_match:
        print("✅ Stars Output: MATCH")
    else:
        diff = np.abs(expected_stars - onnx_stars).max()
        print(f"❌ Stars Output: MISMATCH (Max Diff: {diff:.2e})")
        
    if bg_match:
        print("✅ Background Output: MATCH")
    else:
        diff = np.abs(expected_bg - onnx_bg).max()
        print(f"❌ Background Output: MISMATCH (Max Diff: {diff:.2e})")
        
    if stars_match and bg_match:
        print("\n🏆 PARITY VERIFIED! Model is safe for handover to Pollux.")
    else:
        print("\n🚨 PARITY FAILED! Do not hand over this model.")
        exit(1)

if __name__ == "__main__":
    main()
