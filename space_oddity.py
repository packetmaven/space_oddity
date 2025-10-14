#!/usr/bin/env python3
"""
Binary Analyzer with Real Embedding Drift Analysis

This script combines:
1. Real binary analysis (from complete_binary_analyzer.py)
2. Iterative binary mutation (actual byte-level modifications)
3. Embedding drift visualization (tracks how mutations change semantic embeddings)

Takes real binaries as input and shows how adversarial mutations affect their
embedding representations in the CodeBERT semantic space.
"""

import os
import sys
import argparse
import torch
import capstone
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

class BinaryAnalyzerWithDrift:
    """
    binary analyzer that performs real mutations and tracks embedding drift.
    """
    
    def __init__(self, model_name="microsoft/codebert-base"):
        print("???? Initializing Binary Analyzer with Drift Tracking...")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        self.disassembler = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
        self.disassembler.skipdata = True
        
        print(f"   ??? Model: {model_name}")
        print(f"   ??? Device: {self.device}")
    
    def embed_binary(self, code_bytes, max_length=512):
        """Generate CodeBERT embedding for binary code."""
        try:
            hex_tokens = [f"{b:02x}" for b in code_bytes[:max_length*2]]
            hex_string = " ".join(hex_tokens[:max_length])
            
            tokens = self.tokenizer(
                hex_string,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**tokens)
            
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            return embedding
            
        except Exception as e:
            print(f"   ??? Embedding error: {e}")
            return None
    
    def mutate_binary(self, binary_data, mutation_type='nop', intensity='conservative'):
        """
        Apply controlled mutations to binary data with similarity preservation.
        
        Mutation types:
        - 'nop': Insert NOP instructions (0x90) - minimal impact
        - 'pad': Add padding bytes - moderate impact
        
        Intensity:
        - 'conservative': 1-2 bytes per mutation (preserves similarity > 0.98)
        - 'moderate': 3-5 bytes per mutation (similarity > 0.95)
        - 'aggressive': 5-10 bytes per mutation (may break similarity)
        """
        mutated = bytearray(binary_data)
        
        # Adjust mutation size based on intensity
        if intensity == 'conservative':
            mutation_count = 2
        elif intensity == 'moderate':
            mutation_count = 4
        else:  # aggressive
            mutation_count = 8
        
        if mutation_type == 'nop':
            # Insert minimal NOPs to preserve similarity
            for _ in range(mutation_count):
                pos = np.random.randint(0, len(mutated))
                mutated.insert(pos, 0x90)  # NOP instruction
        
        elif mutation_type == 'pad':
            # Add minimal padding bytes
            padding = np.random.bytes(mutation_count)
            mutated.extend(padding)
        
        return bytes(mutated)
    
    def analyze_with_drift(self, binary_path, num_mutations=8, intensity='conservative'):
        """
        Analyze a real binary and track embedding drift through mutations.
        
        FIXED: Added intensity control and early stopping
        
        Args:
            binary_path: Path to binary file
            num_mutations: Number of mutation steps to perform
            intensity: Mutation intensity ('conservative', 'moderate', 'aggressive')
        
        Returns:
            dict: Analysis results with drift data
        """
        print(f"\n???? Analyzing binary with drift tracking: {binary_path}")
        
        # Load original binary
        try:
            with open(binary_path, 'rb') as f:
                original_data = f.read()
            print(f"   ???? Original size: {len(original_data)} bytes")
        except Exception as e:
            print(f"   ??? Error loading binary: {e}")
            return None
        
        # Track embeddings through mutations
        embeddings = []
        labels = []
        binary_versions = [original_data]
        
        # Original embedding
        print("   ???? Generating original embedding...")
        orig_emb = self.embed_binary(original_data)
        if orig_emb is None:
            return None
        
        embeddings.append(orig_emb)
        labels.append("Original")
        
        # Iterative mutations with SIMILARITY-BASED EARLY STOPPING
        print(f"   ???? Performing up to {num_mutations} mutation iterations...")
        print(f"      (Will stop early if similarity drops below 0.95)")
        current_binary = original_data
        
        for i in range(num_mutations):
            # Alternate mutation types with conservative intensity
            mutation_types = ['nop', 'pad', 'nop', 'pad']
            mut_type = mutation_types[i % len(mutation_types)]
            
            # Apply mutation with specified intensity
            current_binary = self.mutate_binary(current_binary, mut_type, intensity=intensity)
            binary_versions.append(current_binary)
            
            # Generate embedding
            mut_emb = self.embed_binary(current_binary)
            if mut_emb is not None:
                embeddings.append(mut_emb)
                labels.append(f"Mutation {i+1}")
                
                # Calculate similarity to original
                similarity = cosine_similarity(
                    orig_emb.reshape(1, -1),
                    mut_emb.reshape(1, -1)
                )[0][0]
                
                print(f"      Step {i+1}: Size={len(current_binary)}b, Similarity={similarity:.4f}")
                
                # EARLY STOPPING: Stop if similarity drops too low
                if similarity < 0.95:
                    print(f"      ??????  Similarity dropped below 0.95 - stopping to preserve functionality")
                    print(f"      Completed {i+1} iterations (early stop)")
                    break
        
        return {
            'embeddings': embeddings,
            'labels': labels,
            'binary_versions': binary_versions,
            'original_size': len(original_data)
        }
    
    def visualize_drift(self, results, binary_name, save_path="embedding_drift_analysis.png"):
        """Create clear, actionable embedding drift visualization for red team ops."""
        if not results or len(results['embeddings']) < 2:
            print("??? Insufficient data for visualization")
            return
        
        # Apply PCA
        embeddings_array = np.stack(results['embeddings'])
        pca = PCA(n_components=2)
        proj = pca.fit_transform(embeddings_array)
        
        # Calculate drift metrics
        orig_emb = results['embeddings'][0]
        distances = [np.linalg.norm(results['embeddings'][i] - orig_emb) 
                    for i in range(len(results['embeddings']))]
        similarities = [cosine_similarity(orig_emb.reshape(1, -1), 
                                         results['embeddings'][i].reshape(1, -1))[0][0]
                       for i in range(len(results['embeddings']))]
        
        # Create professional red team visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=120)
        
        # LEFT PANEL: Embedding Space Drift Path
        n_points = len(proj)
        
        # Plot points with clear color progression (dark ??? bright)
        for i in range(n_points):
            color_intensity = i / (n_points - 1)  # 0 to 1
            color = plt.cm.RdYlGn_r(color_intensity)  # Red (original) ??? Yellow ??? Green (mutated)
            
            ax1.scatter(proj[i, 0], proj[i, 1],
                       c=[color], s=200, alpha=0.9,
                       edgecolors='black', linewidths=2,
                       zorder=10)
        
        # Draw clear path with arrows
        for i in range(n_points - 1):
            ax1.annotate('', xy=(proj[i+1, 0], proj[i+1, 1]),
                        xytext=(proj[i, 0], proj[i, 1]),
                        arrowprops=dict(arrowstyle='->', lw=2, color='darkblue', alpha=0.6))
        
        # Calculate appropriate offsets based on data range
        x_range = proj[:, 0].max() - proj[:, 0].min()
        y_range = proj[:, 1].max() - proj[:, 1].min()
        
        # Use percentage-based offsets for better positioning
        x_offset = x_range * 0.05 if x_range > 0 else 0.1
        y_offset = y_range * 0.08 if y_range > 0 else 0.1
        
        # Clear labels - start, midpoint, end (no emojis - professional)
        ax1.text(proj[0, 0], proj[0, 1] + y_offset, "Original\nDetectable",
                fontsize=9, ha='center', va='bottom', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#ffcccc', alpha=0.9, edgecolor='#8B0000', linewidth=2))
        
        mid_idx = n_points // 2
        # Position midpoint label to the side if points are vertically aligned
        if abs(proj[mid_idx, 0] - proj[0, 0]) < x_range * 0.1:  # Vertically aligned
            ax1.text(proj[mid_idx, 0] + x_offset, proj[mid_idx, 1], f"Mutation {mid_idx}\nTransitioning",
                    fontsize=9, ha='left', va='center', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='#ffffcc', alpha=0.9, edgecolor='#E67E22', linewidth=2))
        else:
            ax1.text(proj[mid_idx, 0], proj[mid_idx, 1] - y_offset, f"Mutation {mid_idx}\nTransitioning",
                    fontsize=9, ha='center', va='top', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='#ffffcc', alpha=0.9, edgecolor='#E67E22', linewidth=2))
        
        ax1.text(proj[-1, 0], proj[-1, 1] + y_offset, f"Final State\nEvaded",
                fontsize=9, ha='center', va='bottom', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#ccffcc', alpha=0.9, edgecolor='#27AE60', linewidth=2))
        
        ax1.set_title(f'Embedding Drift Path: {binary_name}',
                     fontsize=13, fontweight='bold', color='#2C3E50')
        ax1.set_xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)',
                      fontsize=11, fontweight='bold')
        ax1.set_ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)',
                      fontsize=11, fontweight='bold')
        ax1.grid(True, linestyle='--', alpha=0.3)
        
        # Set reasonable axis limits with padding for labels
        x_range = proj[:, 0].max() - proj[:, 0].min()
        y_range = proj[:, 1].max() - proj[:, 1].min()
        
        # Add 20% padding on each side to ensure labels don't fall off
        x_padding = x_range * 0.2 if x_range > 1e-10 else 1
        y_padding = y_range * 0.2 if y_range > 1e-10 else 1
        
        ax1.set_xlim(proj[:, 0].min() - x_padding, proj[:, 0].max() + x_padding)
        ax1.set_ylim(proj[:, 1].min() - y_padding, proj[:, 1].max() + y_padding)
        
        # RIGHT PANEL: Red Team Intelligence Metrics
        ax2.axis('off')
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        
        # Title (no emoji - professional)
        ax2.text(0.5, 0.95, 'RED TEAM INTELLIGENCE',
                ha='center', fontsize=13, fontweight='bold', color='#8B0000')
        
        # Mutation metrics
        y_pos = 0.85
        line_spacing = 0.08
        
        ax2.text(0.5, y_pos, 'EVASION METRICS', ha='center', fontsize=11,
                fontweight='bold', color='#2C3E50')
        y_pos -= line_spacing * 1.5
        
        final_similarity = similarities[-1]
        
        # Professional status determination (no emojis)
        if final_similarity > 0.98:
            status_text = 'EVADED'
            status_color = '#27AE60'
        elif final_similarity > 0.95:
            status_text = 'PARTIAL EVASION'
            status_color = '#E67E22'
        else:
            status_text = 'DETECTED'
            status_color = '#E74C3C'
        
        # Evasion metrics with professional styling
        ax2.text(0.05, y_pos, f'Final Similarity: {final_similarity:.4f}', fontsize=10)
        y_pos -= line_spacing
        
        ax2.text(0.05, y_pos, f'L2 Distance: {distances[-1]:.4f}', fontsize=10)
        y_pos -= line_spacing
        
        ax2.text(0.05, y_pos, f'Status: {status_text}', fontsize=10, fontweight='bold', color=status_color)
        y_pos -= line_spacing * 2
        
        # Mutation strategy
        ax2.text(0.5, y_pos, 'MUTATION STRATEGY', ha='center', fontsize=11,
                fontweight='bold', color='#2C3E50')
        y_pos -= line_spacing * 1.5
        
        ax2.text(0.05, y_pos, f'Iterations: {n_points - 1}', fontsize=10)
        y_pos -= line_spacing
        
        size_increase = len(results['binary_versions'][-1]) - results['original_size']
        ax2.text(0.05, y_pos, f'Size increase: +{size_increase} bytes ({size_increase/results["original_size"]*100:.1f}%)', fontsize=10)
        y_pos -= line_spacing
        
        ax2.text(0.05, y_pos, 'Techniques: NOP insertion, padding', fontsize=10)
        y_pos -= line_spacing * 2
        
        # Tactical guidance (professional - no emojis)
        ax2.text(0.5, y_pos, 'TACTICAL GUIDANCE', ha='center', fontsize=11,
                fontweight='bold', color='#2C3E50')
        y_pos -= line_spacing * 1.5
        
        if final_similarity > 0.98:
            guidance = 'RECOMMENDED: Payload ready for deployment\nEvasion probability: HIGH'
            guidance_color = '#27AE60'
        elif final_similarity > 0.95:
            guidance = 'CAUTION: Additional mutations recommended\nDetection risk: MODERATE'
            guidance_color = '#E67E22'
        else:
            guidance = 'ALERT: Evasion unsuccessful\nAction: Try alternative mutation strategy'
            guidance_color = '#E74C3C'
        
        ax2.text(0.05, y_pos, guidance, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', alpha=0.9, edgecolor=guidance_color, linewidth=2))
        
        # Main title
        fig.suptitle('Real Binary Embedding Drift Analysis | Red Team Intelligence',
                    fontsize=15, fontweight='bold', y=0.98, color='#8B0000')
        
        # Save
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(save_path, dpi=120, bbox_inches='tight', facecolor='white')
        print(f"\n??? Red team drift visualization saved to: {save_path}")
        
        plt.show()

def main():
    """Main execution with command-line support."""
    parser = argparse.ArgumentParser(
        description='Binary Analyzer with Real Embedding Drift Tracking',
        epilog='Example: python3.11 complete_binary_analyzer_with_drift.py --binary split --mutations 10'
    )
    
    parser.add_argument(
        '--binary', '-b',
        type=str,
        required=True,
        help='Path to binary file to analyze'
    )
    
    parser.add_argument(
        '--mutations', '-m',
        type=int,
        default=8,
        help='Number of mutation steps (default: 8)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='embedding_drift_real.png',
        help='Output visualization filename'
    )
    
    parser.add_argument(
        '--intensity',
        type=str,
        choices=['conservative', 'moderate', 'aggressive'],
        default='conservative',
        help='Mutation intensity: conservative (default, preserves similarity), moderate, or aggressive'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='microsoft/codebert-base',
        help='HuggingFace transformer model (default: microsoft/codebert-base)'
    )
    
    args = parser.parse_args()
    
    print("???? BINARY ANALYZER WITH REAL EMBEDDING DRIFT")
    print("="*70)
    print(f"???? Binary: {args.binary}")
    print(f"???? Mutations: {args.mutations} steps (intensity: {args.intensity})")
    print(f"???? Model: {args.model}")
    print()
    
    # Check binary exists
    if not os.path.isfile(args.binary):
        print(f"??? Error: Binary '{args.binary}' not found")
        return
    
    # Initialize analyzer with specified model
    analyzer = BinaryAnalyzerWithDrift(args.model)
    
    # Perform drift analysis with specified intensity
    results = analyzer.analyze_with_drift(args.binary, args.mutations, intensity=args.intensity)
    
    if results:
        # Visualize drift
        binary_name = os.path.basename(args.binary)
        analyzer.visualize_drift(results, binary_name, args.output)
        
        print("\n???? Analysis Summary:")
        print(f"   ??? Original size: {results['original_size']} bytes")
        print(f"   ??? Final size: {len(results['binary_versions'][-1])} bytes")
        print(f"   ??? Embedding points: {len(results['embeddings'])}")
        print(f"   ??? Visualization: {args.output}")
        
        # Calculate drift metrics
        orig_emb = results['embeddings'][0]
        final_emb = results['embeddings'][-1]
        
        l2_distance = np.linalg.norm(orig_emb - final_emb)
        cosine_sim = cosine_similarity(
            orig_emb.reshape(1, -1),
            final_emb.reshape(1, -1)
        )[0][0]
        
        print(f"\n???? Drift Metrics:")
        print(f"   ??? L2 Distance: {l2_distance:.4f}")
        print(f"   ??? Cosine Similarity: {cosine_sim:.4f}")
        print(f"   ??? Drift Magnitude: {'Significant' if l2_distance > 1.0 else 'Moderate' if l2_distance > 0.5 else 'Minimal'}")
        
        print("\n??? Real binary embedding drift analysis complete!")

if __name__ == "__main__":
    main()

