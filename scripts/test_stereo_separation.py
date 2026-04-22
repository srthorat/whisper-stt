#!/usr/bin/env python3
"""Test stereo channel separation quality"""
import wave
import numpy as np

def analyze_stereo_separation(wav_path: str):
    """Analyze stereo channel separation and quality"""
    with wave.open(wav_path, 'rb') as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        sample_width = wf.getsampwidth()
        n_frames = wf.getnframes()
        
        print(f"=== Audio File Info ===")
        print(f"File: {wav_path}")
        print(f"Channels: {channels}")
        print(f"Sample Rate: {sample_rate} Hz")
        print(f"Sample Width: {sample_width} bytes")
        print(f"Duration: {n_frames / sample_rate:.2f} seconds")
        print(f"Total Frames: {n_frames}")
        
        if channels != 2:
            print(f"\n⚠️  Not a stereo file (channels={channels})")
            return
        
        # Read all audio data
        raw_data = wf.readframes(n_frames)
        samples = np.frombuffer(raw_data, dtype=np.int16)
        
        print(f"\n=== Raw Data ===")
        print(f"Total samples: {len(samples)}")
        print(f"Expected samples: {n_frames * channels}")
        
        # Current implementation (from app/main.py)
        if samples.size % channels != 0:
            samples = samples[: samples.size - (samples.size % channels)]
        
        stereo = samples.reshape(-1, channels).astype(np.float32) / 32768.0
        left_channel = stereo[:, 0]
        right_channel = stereo[:, 1]
        
        print(f"\n=== Channel Separation ===")
        print(f"Left channel samples: {len(left_channel)}")
        print(f"Right channel samples: {len(right_channel)}")
        
        # Analyze channel content
        left_rms = np.sqrt(np.mean(np.square(left_channel)))
        right_rms = np.sqrt(np.mean(np.square(right_channel)))
        left_peak = np.max(np.abs(left_channel))
        right_peak = np.max(np.abs(right_channel))
        
        print(f"\n=== Channel Statistics ===")
        print(f"Left Channel:")
        print(f"  RMS: {left_rms:.4f}")
        print(f"  Peak: {left_peak:.4f}")
        print(f"  Dynamic Range: {20 * np.log10(left_peak / (left_rms + 1e-10)):.2f} dB")
        
        print(f"\nRight Channel:")
        print(f"  RMS: {right_rms:.4f}")
        print(f"  Peak: {right_peak:.4f}")
        print(f"  Dynamic Range: {20 * np.log10(right_peak / (right_rms + 1e-10)):.2f} dB")
        
        # Cross-correlation analysis
        correlation = np.corrcoef(left_channel, right_channel)[0, 1]
        print(f"\n=== Channel Correlation ===")
        print(f"Correlation coefficient: {correlation:.4f}")
        
        if correlation > 0.9:
            print("⚠️  HIGH correlation - channels may be very similar or mono copied to stereo")
        elif correlation < 0.3:
            print("✅ LOW correlation - good stereo separation")
        else:
            print("ℹ️  MODERATE correlation - typical stereo content")
        
        # Energy balance
        balance = left_rms / (right_rms + 1e-10)
        print(f"\n=== Channel Balance ===")
        print(f"Left/Right energy ratio: {balance:.2f}")
        
        if 0.5 < balance < 2.0:
            print("✅ Balanced channels")
        else:
            print(f"⚠️  Unbalanced - one channel significantly louder")
        
        # Test first 100 samples
        print(f"\n=== Sample Data (first 5 frames) ===")
        for i in range(min(5, len(left_channel))):
            print(f"Frame {i}: L={left_channel[i]:7.4f}, R={right_channel[i]:7.4f}")
        
        # Verify implementation correctness
        print(f"\n=== Implementation Verification ===")
        
        # Alternative method using numpy audio tools
        samples_alt = np.frombuffer(raw_data, dtype=np.int16).reshape(-1, 2)
        left_alt = samples_alt[:, 0].astype(np.float32) / 32768.0
        right_alt = samples_alt[:, 1].astype(np.float32) / 32768.0
        
        left_match = np.allclose(left_channel, left_alt)
        right_match = np.allclose(right_channel, right_alt)
        
        print(f"Left channel matches alternative method: {left_match}")
        print(f"Right channel matches alternative method: {right_match}")
        
        if left_match and right_match:
            print("✅ Current implementation is CORRECT")
        else:
            print("❌ Current implementation has ERRORS")
        
        return {
            'left': left_channel,
            'right': right_channel,
            'left_rms': left_rms,
            'right_rms': right_rms,
            'correlation': correlation,
            'balance': balance
        }

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python test_stereo_separation.py <wav_file>")
        sys.exit(1)
    
    analyze_stereo_separation(sys.argv[1])
