from manim import *
import numpy as np
import config
from scipy import signal as scipy_signal


class PipeToWelchTransition(Scene):
    """
    Scene demonstrating the transition from physical pipe waves to Welch's Method analysis.
    Part 1: Shows sound waves in a pipe (physical process)
    Part 2: Fades pipe and shows mathematical analysis with proper graphs
    """
    def construct(self):
        # ========== PART 1: PHYSICAL PIPE WITH WAVES ==========
        intro_text = Title("Sound Waves in a Whistle Pipe")
        self.add(intro_text)
        self.wait(0.5)
        
        # Draw simplified 2D pipe
        pipe_body = Rectangle(width=6, height=1.5, color=BLUE_B, fill_color=BLUE_A, fill_opacity=0.3)
        pipe_body.shift(DOWN * 0.5)
        
        pipe_label = Text("Whistle Pipe", font_size=12, color=WHITE).next_to(pipe_body, UP, buff=0.3)
        
        self.play(Create(pipe_body))
        self.play(Write(pipe_label))
        self.wait(0.7)
        
        # Draw sound wave representation in pipe
        wave_lines = VGroup()
        wave_tails = np.linspace(-2.8, 2.8, 20)
        
        for x_pos in wave_tails:
            # Create simple sine wave representation
            wave_points = []
            for y_val in np.linspace(-0.7, 0.7, 30):
                x = x_pos + 0.15 * np.sin(3 * y_val)
                wave_points.append([x, y_val - 0.5, 0])
            
            wave_line = VMobject()
            wave_line.set_points_as_corners(wave_points)
            wave_line.set_stroke(YELLOW, width=1, opacity=0.6)
            wave_lines.add(wave_line)
        
        self.play(Create(wave_lines), run_time=2)
        
        # Add frequency indicators
        freq_text = VGroup(
            Text("50 Hz", font_size=11, color=RED).shift(LEFT * 2.5 + DOWN * 2),
            Text("120 Hz", font_size=11, color=GREEN).shift(DOWN * 2),
            Text("200 Hz", font_size=11, color=BLUE).shift(RIGHT * 2.5 + DOWN * 2),
        )
        self.play(Write(freq_text))
        self.wait(1.5)
        
        # Fade out the pipe visualization
        pipe_group = VGroup(intro_text, pipe_body, pipe_label, wave_lines, freq_text)
        self.play(FadeOut(pipe_group), run_time=1.5)
        self.wait(0.5)
        
        # ========== PART 2: WELCH'S METHOD WITH PROPER GRAPHS ==========
        
        # Generate synthetic signal (multi-frequency wave in a pipe)
        fs = config.fs      # Sampling frequency (1000 Hz)
        duration = 2        # seconds
        t = np.linspace(0, duration, fs * duration, endpoint=False)
        
        # Multi-frequency signal + noise
        signal_data = (1.5 * np.sin(2 * np.pi * 50 * t) +      # 50 Hz
                       1.0 * np.sin(2 * np.pi * 120 * t) +     # 120 Hz
                       0.5 * np.sin(2 * np.pi * 200 * t) +     # 200 Hz
                       0.3 * np.random.randn(len(t)))           # Noise
        
        # Welch's method parameters
        nperseg = config.welch_npserseg  # Segment length
        noverlap = int(nperseg * 0.5)    # 50% overlap
        nfft = nperseg
        
        # ===== STEP 1: Time-Domain Signal =====
        step1_title = Text("Step 1: Time-Domain Signal", font_size=14, weight=BOLD).to_edge(UP)
        self.add(step1_title)
        self.wait(0.3)
        
        # Create proper axes for time signal
        time_ax = Axes(
            x_range=[0, 2, 0.5],
            y_range=[-3, 3, 1],
            axis_config={
                "color": LIGHT_GREY,
                "stroke_width": 1.5,
                "include_ticks": True,
            },
            tips=True,
        )
        time_ax.set_height(3.5)
        time_ax.shift(DOWN * 0.5)
        
        # Add labels to axes
        time_x_label = time_ax.get_x_axis_label("Time (s)", edge=DOWN, direction=DOWN)
        time_y_label = time_ax.get_y_axis_label("Amplitude (V)", edge=LEFT, direction=LEFT)
        
        # Plot the signal
        time_curve = time_ax.plot_line_graph(
            x_values=t[::5],  # Sample every 5th point for clarity
            y_values=signal_data[::5],
            line_color=YELLOW,
            vertex_dot_radius=0.01,
        )
        
        self.play(Create(time_ax), Write(time_x_label), Write(time_y_label), run_time=1)
        self.play(Create(time_curve), run_time=2)
        self.wait(1)
        
        # ===== STEP 2: Segmentation =====
        step2_title = Text("Step 2: Signal Segmentation (50% Overlap)", font_size=14, weight=BOLD).to_edge(UP)
        self.play(Transform(step1_title, step2_title))
        self.wait(0.3)
        
        # Show 3 segments with different colors
        seg_colors = [RED, GREEN, BLUE]
        num_segs_show = 3
        
        segment_markers = VGroup()
        for seg_idx in range(num_segs_show):
            start_idx = seg_idx * (nperseg - noverlap)
            end_idx = start_idx + nperseg
            end_time = end_idx / fs
            start_time = start_idx / fs
            
            # Create vertical lines to mark segments
            seg_line = time_ax.get_vertical_line(time_ax.coords_to_point(start_time, 0), color=seg_colors[seg_idx], stroke_width=2)
            segment_markers.add(seg_line)
        
        self.play(Create(segment_markers), run_time=1)
        seg_info = VGroup(
            Text("Segment 1", font_size=10, color=RED),
            Text("Segment 2", font_size=10, color=GREEN),
            Text("Segment 3", font_size=10, color=BLUE),
        ).arrange(RIGHT, buff=0.4).shift(DOWN * 3.2)
        self.play(Write(seg_info))
        self.wait(1)
        
        # ===== STEP 3: Windowing =====
        step3_title = Text("Step 3: Apply Hann Window", font_size=14, weight=BOLD).to_edge(UP)
        self.play(Transform(step1_title, step3_title))
        self.wait(0.3)
        
        self.play(FadeOut(segment_markers), FadeOut(seg_info), FadeOut(time_curve))
        
        # Create window function plot
        window_ax = Axes(
            x_range=[0, nperseg, nperseg/5],
            y_range=[0, 1.1, 0.2],
            axis_config={
                "color": LIGHT_GREY,
                "stroke_width": 1.5,
                "include_ticks": True,
            },
            tips=True,
        )
        window_ax.set_height(3.5)
        window_ax.shift(DOWN * 0.5)
        
        # Generate Hann window
        window_data = scipy_signal.get_window('hann', nperseg)
        window_samples = np.arange(nperseg)
        
        window_curve = window_ax.plot_line_graph(
            x_values=window_samples[::10],
            y_values=window_data[::10],
            line_color=PURPLE,
            vertex_dot_radius=0.02,
        )
        
        window_x_label = window_ax.get_x_axis_label("Sample Index", edge=DOWN, direction=DOWN)
        window_y_label = window_ax.get_y_axis_label("Window Value", edge=LEFT, direction=LEFT)
        
        self.play(FadeOut(time_ax), FadeOut(time_x_label), FadeOut(time_y_label))
        self.play(Create(window_ax), Write(window_x_label), Write(window_y_label), run_time=1)
        self.play(Create(window_curve), run_time=1.5)
        
        window_desc = Text("Hann window reduces spectral leakage", font_size=11, color=LIGHT_GREY).shift(DOWN * 3.2)
        self.play(Write(window_desc))
        self.wait(1.5)
        
        # ===== STEP 4: FFT Computation =====
        step4_title = Text("Step 4: FFT for Each Segment", font_size=14, weight=BOLD).to_edge(UP)
        self.play(Transform(step1_title, step4_title))
        self.wait(0.3)
        
        self.play(FadeOut(window_ax), FadeOut(window_x_label), FadeOut(window_y_label), FadeOut(window_curve), FadeOut(window_desc))
        
        # Create FFT plots for 3 segments side by side
        fft_plots = VGroup()
        
        for seg_idx in range(num_segs_show):
            start_idx = seg_idx * (nperseg - noverlap)
            end_idx = start_idx + nperseg
            
            # Extract and window segment
            segment = signal_data[start_idx:end_idx] * window_data
            
            # Compute FFT
            fft_result = np.fft.fft(segment, n=nfft)
            psd_segment = np.abs(fft_result[:nfft//2])**2 / fs
            freqs = np.fft.fftfreq(nfft, 1/fs)[:nfft//2]
            
            # Create axes for this segment
            seg_ax = Axes(
                x_range=[0, 300, 50],
                y_range=[0, np.max(psd_segment[:150])*1.2, np.max(psd_segment[:150])/4],
                axis_config={
                    "color": LIGHT_GREY,
                    "stroke_width": 1,
                    "include_ticks": True,
                },
                tips=False,
            )
            seg_ax.set_height(2.5)
            seg_ax.set_width(2)
            
            # Position axes
            x_pos = -4.5 + seg_idx * 3
            seg_ax.shift([x_pos, -1, 0])
            
            # Plot FFT magnitude
            fft_line = seg_ax.plot_line_graph(
                x_values=freqs[:150:10],
                y_values=psd_segment[:150:10],
                line_color=seg_colors[seg_idx],
                vertex_dot_radius=0.01,
            )
            
            # Add labels
            seg_label = Text(f"Seg {seg_idx+1}", font_size=9, color=seg_colors[seg_idx]).next_to(seg_ax, DOWN, buff=0.15)
            
            fft_plots.add(seg_ax, fft_line, seg_label)
        
        # Add axis labels (shared)
        fft_x_label = Text("Frequency (Hz)", font_size=9).shift(DOWN * 3.2)
        fft_y_label = Text("Power (V²/Hz)", font_size=9).shift(LEFT * 5 + DOWN * 1)
        
        self.play(Create(fft_plots), Write(fft_x_label), Write(fft_y_label), run_time=2)
        self.wait(1.5)
        
        # ===== STEP 5: Final Averaged Spectrum =====
        step5_title = Text("Step 5: Average Spectra → Final PSD (Welch's Method)", font_size=14, weight=BOLD).to_edge(UP)
        self.play(Transform(step1_title, step5_title))
        self.wait(0.3)
        
        self.play(FadeOut(fft_plots), FadeOut(fft_x_label), FadeOut(fft_y_label))
        
        # Compute Welch PSD using scipy
        freqs_welch, psd_welch = scipy_signal.welch(
            signal_data, fs, window='hann', nperseg=nperseg, 
            noverlap=noverlap, nfft=nfft, scaling='density'
        )
        
        # Create final plot with proper axes
        final_ax = Axes(
            x_range=[0, 300, 50],
            y_range=[0, np.max(psd_welch[:100])*1.2, np.max(psd_welch[:100])/3],
            axis_config={
                "color": LIGHT_GREY,
                "stroke_width": 1.5,
                "include_ticks": True,
            },
            tips=True,
        )
        final_ax.set_height(3.5)
        final_ax.set_width(6)
        final_ax.shift(DOWN * 0.3)
        
        # Plot final spectrum
        final_curve = final_ax.plot_line_graph(
            x_values=freqs_welch[1:150:5],  # Skip DC component
            y_values=psd_welch[1:150:5],
            line_color=BLUE_D,
            vertex_dot_radius=0.02,
        )
        
        # Add frequency peak annotations
        peak_freqs = [50, 120, 200]
        peak_annotations = VGroup()
        for peak_f in peak_freqs:
            peak_idx = np.argmin(np.abs(freqs_welch - peak_f))
            if peak_idx < len(psd_welch):
                point = final_ax.coords_to_point(freqs_welch[peak_idx], psd_welch[peak_idx])
                dot = Dot(point, color=RED, radius=0.08)
                label = Text(f"{peak_f} Hz", font_size=9, color=RED).next_to(dot, UP, buff=0.15)
                peak_annotations.add(dot, label)
        
        final_x_label = final_ax.get_x_axis_label("Frequency (Hz)", edge=DOWN, direction=DOWN)
        final_y_label = final_ax.get_y_axis_label("Power (V²/Hz)", edge=LEFT, direction=LEFT)
        
        self.play(Create(final_ax), Write(final_x_label), Write(final_y_label), run_time=1)
        self.play(Create(final_curve), run_time=2)
        self.play(Create(peak_annotations), run_time=1)
        self.wait(1)
        
        # Summary
        summary = VGroup(
            Text("Welch's Method Results:", font_size=12, weight=BOLD),
            Text("✓ Identified 3 frequency components", font_size=10),
            Text("✓ Reduced variance through averaging", font_size=10),
            Text("✓ Improved spectral resolution", font_size=10),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).shift(UP * 2.5 + RIGHT * 2.5)
        
        self.play(Write(summary))
        self.wait(3)
