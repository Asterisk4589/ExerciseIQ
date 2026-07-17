# graphs.py
"""
Workout graphing utilities for ExerciseIQ.
Plots knee angle timelines, highlights bad posture intervals, and visualizes rep completions.
"""

import matplotlib.pyplot as plt

def generate_session_graph(state):
    """
    Constructs a visual graph detailing knee joint angles over time.
    Draws threshold lines and shades bad form and rep completion regions.
    Saves the final plot as 'session_graph.png'.
    """
    if len(state.angle_history) == 0:
        return
        
    plt.figure(figsize=(12, 5))
    plt.style.use('dark_background')
    
    # Plot smooth knee joint angle history line
    plt.plot(state.angle_history, color='#00FF88', linewidth=1.5, label='Knee Angle')
    
    # Shade bad form intervals (knee over toe) red
    for f in state.bad_form_frames:
        plt.axvspan(f - 2, f + 2, color='red', alpha=0.3)
    
    # Mark frames where rep depth was successfully achieved with cyan dashed vertical lines
    for f in state.rep_frames:
        plt.axvline(x=f, color='cyan', linewidth=1, linestyle='--', alpha=0.7)
    
    # Static threshold limit lines
    plt.axhline(y=90, color='yellow', linewidth=1, linestyle=':', label='Squat Depth (90°)')
    plt.axhline(y=160, color='orange', linewidth=1, linestyle=':', label='Standing (160°)')
    
    plt.title('ExerciseIQ — Knee Angle Timeline', color='white', fontsize=14)
    plt.xlabel('Frame', color='white')
    plt.ylabel('Knee Angle (degrees)', color='white')
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    # Save timeline figure as high quality PNG
    plt.savefig('session_graph.png', dpi=150)
    plt.show()
    print("Graph saved as session_graph.png")
